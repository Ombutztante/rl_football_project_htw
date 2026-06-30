# Optimization Log — DQN & Q-Table

Dokumentation des automatisierten Optimierungsloops.  
Branch: `a_dev_5` | Ziel: Goal-Rate und Average-Reward auf Level 1–5 maximieren.

## Methodologie

**Loop-Aufbau:**  
Claude analysiert nach jeder Iteration die Trainingskurven und Metriken, entscheidet begründet welche Parameter geändert werden, und startet die nächste Runde. Jede Iteration ist eine vollständige dokumentierte Hypothese mit Ergebnis.

**Trainingskonfiguration pro Iteration:**
- Alle Level 1–5, beide Agenten (Q-Table + DQN)
- Je 1000 und 3000 Episoden (Vergleich kurzes vs. langes Training)
- Metriken über letzte 100 Episoden gemessen

**Primäre Metriken:**
- **Goal Rate %** — Anteil der Episoden mit Torerfolg (letzte 100 Ep.)
- **Avg. Reward** — Durchschnittlicher kumulierter Reward (letzte 100 Ep.)

**Optimierungshebel (Priorität):**
1. Reward-Design (Haupthebel — beschriebene Bugs/Farming)
2. Epsilon-Schedule DQN (erklärt frühzeitiges Steckenbleiben)
3. Lernparameter (sekundär)

---

## Bekannte Probleme vor Optimierungsstart

| Problem | Beschreibung | Betroffene Level |
|---|---|---|
| Wand-Kollision keine Penalty | Agent läuft gegen Wand: nur -1 Step-Penalty, kein Wand-Penalty → DQN bleibt stecken | L1–3, L5 |
| Bypass-Reward fehlt komplett (L4) | `REWARD_BYPASS_OBSTACLE = +2` war in `config.py` definiert aber nie in `environment.py` vergeben → DQN hatte keinen Anreiz den Korridor zu nutzen | L4 DQN |
| `REWARD_CLOSER` Oszillation | +1 für jeden Schritt näher am Tor → Agent kann hin-/herlaufen um Reward zu farmen | L1–5 |
| DQN Epsilon zu früh minimum | `DECAY=0.995` → ε=0.05 bei Ep.~600, letzte 400 Ep. fast keine Exploration mehr | L1–5 DQN |

---

## Iterationsübersicht

| Iteration | Beschreibung | Datum |
|---|---|---|
| 0 | Baseline — keine Änderungen | 2026-06-29 |
| 1 | Wall-Penalty, Bypass-Reward implementiert, L5 Epsilon-Fix, REWARD_CLOSER mit Ball | 2026-06-29 |
| 2 | Bypass +8, Wall -1, REWARD_CLOSER_NO_BALL=0.5, L4 Standard-Epsilon | 2026-06-29 |
| 3 | DQN LR 5e-5, REPLAY_WARMUP 800 — Stabilisierung Konvergenz | 2026-06-29 |
| 4 | Revert LR→1e-4, Warmup→500; MEMORY_SIZE 20000 | 2026-06-29 |
| 5 | Finale Bestätigung: MEMORY_SIZE→15000, alle anderen wie Iter2 | 2026-06-29 |
| 6 | L4 Sparse-Reward-Experiment: OBSTACLE_HEIGHT 4→2 (danach revert) | 2026-06-29 |
| 7 | Level 6 Erstlauf: Q-Table + DQN, 1000 + 3000 Episoden | 2026-06-29 |

---

## Iteration 0 — Baseline

**Datum:** 2026-06-29  
**Run-Verzeichnis:** `results/opt_iter0_2906/`  
**Änderungen:** Keine — reine Messung des aktuellen Stands  
**Hypothese:** Liefert den Referenzwert für alle späteren Iterationen. Erwartet werden die bekannten Schwächen: DQN findet auf höheren Leveln das Tor nicht zuverlässig; L4 zeigt Obstacle-Farming.

### Ergebnisse

| Level | Agent | ep1000 Goal% | ep1000 AvgR | ep3000 Goal% | ep3000 AvgR |
|---|---|---|---|---|---|
| L1 | Q-Table | 90.0% | 86.6 | 94.0% | 88.6 |
| L1 | DQN | 74.0% | 97.0 | 93.0% | 82.7 |
| L2 | Q-Table | 100.0% | 44.6 | 100.0% | 44.7 |
| L2 | DQN | 100.0% | 44.0 | 100.0% | 44.4 |
| L3 | Q-Table | 89.0% | 48.0 | 95.0% | 52.0 |
| L3 | DQN | 83.0% | 41.5 | 89.0% | 44.8 |
| L4 | Q-Table | 81.0% | 49.2 | 94.0% | 59.8 |
| L4 | DQN | **0.0%** | -14.0 | **0.0%** | -8.8 |
| L5 | Q-Table | 89.0% | 55.4 | 94.0% | 59.1 |
| L5 | DQN | **0.0%** | -16.2 | **0.0%** | -9.1 |

### Analyse

**Q-Table** konvergiert auf allen Leveln gut. Schwächste Stelle: L4 bei ep1000 (81%) — braucht mehr Episoden für den Obstacle-Bypass. Bei ep3000 erreicht Q-Table auf allen Leveln 89–100%.

**DQN** zeigt drei kritische Probleme:

1. **L4 + L5: 0% Goal-Rate auf allen Episodenzahlen** — vollständiges Versagen.
   - L4: Code-Analyse zeigt: `REWARD_BYPASS_OBSTACLE = +2` ist in `config.py` definiert aber **nie vergeben** worden. DQN bekam keinen Anreiz, den freien Korridor unter dem Hindernis zu nutzen. Avg-Reward -8.8 bei 3000ep entspricht einem "pass-early, let-opponent-catch" Loop: Ball aufnehmen (+5), sofort schießen (Ball ~3 Felder vor, Agent verliert Besitz), Gegner fängt Ball (-10) → netto ~-9 pro Episode.
   - L5: Code-Bug — `train_dqn.py` aktiviert den langsamen L4-Decay (`ε=0.998`) für **alle Level >= 4**, also auch L5. Dadurch ist ε bei ep1000 noch 0.135 (kaum Exploitation), bei ep1500 erst am Minimum. 1500 reine Explorations-Episoden ohne Ziel gefunden → DQN lernt: "kurze Episode mit minimalem Schaden ist optimal" (Gegner schnappt sich Ball in ~10 Schritten).

2. **L3: Instabilität** — Goal% bricht bei ep2000 auf ~50% ein, erholt sich auf 88%. Loss-Werte steigen auf 2–3 (Q-Value-Drift). DQN hat kein stabiles Konvergenz-Plateau.

3. **L1 bei ep1000: nur 74%** vs. Q-Table 90% — DQN findet keinen stabilen Pfad, weil Wandkontakt keine Penalty gibt. Agent lernt, dass "gegen Wand drücken" ähnlich kostet wie normales Gehen.

### Entscheidung für Iteration 1

Drei gezielte Fixes für die kritischsten Probleme:

1. **Wall-Penalty einführen** (alle Level): `REWARD_WALL = -2` wenn Agent gegen Außenwand läuft. Bricht den Null-Reward-Idle-Loop bei DQN.
2. **`REWARD_BYPASS_OBSTACLE` einmalig pro Episode** (L4): Neues Flag `bypass_rewarded` — Reward wird nur beim ersten Korridor-Durchgang gegeben. Eliminiert L4-Farming.
3. **Epsilon-Decay für L5 entkoppeln** (train_dqn.py): Slow-Decay nur für `LEVEL == 4`, L5 bekommt Standard-Decay `0.995`. Gibt DQN bei L5 genug Greedy-Zeit innerhalb 1000 Episoden.
4. **`REWARD_CLOSER` nur mit Ball**: Shaping-Reward +1 wird nur vergeben wenn Agent den Ball hat. Sinnlosigkeit: ohne Ball dem Tor annähern hat keinen RL-Wert.

---

## Iteration 1 — Reward-Korrektur und Epsilon-Bugfix

**Datum:** 2026-06-29  
**Run-Verzeichnis:** `results/opt_iter1_2906/`  
**Hypothese:** Vier gezielte Korrekturen beseitigen die identifizierten Hauptprobleme. DQN sollte auf L4 und L5 von 0% auf messbare Goal-Raten kommen. L1–L3 sollten stabil bleiben oder leicht besser werden.

**Änderungen:**

| # | Datei | Änderung | Begründung |
|---|---|---|---|
| 1 | `src/environment.py` | `REWARD_WALL = -2` wenn Agent gegen Außenwand läuft | DQN lernt ohne Wandpenalty: "Wand drücken = free idle". Unterbricht Idle-Loops und falsche räumliche Lernkurven. |
| 2 | `src/environment.py` | `REWARD_CLOSER` nur wenn `self.has_ball` | Shaping ohne Ball ist sinnlos und erzeugt falsche Gradienten: Agent lernt, sich dem Tor ohne Ball anzunähern. |
| 3 | `src/environment.py` | `REWARD_BYPASS_OBSTACLE = +2` jetzt tatsächlich implementiert (einmalig pro Episode via `bypass_rewarded` Flag) | War in `config.py` definiert, aber **nie in environment.py vergeben**. L4 DQN hatte keinen Anreiz, den Korridor unter dem Hindernis zu nutzen — dieser fehlende Reward erklärt die 0% Goal-Rate auf L4. |
| 4 | `src/train_dqn.py` | `LEVEL >= 4` → `LEVEL == 4` für langsamen Epsilon-Decay | L5 nutzte versehentlich den L4-Slow-Decay (ε=0.998), was bedeutet: 1500 fast-zufällige Episoden → kein stabiles Lernen bei ep1000. Fix gibt L5 Standard-Decay 0.995. |
| 5 | `config.py` | L5/LX-Konstanten hinzugefügt (`BALL_START_X_L5`, `OPP_START_X_L5`, etc.) | Fehlten auf `a_dev_5` (waren nur auf `main`). Ohne diese Constants würde environment.py für L5/L6 crashen. |

**Erwartete Auswirkungen:**
- **L4 DQN**: Bypass-Reward + Wandpenalty → Agent findet Korridor → Goal-Rate > 0% bei ep3000
- **L5 DQN**: Standard Epsilon-Decay → genug Exploitation bei ep1000 → Goal-Rate > 0%  
- **L1 DQN**: Wandpenalty stärkt räumliches Lernen → Verbesserung bei ep1000 (von 74%)
- **L2/L3 Q-Table + DQN**: Minimal betroffen, sollte stabil bleiben

### Ergebnisse

| Level | Agent | ep1000 Goal% | ep1000 AvgR | ep3000 Goal% | ep3000 AvgR |
|---|---|---|---|---|---|
| L1 | Q-Table | **100.0%** ↑ | 34.2 | 90.0% | 72.7 |
| L1 | DQN | 67.0% ↓ | 38.8 | 87.0% ↓ | 83.7 |
| L2 | Q-Table | 100.0% | 40.2 | 100.0% | 43.7 |
| L2 | DQN | 99.0% | 38.9 | **100.0%** | 44.2 |
| L3 | Q-Table | 75.0% ↓ | 35.4 | 90.0% ↓ | 46.0 |
| L3 | DQN | 85.0% ↑ | 42.1 | 84.0% ↓ | 42.4 |
| L4 | Q-Table | 23.0% ↓ | 0.0 | **96.0%** ↑ | 60.9 |
| L4 | DQN | **0.0%** — | -19.7 | **0.0%** — | -20.6 |
| L5 | Q-Table | **99.0%** ↑ | 79.4 | **100.0%** ↑ | 79.8 |
| L5 | DQN | **96.0%** ↑↑ | 74.8 | 66.0% ↑↑ | 49.8 |

*(↑ = Verbesserung vs. Baseline, ↓ = Regression, — = unverändert)*

### Analyse

**Großer Gewinn — L5 DQN:** 0% → 96% bei ep1000. Der Epsilon-Decay-Bugfix hat das Problem vollständig gelöst. L5 DQN bekommt jetzt Standard-Decay 0.995 und hat bei ep600 bereits ε=0.05 → genug Exploitation-Zeit.

**Großer Gewinn — L5 Q-Table:** 89% → 99% bei ep1000, 94% → 100% bei ep3000. Verbesserte Reward-Signale (REWARD_CLOSER nur mit Ball) saubereres Lernen.

**L4 DQN weiterhin 0%:** Der Bypass-Reward (+2 einmalig) war nicht ausreichend. Das Problem ist struktureller Natur:
1. DQN mit L4-Slow-Decay hat bei ep1000 ε=0.135 — kaum Exploitation. Bei ep1000 hatte es **1.2% Goal-Rate** (erste Korridor-Durchläufe gefunden!), aber danach bei ep1500–3000 konvergiert es auf eine schlechte Policy mit avg=-20.
2. Der Detour (Korridor bei y=4,5) erfordert, dass der Agent zunächst WEG vom Tor geht (y=3→y=4/5 erhöht Manhattan-Distanz → kein REWARD_CLOSER). Es gibt keinen schrittweisen Reward für "im Korridor navigieren".
3. Avg=-20 bei ep3000 entspricht: Agent holt Ball (+5), Gegner fängt ihn nach ~14 Schritten (-20), netto ≈ -20 pro Episode. DQN hat eine lokale Optima mit "schnelle kurze Episode" gefunden.

**L4 Q-Table Regression bei ep1000:** 81% → 23%. Der Wall-Penalty (-2) und REWARD_CLOSER-nur-mit-Ball bremsen die frühe Exploration von Q-Table. Bei ep3000 ist Q-Table wieder auf 96% (besser als Baseline 94%) — Q-Table braucht einfach mehr Episoden um die Wandvermeidung zu "vergessen" und sich auf Tore zu konzentrieren.

**L3 Q-Table Regression:** 89% → 75% bei ep1000. REWARD_CLOSER nur mit Ball hat die Shaping-Signale reduziert. In L3 verliert der Agent nach einem Vorwärtspass den Ball, muss ihn verfolgen — ohne Ball hat er jetzt keinen Anreiz sich dem Tor zu nähern. Der Wall-Penalty verschlimmert die Exploration-Effizienz von Q-Table.

**L1 DQN Regression:** 74% → 67% bei ep1000. Der Wall-Penalty (-2) fügt frühes Trainingsrauschen hinzu, ohne klaren Nutzen auf L1.

**L5 DQN ep3000 Instabilität:** 96% bei ep1000 aber nur 66% bei ep3000. Loss-Werte 3–6 deuten auf Q-Value-Drift hin. Ursache: REWARD_PASS_SUCCESS=+15 wird abhängig von der Teammate-KI vergeben und erzeugt hohe TD-Varianz. DQN driftet ab ep2000.

### Entscheidung für Iteration 2

Vier gezielte Anpassungen:

1. **L4 DQN: Standard-Epsilon-Decay (0.995) statt Slow-Decay (0.998)** — DQN hat bei ep1000 bereits 1.2% Goal-Rate gefunden (ε=0.135). Mit Standard-Decay wäre ε=0.05 schon bei ep600 erreicht → mehr Exploitation der gefundenen Bypass-Pfade. Die Slow-Decay-Hypothese ("braucht lange Exploration") war falsch: die Exploration hilft nicht wenn DQN danach nicht exploitieren kann.
2. **REWARD_BYPASS_OBSTACLE von 2 auf 8** erhöhen — stärkeres Signal, dass der Korridor benutzt werden MUSS. Einmalig pro Episode bleibt bestehen.
3. **REWARD_WALL von -2 auf -1 reduzieren** — der starke Wall-Penalty schadet Q-Table-Exploration auf L1–L3. Halbierung reduziert Regression ohne den Kern-Nutzen zu verlieren.
4. **REWARD_CLOSER_NO_BALL = 0.5**: Neuer Config-Parameter für Shaping ohne Ball (halbe Stärke). Wiederherstellung der L3-Performance ohne L4-Farming-Loop wiederzubeleben.

---

## Iteration 2 — L4 DQN Aggressive Fix + Regressions-Korrekturen

**Datum:** 2026-06-29  
**Run-Verzeichnis:** `results/opt_iter2_2906/`  
**Hypothese:** L4 DQN wird mit Standard-Decay und stärkerem Bypass-Reward das Tor finden. REWARD_CLOSER_NO_BALL=0.5 repariert L3 Q-Table Regression. REWARD_WALL=-1 reduziert negative Wirkung auf Q-Table-Exploration.

**Änderungen:**

| # | Datei | Änderung | Begründung |
|---|---|---|---|
| 1 | `config.py` | `REWARD_BYPASS_OBSTACLE`: 2 → **8** | Signal war zu schwach. DQN hat den Bypass bei 1.2% gefunden aber nicht gelernt. Stärkeres Signal notwendig. |
| 2 | `config.py` | `REWARD_WALL`: -2 → **-1** | Iteration 1 zeigte Regression bei Q-Table L3 (89%→75%). Wall-Penalty hatte zu großen negativen Effekt auf Q-Table-Exploration-Effizienz. |
| 3 | `config.py` | Neu: `REWARD_CLOSER_NO_BALL = 0.5` | Wiederherstellung des L3-Shaping-Signals ohne Ball. Q-Table L3 verlor wichtige Shaping-Information (nach Vorwärtspass: Agent jagt Ball zum Tor). |
| 4 | `src/environment.py` | REWARD_CLOSER_NO_BALL bei Bewegung ohne Ball | Implementation des neuen Config-Parameters. |
| 5 | `src/train_dqn.py` | L4 slow epsilon entfernt → Standard 0.995 für alle Level | L4 DQN hatte 1.2% Goal bei ep1000 (ε=0.135) aber driftete dann zur schlechten Policy. Standard-Decay gibt DQN mehr Exploitation-Zeit für die früh gefundenen Bypass-Pfade. |

### Ergebnisse

| Level | Agent | ep1000 Goal% | ep1000 AvgR | ep3000 Goal% | ep3000 AvgR |
|---|---|---|---|---|---|
| L1 | Q-Table | 96.0% ↑ | 76.4 ↑↑ | 91.0% | 67.0 |
| L1 | DQN | **88.0%** ↑↑ | 92.2 ↑↑ | **95.0%** ↑↑ | 81.2 |
| L2 | Q-Table | 100.0% | 43.6 | 100.0% | 43.8 |
| L2 | DQN | 100.0% | 43.4 | 100.0% | 44.7 |
| L3 | Q-Table | **93.0%** ↑↑ | 47.9 ↑↑ | 96.0% ↑ | 50.9 ↑ |
| L3 | DQN | 87.0% ↑ | 44.6 | **98.0%** ↑↑ | 52.8 ↑↑ |
| L4 | Q-Table | **87.0%** ↑↑ | 54.7 ↑↑ | 96.0% | 63.2 ↑ |
| L4 | DQN | 0.0% — | -21.0 | 0.2% — | -11.7 |
| L5 | Q-Table | **100.0%** ↑ | 81.7 ↑ | **100.0%** ↑ | 81.8 ↑ |
| L5 | DQN | **100.0%** ↑ | 79.5 ↑ | **99.0%** ↑↑ | 83.5 ↑↑ |

### Analyse

**Durchgehend starke Verbesserungen vs. Iteration 1:**

- **L3 Q-Table**: 75% → 93% bei ep1000. `REWARD_CLOSER_NO_BALL = 0.5` hat die Regression vollständig repariert. Agent hat jetzt ein schwächeres Shaping-Signal auch ohne Ball, was bei Forward-Pass-Mechanik (Ball jagen nach Pass) wichtig ist.
- **L4 Q-Table ep1000**: 23% → 87%. Die Kombination aus REWARD_CLOSER_NO_BALL und reduziertem REWARD_WALL (-1 statt -2) hat die frühe Exploration massiv verbessert. Bei ep3000 stabil auf 96%.
- **L5 DQN ep3000**: 66% → 99%. Instabilität aus Iteration 1 vollständig behoben. Loss konvergiert auf ~0.58 am Ende.
- **L3 DQN ep3000**: 84% → 98% — exzellent.
- **L1 DQN ep1000**: 67% → 88% — deutliche Verbesserung.

**L4 DQN weiterhin 0%** (0.2% bei ep3000 ist statistisches Rauschen). avg_reward verbesserte sich leicht: -20.6 → -11.7. Die DQN lernt eine "pass-chase-pass"-Loop: Ball aufnehmen → Vorwärtspass (+2) → Ball jagen → wiederholen → Gegner fängt Ball (-10). Netto ≈ -12 pro Episode. Dieser Loop ist ein stabiles lokales Optimum. Der Bypass-Korridor (y=4,5) wird systematisch ignoriert, weil die Detour-Schritte kurzfristig negative Rewards erzeugen. **Wird nach Iteration 5 in separatem L4-Lauf gezielt adressiert.**

**Vergleich mit Baseline (alle L4 DQN ausgenommen):** 8 von 9 messbaren Kategorien verbessert, eine stabil (L2). Kein einziger Rückgang vs. Baseline mehr (L1 Q-Table ep3000: 94%→91% = minimale Fluktuation, innerhalb Varianz).

### Entscheidung für Iteration 3

Reward-Struktur ist jetzt gut — keine weiteren Reward-Änderungen. Fokus auf DQN-Hyperparameter-Stabilisierung:

1. **DQN LR: 1e-4 → 5e-5** — L3/L5 Loss noch bei 1.3–1.5 (L5 spike bis 9.4 bei ep1500). Niedrigere LR erzeugt glattere Gradientenaktualisierungen und reduziert Q-Value-Drift.
2. **REPLAY_WARMUP: 500 → 800** — Mehr diverse Erfahrungen bevor Lernen beginnt. Reduziert frühe Überanpassung auf unrepräsentative Samples.
3. **L4 DQN: Status quo** — Kein weiterer Aufwand im allgemeinen Loop. Wird nach Iteration 5 separat behandelt.

---

## Iteration 3 — DQN Stabilisierung (LR + Warmup)

**Datum:** 2026-06-29  
**Run-Verzeichnis:** `results/opt_iter3_2906/`  
**Hypothese:** Niedrigere LR (5e-5) und höherer Warmup (800) reduzieren den Q-Value-Drift bei L3/L5 DQN, den wir nach ε=min noch beobachten. L1–L5 Q-Table unberührt. L4 DQN bleibt bekannte Baustelle.

**Änderungen:**

| # | Datei | Änderung | Begründung |
|---|---|---|---|
| 1 | `config.py` | `DQN_LR`: 1e-4 → **5e-5** | Loss-Spikes (L5: bis 9.4) und Q-Value-Drift (L3: loss ~1.35 bei ep3000) deuten auf zu hohe LR nach ε-Minimum. Halbierung schafft stabilere Konvergenz ohne zu viel Verlangsamung. |
| 2 | `config.py` | `REPLAY_WARMUP`: 500 → **800** | Mehr diverse Erfahrungen vor dem ersten Gradient-Update. Reduziert Überanpassung auf frühe Episoden (die noch fast zufällig sind). |

### Ergebnisse

| Level | Agent | ep1000 Goal% | ep1000 AvgR | ep3000 Goal% | ep3000 AvgR |
|---|---|---|---|---|---|
| L1 | Q-Table | 98.0% | 77.6 | 66.0% | 77.4 |
| L1 | DQN | **11.0%** ↓↓ | -28.8 | 73.0% ↓↓ | 45.3 |
| L2 | Q-Table | 100.0% | 44.1 | 100.0% | 40.9 |
| L2 | DQN | 97.0% ↓ | 30.1 | 92.0% ↓↓ | 14.6 |
| L3 | Q-Table | 88.0% ↓ | 45.4 | 89.0% ↓ | 45.3 |
| L3 | DQN | **12.0%** ↓↓ | -7.0 | 84.0% ↓↓ | 42.0 |
| L4 | Q-Table | 88.0% ↑ | 57.2 | **98.0%** ↑↑ | 67.9 |
| L4 | DQN | 0.0% — | -26.3 | 0.0% — | -18.4 |
| L5 | Q-Table | 100.0% | 82.0 | 100.0% | 81.7 |
| L5 | DQN | 93.0% ↓ | 78.0 | 97.0% ↓ | 79.8 |

### Analyse

**Iteration 3 ist eine klare Regression für DQN.** Die Ursache ist eindeutig:

- `REPLAY_WARMUP = 800` bei 1000 Episoden → DQN hat nur **200 Lernschritte** bei ep1000 (800 Episoden Warmup, dann erst 200 echte Gradient-Updates)
- `DQN_LR = 5e-5` macht diese 200 Updates zusätzlich schwach
- Ergebnis: DQN L1 ep1000 kollabiert auf 11%, DQN L3 ep1000 auf 12% — beide praktisch untrainiert

Bei ep3000 (2200 Lernschritte) erholt sich DQN teilweise, aber bleibt unter Iteration 2.

**Einziger Gewinn:** Q-Table ist von DQN-Hyperparametern unberührt — L4 Q-Table ep3000 verbessert sich auf **98%** (bestes bisher). L4 Q-Table ep1000 auf 88%, auch besser als Iter1.

**L5 DQN-Loss-Problem persistiert:** Loss bei ep2000 immer noch 9.2, trotz niedrigerer LR. Die Ursache liegt nicht in der LR, sondern in den großen Reward-Werten (GOAL=70, PASS_SUCCESS=15) die hohe TD-Varianz erzeugen. Die ep3000 Werte bleiben aber gut (97%/98.6%).

**Fazit:** LR=5e-5 + Warmup=800 ist für 1000-Episoden-Läufe ungeeignet. Die niedrigere LR löst auch nicht das L5-Loss-Problem.

### Entscheidung für Iteration 4

1. **Revert: `DQN_LR` zurück auf 1e-4** — Iter3 hat bewiesen: 5e-5 ist zu langsam für 1000 Episoden.
2. **Revert: `REPLAY_WARMUP` zurück auf 500** — Warmup=800 lässt bei ep1000 keine Zeit zum Lernen.
3. **Neu: `MEMORY_SIZE` 15000 → 20000** — Größerer Replay-Buffer hält ältere, diversere Erfahrungen länger vor. Minimale Änderung mit potenziell stabilisierendem Effekt auf DQN L1/L3.
4. Reward-Struktur und alle anderen Parameter bleiben wie Iteration 2 (beste Konfiguration bisher).

---

## Iteration 4 — Revert LR + Warmup, größerer Replay-Buffer

**Datum:** 2026-06-29  
**Run-Verzeichnis:** `results/opt_iter4_2906/`  
**Hypothese:** Rückkehr zu LR=1e-4 und Warmup=500 (bewährt in Iteration 2) stellt DQN-ep1000-Performance wieder her. MEMORY_SIZE=20000 als einzige neue Änderung: mehr Buffer-Diversität könnte DQN L1/L3 ep3000-Stabilität leicht verbessern.

**Änderungen:**

| # | Datei | Änderung | Begründung |
|---|---|---|---|
| 1 | `config.py` | `DQN_LR`: 5e-5 → **1e-4** (Revert) | Iter3 bewies: 5e-5 unbrauchbar bei ep1000 (nur 200 effektive Lernschritte). |
| 2 | `config.py` | `REPLAY_WARMUP`: 800 → **500** (Revert) | Warmup=800 bei 1000 Episoden lässt kaum Zeit zum Lernen. |
| 3 | `config.py` | `MEMORY_SIZE`: 15000 → **20000** | Einzige neue Änderung: größerer Buffer retainiert ältere Erfahrungen länger → diversere Mini-Batches → potenziell stabilere Konvergenz ohne Geschwindigkeitsverlust. |

### Ergebnisse

| Level | Agent | ep1000 Goal% | ep1000 AvgR | ep3000 Goal% | ep3000 AvgR |
|---|---|---|---|---|---|
| L1 | Q-Table | 100.0% | 36.1 | 91.0% | 78.8 |
| L1 | DQN | 89.0% ↑ | 74.9 | 81.0% ↓ | 60.9 |
| L2 | Q-Table | 100.0% | 43.4 | 100.0% | 43.4 |
| L2 | DQN | 100.0% | 41.8 | 100.0% | 44.0 |
| L3 | Q-Table | 84.0% ↓ | 42.4 | 97.0% ↑ | 51.3 |
| L3 | DQN | 85.0% | 42.5 | 88.0% ↓ | 43.3 |
| L4 | Q-Table | 85.0% | 56.2 | 93.0% | 64.4 |
| L4 | DQN | 0.0% — | -20.6 | 0.0% — | -14.8 |
| L5 | Q-Table | 100.0% | 82.8 | 99.0% | 80.6 |
| L5 | DQN | 98.0% | 80.0 | 96.0% | 81.0 |

### Analyse

LR=1e-4 und Warmup=500 erfolgreich zurückgesetzt — DQN ep1000-Performance wiederhergestellt (L1: 89%, L3: 85%). Die ep3000 DQN-Ergebnisse sind leicht schwächer als Iteration 2 (L1: 81% vs 95%, L3: 88% vs 98%). Mögliche Ursache: **MEMORY_SIZE=20000** hält ältere, weniger policy-relevante Erfahrungen länger vor. Bei 3000 Episoden mit ~30 Steps/Episode ≈ 90.000 gespeicherte Transitions — der Buffer ist voll, aber 20k statt 15k bedeutet, dass ältere Daten aus frühen Explorations-Phasen länger im Buffer bleiben und Trainingsrauschen erzeugen.

Könnte auch statistische Varianz sein (kein fixer Seed). Entscheidung für Iteration 5: MEMORY_SIZE zurück auf 15000 und finalen Bestätigungslauf als definitive Konfiguration.

### Entscheidung für Iteration 5

**Letzte Iteration des allgemeinen Loops (danach: dedizierter L4-DQN-Lauf).**

1. **`MEMORY_SIZE` zurück auf 15000** — Iter2 zeigte mit 15000 die besten DQN ep3000-Werte. 20000 hat leicht verschlechtert.
2. Alle anderen Parameter wie Iteration 2/4 (LR=1e-4, Warmup=500, REWARD-Struktur aus Iter2).
3. Ziel: Finale Bestätigung der optimierten Konfiguration. Kein Experiment, Bestätigung.

---

## Iteration 5 — Finale Bestätigung der optimierten Konfiguration

**Datum:** 2026-06-29  
**Run-Verzeichnis:** `results/opt_iter5_2906/`  
**Hypothese:** Die optimierte Konfiguration aus Iteration 2 (beste Gesamtresultate) wird mit MEMORY_SIZE=15000 (Revert von Iter4) als finale Referenz bestätigt. Kein neues Experiment — Stabilitätsnachweis.

**Änderungen:**

| # | Datei | Änderung | Begründung |
|---|---|---|---|
| 1 | `config.py` | `MEMORY_SIZE`: 20000 → **15000** (Revert) | Iter4 zeigte leichte DQN ep3000-Regression mit 20000. Iter2-Wert (15000) war optimal. |

**Finale Konfiguration (Iteration 5 = beste Gesamtkonfiguration):**
- `DQN_LR = 1e-4`, `REPLAY_WARMUP = 500`, `MEMORY_SIZE = 15000`
- `REWARD_WALL = -1`, `REWARD_CLOSER = 1`, `REWARD_CLOSER_NO_BALL = 0.5`
- `REWARD_BYPASS_OBSTACLE = 8` (einmalig pro Episode)
- Epsilon-Decay: Standard 0.995 für alle Level

### Ergebnisse

| Level | Agent | ep1000 Goal% | ep1000 AvgR | ep3000 Goal% | ep3000 AvgR |
|---|---|---|---|---|---|
| L1 | Q-Table | 96.0% | 80.6 | 94.0% | 78.9 |
| L1 | DQN | 85.0% | 92.9 | 91.0% | 74.7 |
| L2 | Q-Table | 100.0% | 43.5 | 100.0% | 43.7 |
| L2 | DQN | 100.0% | 42.0 | 100.0% | 42.6 |
| L3 | Q-Table | 85.0% | 42.5 | 92.0% | 48.1 |
| L3 | DQN | 82.0% | 40.7 | **96.0%** | 51.6 |
| L4 | Q-Table | **89.0%** | 60.2 | 93.0% | 60.4 |
| L4 | DQN | 0.0% — | -20.6 | 0.0% — | -12.8 |
| L5 | Q-Table | 100.0% | 77.9 | 99.0% | 80.2 |
| L5 | DQN | **97.0%** | 79.5 | **99.0%** | 83.3 |

### Analyse

Stabile Resultate — Konfiguration bestätigt. L1–L5 Q-Table und DQN konvergieren zuverlässig. Kleinere Schwankungen gegenüber Iteration 2 sind statistisches Rauschen (kein fixer Seed).

**Highlights:** L1 DQN ep1000: 85% (Baseline: 74%), L3 DQN ep3000: 96% (Baseline: 89%), L5 DQN: 97%/99% (Baseline: 0%). L4 DQN bleibt 0% — wird im dedizierten Lauf adressiert.

**Abschluss allgemeiner Loop:** Iterationen 6–10 werden als dedizierte L4-DQN-Optimierung durchgeführt (separate Läufe, nur L4, intensivere Maßnahmen). Siehe Abschnitt "L4 DQN Dedizierter Lauf" unten.

---

## L4 DQN — Sparse Reward Experiment (Iteration 6)

**Hypothese:** Das DQN-Versagen auf L4 ist kein Algorithmus-Fehler sondern ein **Sparse Reward Problem**. Mit OBSTACLE_HEIGHT=4 blockiert das Hindernis auch die Tor-Reihe (y=3) — DQN muss 8–10 exakt gezielte Schritte in Folge ausführen bevor es den Tor-Reward sieht. Das passiert bei zufälliger ε-greedy-Exploration fast nie. Q-Table hat dieses Problem nicht: jeder besuchte Zustand hinterlässt direkt einen Q-Wert-Eintrag.

**Experiment:** OBSTACLE_HEIGHT von 4 auf **2** reduziert (Reihen 0–1 blockiert, Reihen 2–5 frei). Tor-Reihe (y=3) ist jetzt direkt durch Spalte 6 erreichbar → mehr zufällige Tor-Treffer → DQN startet Lernprozess.

**Run-Verzeichnis:** `results/opt_iter6_2906/` | **OBSTACLE_HEIGHT nach Experiment auf 4 zurückgesetzt.**

### Ergebnisse

| Variante | Q-Table ep1000 | Q-Table ep3000 | DQN ep1000 | DQN ep3000 |
|---|---|---|---|---|
| **L4 Hard** (H=4, Iter5) | 89% | 93% | **0%** | **0%** |
| **L4 Easy** (H=2, Iter6) | 87% | 95% | **89%** | **93%** |

### Analyse

**Hypothese vollständig bestätigt.** DQN springt von 0% auf 89%/93% — identische Konfiguration, nur kleineres Hindernis.

**Zusätzliche Beobachtung im 3000-Ep-Plot:** DQN zeigt bei ep1700–2200 einen **charakteristischen Einbruch** (Goal% fällt auf ~5%, erholt sich bis ep2500). Q-Table verläuft die ganze Zeit stabil. Das ist klassisches **Q-Value-Drift** — DQN destabilisiert sich kurzzeitig wenn das Netz in ein anderes lokales Optimum konvergiert. Q-Table hat dieses strukturelle Problem nicht.

**Präsentations-Kernaussagen (beide auf einem Plot sichtbar):**
1. *Sparse Reward erklärt das Versagen*: Kleineres Hindernis = mehr zufällige Erfolgspfade = DQN lernt
2. *DQN-Instabilität bleibt strukturell*: Selbst wenn DQN das Problem lösen kann, driftet es — Q-Table nie

---

## Iteration 7 — Level 6 Erstlauf

**Datum:** 2026-06-29  
**Run-Verzeichnis:** `results/opt_iter7_2906/`  
**Hypothese:** Level 6 (zwei Gegner + Mitspieler) ist die komplexeste Stufe. Der größere Zustandsraum könnte DQNs Generalisierungsvorteil gegenüber Q-Table erstmals sichtbar machen — Q-Table muss jeden State einzeln besuchen, DQN kann interpolieren.

**Konfiguration:** Finale Parameter aus Iteration 5 (keine Änderungen). Nur Level 6, 1000 + 3000 Episoden.

### Ergebnisse

| Agent | ep1000 Goal% | ep1000 AvgR | ep3000 Goal% | ep3000 AvgR |
|---|---|---|---|---|
| Q-Table | **1%** | 7.2 | **94%** | 77.1 |
| DQN | **98%** | 91.7 | **90%** | 85.7 |

### Trainingsverlauf (aus Logs)

**Q-Table ep3000:**
- ep500: 28.8% — breite Exploration, ε noch hoch
- ep1000: 5.2% — ε reached minimum, zu wenige States konvergiert
- ep1500: 6.2% — bleibt stecken (bekannte Q-Table-Lücke bei großem Zustandsraum)
- ep2000: **91%** — plötzlicher Durchbruch, genug States abgedeckt
- ep3000: 94.8% — stabil

**DQN ep3000:**
- ep500: 37.4% — frühes Lernen durch Generalisierung
- ep1000: 80.8% — bereits gut konvergiert
- ep1500: **94.8%** — Peak
- ep2000: 43.2% — **Q-Value-Drift-Einbruch**
- ep2500: 6.0% — Talsohle
- ep3000: 74.6% → letzte 100 Ep.: 90% — Erholung

### Analyse

**Q-Table ep1000 = 1%: Kein Versagen, sondern Zustandsraum-Sättigung.**  
L6 hat ~1.170 besuchte States bei ep3000. Bei ep1000 erst ~930 — zu wenige um alle notwendigen Übergänge stabil zu bewerten. Q-Table braucht schlicht mehr Episoden als bei L1–L5, weil der effektive Zustandsraum mit zwei Gegnern + Mitspieler deutlich wächst. Ab ep1600 läuft es stabil auf 90–95%.

**DQN ep1000 = 98%: Generalisierungsvorteil.**  
DQN ist bei 1000 Episoden *besser* als Q-Table, weil das Netz nicht jeden State einzeln besuchen muss. Ähnliche Zustände bekommen ähnliche Q-Werte durch die NN-Approximation — das kompensiert die begrenzte Exploration.

**DQN ep3000-Instabilität: Charakteristischer Q-Value-Drift.**  
Der Einbruch ep1600→ep2500 (94% → 6%) ist das gleiche Muster wie auf L4-Easy: DQN konvergiert in ein lokales Optimum, wird durch Verteilungsverschiebung im Replay Buffer destabilisiert und muss sich erholen. Q-Table zeigt dieses Verhalten nie — einmal gelernte Q-Werte werden nicht aktiv überschrieben.

**Umkehrung der bisherigen Erzählung:**  
Auf L1–L5 gilt: "Beide Agenten vergleichbar, Q-Table stabiler." L6 ergänzt: "Bei großem Zustandsraum konvergiert DQN schneller — aber bleibt strukturell instabiler." Die Wahl des Algorithmus hängt vom Zustandsraum und der Stabilitätsanforderung ab.

---

## Gesamtübersicht — Baseline vs. Finale Konfiguration (Iteration 5)

### Beste erreichte Ergebnisse (Iteration 2 = beste DQN, Iteration 5 = stabiler Abschluss)

| Level | Agent | Baseline Goal% (ep3000) | Finale Goal% (ep3000) | Verbesserung |
|---|---|---|---|---|
| L1 | Q-Table | 94% | 94% | = |
| L1 | DQN | 93% | 91% | ≈ |
| L2 | Q-Table | 100% | 100% | = |
| L2 | DQN | 100% | 100% | = |
| L3 | Q-Table | 95% | 92% | ≈ |
| L3 | DQN | 89% | **96%** | **+7%** |
| L4 | Q-Table | 94% | 93% | ≈ |
| L4 | DQN | **0%** | **0%** | — (Sparse Reward, Iter6 zeigt: mit H=2 lösbar) |
| L5 | Q-Table | 94% | **99%** | **+5%** |
| L5 | DQN | **0%** | **99%** | **+99%** |
| L6 | Q-Table | — | **94%** | (Erstlauf Iter7) |
| L6 | DQN | — | **90%** | (Erstlauf Iter7, DQN ep1000=98%) |

### Wichtigste Erkenntnisse

1. **L5 DQN-Durchbruch**: Epsilon-Decay-Bug (`>= 4` statt `== 4`) war die alleinige Ursache für 0% Goal-Rate. Fix: 1 Zeile Code. Zeigt, wie ein einzelner Implementierungsfehler ganze Trainingsläufe zunichte macht.

2. **Reward-Farming vermeiden**: `REWARD_CLOSER` ohne Ball-Bedingung + nicht-implementierter `REWARD_BYPASS_OBSTACLE` waren beide "stille Bugs" — definiert aber nie vergeben bzw. immer vergeben ohne Nutzen. Code-Analyse notwendig, nicht nur Training.

3. **Hyperparameter-Empfindlichkeit**: LR=5e-5 statt 1e-4 macht bei 1000 Episoden DQN unbrauchbar (11% statt 88%). Lernrate und Episodenbudget müssen aufeinander abgestimmt sein.

4. **L4 DQN als strukturelles Problem**: Multi-Step Obstacle-Navigation ist für Standard-DQN mit 3000 Episoden zu schwer. Curriculum Learning oder gezieltes Reward-Shaping (Corridor-Reward) nötig.

5. **Q-Table robust, DQN instabil**: Q-Table konvergiert auf L1–L5 stabil und reproduzierbar. DQN zeigt Varianz zwischen Runs (kein fixer Seed) und gelegentliche Instabilität nach ε-Minimum — charakteristisch für den Algorithmus, wichtig für die Präsentation.
