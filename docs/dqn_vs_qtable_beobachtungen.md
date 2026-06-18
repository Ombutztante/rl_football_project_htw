# DQN vs. Q-Table — Beobachtungen & Optimierungsverlauf

Dieses Dokument hält chronologisch fest, welche Probleme mit den Agenten aufgetreten sind,
was die Ursachen waren und welche Maßnahmen ergriffen wurden. Ziel ist die Nachvollziehbarkeit
für Präsentation und Abschlussbericht.

---

## Ausgangslage (vor Optimierung)

**Grid:** 6×4 | **Episoden:** 500 | **Branch:** a_dev_1 / m_dev_1

### Beobachtung
- Beide Algorithmen konvergieren auf allen 3 Leveln auf >90% Goal-Rate.
- Kein sichtbarer Unterschied zwischen Q-Table und DQN in den Ergebnissen.
- Animationen zeigen kein klares Verhalten — Spielfeld zu klein, Problemstellung zu trivial.

### Problem
Das 6×4-Grid hat einen Zustandsraum von nur ~1.800 erreichbaren States.
Beide Algorithmen finden die Lösung schon in den ersten Episoden durch Zufall (ε-greedy),
ohne echtes Lernen demonstrieren zu müssen. Die akademische Aussagekraft ist gering.

---

## Schritt 1 — Grid-Vergrößerung (15.06.2026)

**Änderung:** `GRID_WIDTH` 6→10, `GRID_HEIGHT` 4→6, `MAX_STEPS` 200→300

### Motivation
- Mehr Raum für Dribbling-vs-Pass-Entscheidung (Level 2)
- Längere optimale Trajektorien → Lernen wird sichtbarer
- Zustandsraum L1/L2: 7.200 States; L3/L4: 432.000 States

### Effekt
- Q-Table konvergiert weiterhin schnell (~200 Episoden für L1/L2)
- DQN braucht länger, aber Level 2 klappt gut
- Level-Unterschiede werden in Animationen erstmals sichtbar

---

## Schritt 2 — Ball-Startposition angepasst (15.06.2026)

**Änderung:** `BALL_START_X_L2 = 1`, `BALL_START_X_L3 = 1` (vorher: Feldmitte)

### Motivation
Level 2 war visuell nicht von Level 1 unterscheidbar — der Ball startete schon nah am Tor.
Mit Ball bei x=1 muss der Agent das gesamte Feld überbrücken (8 Felder bis zum Tor).
Erst dadurch entsteht die echte Entscheidung: langsam dribbeln oder schnell passen und nachlaufen.

### Effekt
- Dribbling-vs-Pass-Trajektorien in Animationen klar erkennbar
- Q-Table L2 und DQN L2 konvergieren beide auf 100%

---

## Trainingsergebnisse nach Optimierung (1000 Episoden, 10×6-Grid)

**Datum:** 15.06.2026 | **Branch:** a_dev_2

| Agent | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Q-Table | **100%** ab ep~200 | **100%** ab ep~200 | **91%** ab ep~800 | **90%** ab ep~800 |
| DQN | **~48%** (nie konvergiert) | **100%** ab ep~200 | **82%** ab ep~700 | **0%** (kompletter Ausfall) |

*Goal-Rate = Anteil erfolgreicher Episoden in den letzten 200 Episoden*

---

## Problem 1 — DQN Level 1: Konvergiert nie (~48% Goal-Rate)

### Beobachtung (vor Optimierung, Branch a_dev_1)
- DQN Level 1 schwankt über alle 1000 Episoden zwischen 33–54%.
- Loss bleibt ab Episode 300 konstant bei ~1.0–1.2 — kein Lernfortschritt.
- Epsilon erreicht Minimum (0.05) bei Episode ~600 — keine Verbesserung danach.
- Zum Vergleich: Q-Table Level 1 ist ab Episode 200 bei 100%.

### Ursache
**Deadly Triad + Sparse Reward.**
Level 1 hat einen extrem spärlichen Reward: nur der exakte Zustand
(x ≥ 8 UND y = 3 UND has_ball = 1) mit Aktion `shoot` liefert positives Terminal-Signal.
Das neuronale Netz versucht diese scharfe Bedingung als glatte Funktion zu approximieren —
was strukturell nicht funktioniert. Q-Table dagegen speichert den exakten Zustand direkt.

Ein weiterer Faktor: Zone-Miss (Schuss in x≥8 aber falsche Reihe) gab Reward=0.
Da 0 > -1 (Step-Penalty), lernte das Netz: "In der Zone schießen ist immer besser als
nicht schießen" — unabhängig von der Reihe. Das erzeugte eine Falle bei ~50% Goal-Rate.

Das ist ein bekanntes Versagen der Kombination aus:
- Function Approximation (NN)
- Bootstrapping (TD-Learning)
- Off-Policy Learning (Replay Buffer)

→ Auch als "Deadly Triad" bekannt (Sutton & Barto, 2018).

### Akademische Bedeutung
Zeigt klar: DQN ist auf kleinen, strukturierten Zustandsräumen mit spärlichem Reward
SCHLECHTER als Q-Table, nicht nur langsamer.

---

## Optimierung 1 — DQN Level 1: Reward Shaping + 2D Schusszone (18.06.2026)

**Branch:** a_dev_4

### Änderungen
1. **2D Schusszone** (statt vertikaler Streifen):
   - Alt: `agent_x >= 8` — ganzer rechter Streifen, inklusive Ecken weit vom Tor
   - Neu: `agent_x >= 8 AND |agent_y - goal_y| <= 1` — nur Felder nahe dem Tor
   - Realistische Schusszone: (8,2), (8,3), (8,4) und (9,2), (9,4)

2. **Zone-Miss-Strafe** (`REWARD_SHOOT_ZONE_MISS = -3`):
   - Schuss in Zone, aber falsche Reihe → `-3` statt `0`
   - Bricht die falsche Lerndynamik: Schießen in der Zone ist nur bei y=3 sinnvoll

3. **Reihen-Alignment-Shaping** (`REWARD_GOAL_ROW_ALIGN = +1`):
   - Agent trägt Ball auf goal_y (Reihe 3) → zusätzlich `+1`
   - Gibt dem Netz einen Gradienten zur richtigen Schussreihe, bevor die Zone erreicht wird
   - Feuert nur auf Bewegungsaktionen, nicht auf Schuss

### Ergebnis nach Optimierung (1000 Episoden)

| Agent | Goal-Rate (letzten 100 Ep.) | Verhalten |
|---|---|---|
| Q-Table | **96%** (stabil ab ep~200) | Lernt schnell, bleibt stabil |
| DQN | **~72–84%** (instabil) | Erreicht ~99% bei ep~200, fällt dann zurück |

### Neue Beobachtung: Deadly Triad jetzt sichtbar
DQN **findet** die Lösung jetzt tatsächlich (ep~200: kurzzeitig ~99%) — das war vorher
nie der Fall. Aber die Policy ist nicht stabil: das Netz fällt danach auf ~73% zurück,
erholt sich leicht, und pendelt am Ende um ~72%.

Das ist die Deadly Triad in Reinform:
- Network lernt eine gute Policy
- Bootstrapping + Replay Buffer verursachen Drift, sobald sich die Datenverteilung ändert
- Q-Table ist nach Konvergenz vollständig stabil — exakte Werte werden nicht überschrieben

### Akademische Bedeutung (aktualisiert)
Vorher war DQN L1 „kaputt" (lernte nie). Nach dem Reward Shaping ist das Verhalten
**didaktisch wertvoller**: Man sieht klar den Unterschied zwischen
tabular (einmal konvergiert → stabil) und neural (konvergiert → driftet).
Der Plot zeigt zwei qualitativ verschiedene Konvergenzverläufe.

---

## Problem 2 — DQN Level 3: Späte Konvergenz + leichte Instabilität

### Beobachtung
- DQN benötigt bis Episode ~600 um überhaupt zu lernen (Q-Table schafft das bis ep~400).
- Nach Konvergenz auf ~86% leichter Rückgang auf 82% — Q-Table bleibt stabil bei 91%.
- Loss steigt am Ende wieder an (ep700: 1.65, ep1000: 2.27) → Q-Value Drift.

### Ursache
**Q-Value Drift nach Epsilon-Minimum.**
Sobald ε = 0.05 (ab Episode ~600), verändert sich die Verteilung der Transitionen
im Replay Buffer — der Gegner reagiert auf die nun deterministischere Policy.
Das Netz driftet leicht, weil der Zielwert (Target Network) zu schnell veraltet.

### Akademische Bedeutung
Demonstriert die bekannte Instabilität von DQN auf nicht-stationären Problemen.
Q-Table hat dieses Problem nicht — einmal konvergiert, bleibt die Policy stabil.

---

## Problem 3 — DQN Level 4: Kompletter Ausfall (0% über 1000 Episoden)

### Beobachtung
- DQN erzielt in keiner der 1000 Episoden ein Tor.
- Loss bleibt niedrig (~0.45 bei ep300–500), steigt danach leicht.
- Q-Table erreicht 90% Goal-Rate (bei nur 0.3% Zustandsraumabdeckung!).

### Level 4 Mechanik (zur Erinnerung)
```
. . . . . . # . X .
. . . . . . # . . .
. . . . . . # . . .
A B . . . . # . . G   ← Direktschuss blockiert
. . . . . . . . . .   ← Umweg durch Reihen 4–5
. . . . . . . . . .
```
Agent muss: Ball holen → nach unten ausweichen → Hindernis umgehen → Tor.

### Ursache
**Epsilon zu früh am Minimum + fehlende Reward-Signale für den Umweg.**

Epsilon erreicht 0.05 nach ~600 Episoden — zu früh für ein Problem,
das einen Mehrstufenplan mit räumlichem Umweg erfordert.
Der Pfad um das Hindernis herum wird in 300 Schritten durch Zufalls-Exploration
selten genug gefunden. Der Replay Buffer füllt sich überwiegend mit
"Hindernis getroffen, kein Tor"-Transitionen.

Das Netz lernt: "Obstacle-Kollision vermeiden."
Es lernt NICHT: "Nach unten gehen und dann weiterdribbeln."

Q-Table funktioniert hier weil es nur die ~1.300 tatsächlich besuchten States braucht —
und durch Zufall findet es in frühen Episoden den Umweg-Pfad, der dann direkt gespeichert wird.

### Mögliche Fixes
- Langsamerer Epsilon-Decay: 0.998 statt 0.995 (Minimum erst bei ~1.600 Episoden)
- Mehr Episoden: 2.000–3.000 für DQN L4
- Reward-Shaping: +1 für Erreichen von Reihe ≥ 4 wenn x ≥ 5 (Umweg-Incentive)
- Stärkere Obstacle-Penalty: -5 statt -2 für klareres Signal
- Curriculum Learning: DQN erst auf L3 vortrainieren, dann auf L4 finetunen

---

## Q-Table Zustandsraumabdeckung (Erklärung des überraschenden L3/L4-Erfolgs)

| Level | Theoretische States | Besuchte States | Abdeckung |
|---|---|---|---|
| L1/L2 | 7.200 | ~1.500 | 20% |
| L3/L4 | 432.000 | ~1.300 | **0,3%** |

Q-Table L3/L4 konvergiert mit nur 0,3% Abdeckung! Das ist möglich weil:
- Der tatsächlich erreichbare Zustandsraum viel kleiner ist als der theoretische
  (viele Kombinationen sind in der Praxis unerreichbar)
- Die wichtigen Zustände (Agent mit Ball, nahe Tor) in jeder Episode besucht werden
- Q-Table braucht keine Generalisierung — es reicht, die relevanten States zu kennen

DQN muss hingegen über den GESAMTEN Zustandsraum generalisieren,
was auf kleinen Grids mehr schadet als hilft.

---

## Zusammenfassung: Wann Q-Table, wann DQN?

| Eigenschaft | Q-Table gewinnt | DQN gewinnt |
|---|---|---|
| Zustandsraumgröße | Klein (< ~50.000) | Groß (> 100.000) |
| Reward-Dichte | Spärlich OK | Braucht dichte Signale |
| Konvergenzgeschwindigkeit | Schneller | Langsamer, braucht Warmup |
| Stabilität nach Konvergenz | Sehr stabil | Kann driften |
| Generalisierung | Keine | Funktioniert auf ungesehenen States |
| Räumliche Constraints | Robust (exakte Grenzen) | Schwächer (approximiert Grenzen) |

**Fazit für dieses Projekt:**
Die Level 1–4 auf dem 10×6-Grid sind noch im "Q-Table-Vorteilsbereich".
DQN würde seinen Vorteil erst bei deutlich größeren Grids (≥ 15×10)
oder zusätzlicher Komplexität (zufällige Ball-Starts, mehrere Gegner) ausspielen.

---

## Offene Optimierungsaufgaben

- [x] DQN Level 1: Reward Shaping + 2D Schusszone implementiert (18.06.2026)
- [ ] DQN Level 4: Langsameren Epsilon-Decay + mehr Episoden testen
- [ ] DQN Level 4: Curriculum Learning (L3 → L4) evaluieren
- [ ] Separate Hyperparameter-Sets pro Level (aktuell: alle Level gleiche Config)
- [ ] Double DQN implementieren (reduziert Overestimation Bias)
- [ ] Level 5 oder größeres Grid für DQN-Vorteil-Demonstration
