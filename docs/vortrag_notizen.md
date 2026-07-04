# Vortrag-Notizen — Ergebnisse & Probleme

Stand: 2026-07-04 | Redner: Alexander (DQN/Q-Table Grundlagen + Ergebnisse + Optimierung)

---

## Abschnitt 1: Ergebnisse (Live-Dashboard)

Alle Zahlen aus Greedy-Eval (ε=0, 300 Episoden) und finalen Trainingslogs `results/final_0207/`.  
Im Dashboard: Tab **Agent** zeigt Rollout-GIF, Tab **Plots** zeigt Trainingskurven, Tab **Vergleich** zeigt side-by-side GIF + Vergleichsplot.

---

### Level 1 — Schuss nur aus guter Position

**Plot:** `final_0207/plots/comparison_level1_ep3000.png`  
**Animation:** `final_0207/animations/compare_level1_ep3000.gif`

- **Q-Table**: Konvergiert bei **Ep54** auf 100% Torrate — bleibt danach konstant flach auf ~32 Avg-Reward. Klassisches tabellarisches Lernen: einmal gelernt, nie vergessen.
- **DQN**: Langes Plateau (~20–35% Torrate) bis **Ep2513**, dann abrupter Sprung auf 99%. Endresultat identisch mit Q-Table, aber der Weg dorthin grundlegend anders.
- **Kernaussage**: Gleiche Aufgabe, radikal verschiedene Lernkurven. Zeigt den strukturellen Unterschied zwischen tabular (direkte Lookup-Tabelle) und neural (Funktionsapproximation mit Nebenwirkungen).
- **Was die Animation zeigt**: Agent holt Ball links, navigiert auf die Torreihe (y=3), betritt Schusszone (markiert), schießt → Tor. Direkter, kurzer Pfad (~9–10 Schritte).

---

### Level 2 — Dribbling vs. Vorwärtspass

**Plot:** `final_0207/plots/comparison_level2_ep3000.png`  
**Animation:** `final_0207/animations/compare_level2_ep3000.gif`

- **Q-Table**: Konvergiert bei **Ep49** auf 100%. Schnellste Konvergenz aller Level — Reward-Struktur ohne Penalty-Trap, klares Signal.
- **DQN**: Konvergiert bei **Ep42** auf 100% — hier ist DQN sogar marginal schneller als Q-Table. Beide stabil über alle 3000 Episoden.
- **Kernaussage**: Wenn der Reward-Gradient klar und dicht ist (Pass gibt +2 auch aus suboptimaler Position), funktionieren beide Algorithmen gleich gut und schnell. Level 2 zeigt das Optimum.
- **Was die Animation zeigt**: Q-Table und DQN zeigen unterschiedliche Strategien sichtbar: Q-Table dribbles oft direkter, DQN nutzt häufiger Vorwärtspass und läuft nach. Beide erzielen Tor in ~8–10 Schritten.

---

### Level 3 — Gegner bewegt sich alle 2 Schritte

**Plot:** `final_0207/plots/comparison_level3_ep3000.png`  
**Animation:** `final_0207/animations/compare_level3_ep3000.gif`

- **Q-Table**: Konvergiert bei **Ep960** auf 92%. Später als L1/L2 — der Gegner vergrößert den Zustandsraum auf ~432.000 theoretische States. Am Ende leichte Schwankungen (88–95%) durch Zufall im Gegner-Timing.
- **DQN**: Konvergiert bei **Ep574** auf 96% — hier ist DQN **schneller** als Q-Table. Kann ähnliche Gegner-Positionen durch geteilte Netzgewichte generalisieren; Q-Table muss jeden State einzeln lernen.
- **Kernaussage**: Erster Hinweis auf DQN-Vorteil bei wachsendem Zustandsraum. DQN generalisiert über (Gegner bei x=7) → (Gegner bei x=6), Q-Table muss beide separat besuchen.
- **Was die Animation zeigt**: Beide Agenten weichen dem Gegner aus — oft durch schnelle Richtungswechsel oder frühen Pass, bevor der Gegner den Ball erreicht.

---

### Level 4 — Statisches Hindernis blockiert Direktpfad

**Plot:** `final_0207/plots/comparison_level4_ep3000.png`  
**Animation:** `final_0207/animations/compare_level4_ep3000.gif`

- **Q-Table**: Konvergiert bei **Ep723** auf 93%. Findet den Korridor (Reihen 4–5) durch systematische Exploration. Stabil.
- **DQN**: **0% Torrate über alle 3000 Episoden**. Vollständiges Versagen. AvgR = -12.8 (Agent holt Ball, macht Forward-Pass, verliert Ball an Gegner → loop).
- **Kernaussage** (stärkste Aussage des gesamten Projekts): Dasselbe Problem, derselbe Algorithmus-Setup — Q-Table löst es, DQN nicht. Ursache: Sparse Reward + Multi-Step-Navigation. Das Hindernis blockiert alle zufälligen Erkundungspfade; DQN sieht in 3000 Episoden kaum ein positives Tor-Signal. Q-Table braucht keine Generalisierung — ein paar gefundene Durchläufe reichen.
- **Was die Animation zeigt**: Q-Table navigiert sichtbar nach unten, um das Hindernis zu umgehen, dann zurück zum Tor. DQN-GIF zeigt Agent, der immer wieder nach rechts schießt und den Ball verliert.
- **Hinweis für Dashboard**: Hier den **Vergleichs-Plot** zeigen — der Kontrast (Q-Table steigt auf 93%, DQN bleibt flach bei 0%) ist das klarste Bild des gesamten Projekts.

---

### Level 5 — Mitspieler + Gegner

**Plot:** `final_0207/plots/comparison_level5_ep3000.png`  
**Animation:** `final_0207/animations/compare_level5_ep3000.gif`

- **Q-Table**: Konvergiert bei **Ep197** auf 99%. Schnell — das Pass-an-Mitspieler-Mechanismus gibt direkte positive Rewards (+15 bei erfolgreicher Übergabe).
- **DQN**: Konvergiert bei **Ep211** auf 99%. Nahezu identisch mit Q-Table, beide stabil.
- **Kernaussage**: Kooperatives Spiel funktioniert für beide Agenten überraschend gut und schnell. Der +15-Reward für Mitspieler-Pickup schafft ein dichtes Reward-Signal, das DQN leicht lernen kann.
- **Was die Animation zeigt**: Agent passt diagonal zum Mitspieler (T), Mitspieler dribbled zum Tor und schießt. Kooperative Sequenz klar sichtbar.

---

### Level 6 — Zwei Gegner + Mitspieler (State-Space-Explosion)

**Plot:** `final_0207/plots/comparison_level6_ep3000.png`  
**Animation:** `final_0207/animations/compare_level6_ep3000.gif`

- **Q-Table**: Konvergiert erst bei **Ep1514** — späteste Konvergenz aller Level. Zeigt im Training einen tiefen Dip (ep200–ep1200 nur 3–7% Torrate) weil der Zustandsraum mit zwei Gegnern und Mitspieler deutlich wächst. Ab ep1600 stabil auf 92–96%.
- **DQN**: Konvergiert früh (**Ep609** auf 90%), zeigt aber einen markanten **Q-Drift-Einbruch** ep1600–2200 (von 95% auf 5%!) bevor es sich auf 90% erholt. Greedy-Eval: 100%.
- **Kernaussage**: Level 6 dreht die Verhältnisse um. DQN konvergiert schneller (Generalisierung über ähnliche Zustände mit zwei Gegnern), ist aber strukturell instabiler. Q-Table braucht länger, ist danach aber stabiler. → Die Wahl des Algorithmus hängt vom Zustandsraum und der Stabilitätsanforderung ab.
- **Was die Animation zeigt**: Komplexes Bild — Agent weicht beiden Gegnern aus, sucht Mitspieler-Position, führt Diagonalpass aus. Sichtbar längere Trajektorie als L1–L5.

---

## Abschnitt 2: Typische Probleme und Lösungen

Drei repräsentative Probleme mit konkreten Plot-Referenzen.

---

### Problem 1 — Reward Exploit: Agent schießt nie, obwohl er "lernt"

**Betroffenes Level:** L1 (vor Iter8-Fix)  
**Kein Plot zeigt den Exploit** — das ist die eigentliche Pointe.  
**Plot (nach Fix):** `results/final_0207/plots/comparison_level1_ep3000.png`

**Warum man es im Plot nicht sieht:**  
Die Trainingsmetriken sahen während des Exploits völlig akzeptabel aus — moderate positive Rewards, keine offensichtlichen Warnzeichen in der Kurve. Der Agent sammelte `REWARD_GOAL_ROW_ALIGN = +1` pro Schritt auf y=3 und `+5` beim Ball-Aufnehmen, was zusammen stabil positive Rewards erzeugte. Kein Ausreißer, kein Absturz, kein Alarm.

**Wie es entdeckt wurde:**  
Ausschließlich durch **Greedy-Rollout** (ε=0, Agent spielt ohne Exploration): Agent holt Ball, läuft auf Torreihe y=3 — und bleibt dort bis MAX_STEPS. Kein einziges Tor in 300 Episoden. Erst die Inspektion der eigentlichen Policy hat das Verhalten sichtbar gemacht.

**Ursache:**  
`REWARD_GOAL_ROW_ALIGN` feuerte jeden Schritt ohne Guard, solange der Agent mit Ball auf y=3 stand. Auf der Torreihe gilt: `+1` (goal_row_align) `−1` (step_penalty) = **netto 0 pro Schritt**. Das ist besser als die Torreihe zu verlassen (`−1` pro Schritt). Für den Agenten ist es strikt rational, dort zu bleiben. Der Schuss (`+30`) endet die Episode — das ist für den Agenten unattraktiv verglichen mit "ewig +0 pro Schritt sammeln".

**Fix:**  
`goal_row_rewarded`-Flag → Reward feuert nur einmal pro Episode (analog zu `ball_pickup_rewarded`). Eine Zeile Code.

**Empfehlung für den Vortrag:**  
Nicht versuchen, den Exploit im Plot zu zeigen — er ist dort nicht sichtbar. Stattdessen: **Konzept erklären** ("wir haben 3000 Episoden trainiert, Metriken sahen gut aus, Agent hat nie ein Tor geschossen — erst der Greedy-Rollout hat es aufgedeckt") und dann das `before/after` am Code zeigen: die eine Zeile ohne Guard vs. mit Guard.

**Lehraussage:**  
Reward Engineering ist schwierig. Eine positive Trainingskurve bedeutet nicht, dass der Agent das richtige Verhalten lernt. **Policy-Inspektion (Greedy-Rollout) ist Pflicht** — Trainingsmetriken allein sind kein Qualitätsbeweis. Das ist die stärkste Aussage: das Problem war unsichtbar bis zur direkten Verhaltensbeobachtung.

---

### Problem 2 — Epsilon-Bug: DQN L5 bei 0%, obwohl L4-Fix da war

**Betroffenes Level:** L5 (vor Iter1-Fix)  
**Plot (Problem):** `results/opt_iter0_2906/plots/comparison_level5_ep3000.png`  
**Plot (nach Fix):** `results/final_0207/plots/comparison_level5_ep3000.png`

**Was der alte Plot zeigt:**  
Der Baseline-Vergleichsplot L5 zeigt Q-Table auf 94%, DQN flach auf **0% über alle 3000 Episoden**. DQN lernt buchstäblich nichts — der Plot der DQN-Kurve ist eine Horizontale nahe 0.

**Ursache:**  
In `train_dqn.py` lautete die Bedingung für den langsamen Epsilon-Decay: `if LEVEL >= 4`. Damit bekam L5 denselben sehr langsamen Decay (ε=0.998) wie L4. Das bedeutet: ε erst bei Ep~1800 am Minimum — in den ersten 1800 Episoden fast nur zufällige Exploration, kein gezieltes Lernen. Bei 1000 Episoden praktisch untrainiert.

**Fix:**  
`>= 4` → `== 4`. Eine Zeile Code.

**Lehraussage:**  
Ein einziger Implementierungsfehler kann das gesamte Training eines Agenten zunichte machen. Die Ursache war nicht im Algorithmus, nicht im Reward-Design, sondern in einer falschen Vergleichsoperation. Code-Review ist genauso wichtig wie Hyperparameter-Tuning.

---

### Problem 3 — Sparse Reward: DQN L4 findet das Tor nie

**Betroffenes Level:** L4  
**Plot (Problem):** `results/opt_iter0_2906/plots/comparison_level4_ep3000.png`  
**Plot (Bestätigung durch Experiment):** `results/opt_iter6_2906/plots/comparison_level4_ep3000.png` (L4 mit OBSTACLE_HEIGHT=2)  
**Plot (final):** `results/final_0207/plots/comparison_level4_ep3000.png`

**Was der Baseline-Plot zeigt:**  
Q-Table steigt auf 94% Torrate, DQN bleibt über alle 3000 Episoden exakt auf 0%. Kein einziges Tor. AvgReward pendelt bei ca. -9 (Ball holen +5, Gegner fängt -10, netto -5/Episode).

**Ursache:**  
Sparse Reward + Multi-Step-Navigation. Das Hindernis blockiert den direkten Pfad komplett. Für DQN bedeutet das: Die Zufalls-Exploration muss zufällig die Sequenz ausführen (Ball holen → nach unten → Korridor → zurück auf Torreihe → schießen = ~12 exakt aufeinanderfolgende Aktionen), bevor ein positives Signal entsteht. Die Wahrscheinlichkeit dafür ist bei ε=0.05 (nach Ep600) sehr gering — der Replay Buffer füllt sich ausschließlich mit Misserfolgen. Q-Table hat das Problem nicht: jeder besuchte Zustand bekommt direkt einen Eintrag, ein einziger zufällig gefundener Korridor-Durchlauf reicht zum Lernen.

**Bestätigung durch Experiment (Iter6):**  
OBSTACLE_HEIGHT von 4 auf 2 reduziert → DQN springt sofort auf 89%/93%. Dasselbe Netz, dieselben Hyperparameter, nur kleineres Hindernis. Das beweist: das Problem liegt nicht am Algorithmus, sondern an der Explorations-Zugänglichkeit des Tor-Rewards.

**Fix für die Präsentation:**  
DQN L4 = 0% wird bewusst als Versagen gezeigt, weil es die stärkste Aussage des Projekts illustriert: Q-Table ist strukturell besser für kleine Grids mit Navigations-Constraints. Das Experiment (kleineres Hindernis) wird als zusätzliche Folie gezeigt.

**Lehraussage:**  
DQN braucht dichte Reward-Signale. Wenn der Pfad zum einzigen positiven Reward blockiert ist und zufällige Exploration ihn statistisch kaum findet, versagt DQN. Q-Table braucht das nicht — tabellarisches Lernen skaliert linear mit der Anzahl besuchter Zustände, nicht mit der Dichte der Reward-Signale.

---

## Abschnitt 3: Tipps für den Live-Dashboard-Durchgang

**Reihenfolge im Dashboard:**

1. **Level 1 → Tab Vergleich**: Compare-GIF zeigen (Q-Table schnell, DQN spät). Dann Plot-Tab → Vergleichsplot → Kurvenunterschied betonen.
2. **Level 2 → Tab Vergleich**: Beide konvergieren schnell. Kurz halten.
3. **Level 4 → Tab Vergleich**: Vergleichsplot öffnen. Q-Table steigt, DQN bleibt bei 0%. Längste Erklärung — das ist der Höhepunkt.
4. **Level 6 → Tab Plots**: DQN-Plot zeigt Q-Drift-Einbruch ep1600–2200. Q-Table-Plot zeigt stabilen Anstieg ab ep1600. Dann Agent-GIF zeigen — komplexestes Verhalten.

**Für den Probleme-Teil:**
- Alten Plot aus `results/old/` oder `results/opt_iter0_2906/` öffnen und direkt neben dem finalen Plot zeigen. Der Kontrast erklärt sich von selbst.
- Bei Reward-Exploit: erwähnen, dass der Greedy-Rollout (Agent spielen ohne Zufall) das Problem erst aufgedeckt hat — nicht die Trainingskurve.
