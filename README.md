# rl_football_project_htw

Reinforcement Learning Projekt im Fach Computational Intelligence — HTW Berlin.  
Ein Agent lernt in einer 2D-Gridworld-Fußballumgebung durch Q-Learning und DQN (PyTorch), den Ball zu holen, in Position zu gehen und ein Tor zu schießen.

---

## Repository-Struktur

```
rl_football_project_htw/
│
├── config.py                   # Zentrale Konfiguration: Gridgröße, Level, Rewards, Hyperparameter
│
├── src/
│   ├── environment.py          # Gridworld-Umgebung: Zustand, Aktionen, Rewards, Render
│   ├── q_table_agent.py        # Tabellarischer Q-Learning-Agent (Q-Tabelle als Dictionary)
│   ├── dqn_agent.py            # DQN-Agent mit PyTorch (neuronales Netz, Replay Buffer)
│   ├── train_q_table.py        # Trainingsloop für den Q-Table-Agenten
│   ├── train_dqn.py            # Trainingsloop für den DQN-Agenten
│   ├── utils.py                # Hilfsfunktionen: ReplayBuffer, Seeding
│   └── visualize.py            # Plots: Lernkurven, Epsilon-Verlauf, Vergleich Q-Table vs. DQN
│
├── tests/
│   └── test_environment.py     # Pytest-Tests für reset(), step() und alle Reward-Mechaniken
│
├── notebooks/
│   └── experiments.ipynb       # Jupyter Notebook für interaktive Experimente und Auswertung
│
├── docs/
│   ├── entscheidungen.md       # Dokumentierte Design- und Algorithmik-Entscheidungen
│   ├── projektplan.md          # Projektplan und Zeitplanung
│   └── vortrag_notizen.md      # Notizen für den Abschlussvortrag
│
├── results/
│   ├── models/                 # Gespeicherte Modelle (Q-Tabelle als .pkl, DQN als .pt)
│   ├── logs/                   # Trainingsmetriken pro Episode (JSON)
│   └── plots/                  # Generierte Diagramme (PNG)
│
├── requirements.txt            # Python-Abhängigkeiten
└── CLAUDE.md                   # Projektdokumentation für Claude Code
```

---

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Wichtigste Befehle

```bash
python src/train_q_table.py     # Q-Table Training starten
python src/train_dqn.py         # DQN Training starten
python -m pytest tests/         # Tests ausführen
```

---

## Umgebung: Die drei Level

Das aktive Level wird in `config.py` über `LEVEL = 1 | 2 | 3` gesteuert. Alle Level teilen dasselbe Grid (Standard 6×4) und dieselben 5 Aktionen:

| Aktion | Bedeutung |
|--------|-----------|
| 0 | hoch |
| 1 | runter |
| 2 | links |
| 3 | rechts |
| 4 | schießen |

---

### Level 1 — Schuss aus guter Position

```
. . . . . .
. . . . . .
A . . B , G
. . . . . .

A = Agent  B = Ball  G = Tor  , = Schusszone
```

Der Agent muss den Ball aufnehmen, in die **Schusszone** (`x ≥ SHOOT_ZONE_X`, Standard: x=4) laufen und von dort schießen. `shoot` ist nur nützlich, wenn der Agent in der richtigen Zone UND in der richtigen Reihe steht — der Agent lernt situationsabhängige Aktionsauswahl.

**Zustand (5 Elemente):** `(agent_x, agent_y, ball_x, ball_y, has_ball)`

| Ereignis | Reward |
|----------|--------|
| Tor (Schuss aus Zone, richtige Reihe) | +30 |
| Ball aufgenommen | +5 |
| Näher zum Tor bewegt | +1 |
| Schritt | −1 |
| Schuss ohne Ball | −5 |
| Schuss außerhalb der Zone | −5 |

---

### Level 2 — Dribbling vs. Vorwärtspass

```
. . . . . .
. . . . . .
A . . B . G
. . . . . .
```

`shoot` schießt den Ball **`SHOOT_RANGE` Felder nach rechts** (Standard: 3). Der Agent verliert den Ball und muss ihn nachholen. Es entsteht eine echte Entscheidung: langsam und sicher dribbling zum Tor, oder schneller Vorwärtspass mit dem Risiko, den Ball zu verlieren.

**Zustand (5 Elemente):** `(agent_x, agent_y, ball_x, ball_y, has_ball)`

| Ereignis | Reward |
|----------|--------|
| Tor (Dribbling ins Tor oder Pass landet auf Tor) | +40 |
| Ball aufgenommen | +5 |
| Näher zum Tor bewegt | +1 |
| Ball per Pass näher ans Tor | +2 |
| Schritt | −1 |
| Ball verlässt rechte Wand ohne Tor | −5 |
| Schuss ohne Ball | −3 |

---

### Level 3 — Gegner bewegt sich zum Ball

```
. . . . . .
. . . . . .
A . . B X G
. . . . . .

X = Gegner (regelbasiert)
```

Gleiche Mechanik wie Level 2, aber ein **regelbasierter Gegner** kommt hinzu. Der Gegner startet nahe dem Tor (`goal_x − OPP_START_X_FROM_GOAL`, Standard: x=4) und bewegt sich alle `OPP_MOVE_EVERY` Schritte (Standard: 2) einen Schritt in Richtung Ball. Erreicht er den Ball, endet die Episode mit einer Strafe.

Der Agent muss lernen: *Schieße ich sofort weiter, bevor der Gegner kommt? Oder dribble ich sicher?*

**Zustand (7 Elemente):** `(agent_x, agent_y, ball_x, ball_y, has_ball, opp_x, opp_y)`

| Ereignis | Reward |
|----------|--------|
| Tor (Dribbling ins Tor oder Pass landet auf Tor) | +50 |
| Ball aufgenommen | +5 |
| Näher zum Tor bewegt | +1 |
| Ball per Pass näher ans Tor | +2 |
| Schritt | −1 |
| Ball verlässt rechte Wand ohne Tor | −5 |
| Schuss ohne Ball | −5 |
| Gegner erreicht losen Ball (Episode endet) | −10 |
| Gegner tackelt Agent mit Ball (Episode endet) | −20 |
