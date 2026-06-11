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
