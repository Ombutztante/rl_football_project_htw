# rl_football_project_htw

Reinforcement Learning Projekt im Fach Computational Intelligence — HTW Berlin.  
Eine 2D-Gridworld-Fußballumgebung, in der ein Agent den Ball holen, in Position gehen und ein Tor schießen muss.

---

## Repository-Struktur

```
rl_football_project_htw/
│
├── config.py           # Gridgröße, Level, Rewards, Spielparameter
│
├── src/
│   ├── environment.py  # Gridworld-Umgebung: Zustand, Aktionen, Rewards, Render
│   └── play.py         # Manuelles Spielen über Matplotlib-Fenster (Tastatur)
│
└── requirements.txt    # Python-Abhängigkeiten (numpy, matplotlib)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Manuell spielen

```bash
python src/play.py
```

Ein Matplotlib-Fenster öffnet sich. Steuerung:

| Taste | Aktion |
|-------|--------|
| Pfeiltasten | Agent bewegen (hoch / runter / links / rechts) |
| Leertaste oder `S` | Schießen |
| `R` | Episode neu starten |
| `Q` | Beenden |

Das aktive Level wird in `config.py` über `LEVEL = 1 | 2 | 3` gewählt.

---

## Umgebung: Die drei Level

Das Grid ist standardmäßig 6×4 Felder groß. Alle Level teilen dieselben 5 Aktionen:

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

Der Agent muss den Ball aufnehmen, in die **Schusszone** (`x ≥ SHOOT_ZONE_X`, Standard: x=4) laufen und von dort schießen.

**Zustand (5 Elemente):** `(agent_x, agent_y, ball_x, ball_y, has_ball)`

| Ereignis | Reward |
|----------|--------|
| Tor (Schuss aus Zone) | +30 |
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

`shoot` schießt den Ball **`SHOOT_RANGE` Felder nach rechts** (Standard: 3). Der Agent verliert den Ball und muss ihn nachholen — echte Entscheidung zwischen langsamem Dribbling und riskantem Vorwärtspass.

**Zustand (5 Elemente):** `(agent_x, agent_y, ball_x, ball_y, has_ball)`

| Ereignis | Reward |
|----------|--------|
| Tor | +40 |
| Ball aufgenommen | +5 |
| Näher zum Tor bewegt | +1 |
| Ball per Pass näher ans Tor | +2 |
| Schritt | −1 |
| Ball verlässt rechte Wand ohne Tor | −5 |
| Schuss ohne Ball | −3 |

---

### Level 3 — Gegner bewegt sich zum Ball

```
. . . . X .   ← Gegner startet oben rechts
. . . . . .
A . . B . G
. . . . . .

X = Gegner (regelbasiert)
```

Gleiche Mechanik wie Level 2, aber ein **regelbasierter Gegner** kommt hinzu. Er startet bei `(goal_x − OPP_START_X_FROM_GOAL, 0)` (Standard: x=4, y=0) und bewegt sich alle `OPP_MOVE_EVERY` Schritte (Standard: 2) einen Schritt in Richtung Ball. Erreicht er den Ball, endet die Episode mit einer Strafe.

**Zustand (7 Elemente):** `(agent_x, agent_y, ball_x, ball_y, has_ball, opp_x, opp_y)`

| Ereignis | Reward |
|----------|--------|
| Tor | +50 |
| Ball aufgenommen | +5 |
| Näher zum Tor bewegt | +1 |
| Ball per Pass näher ans Tor | +2 |
| Schritt | −1 |
| Ball verlässt rechte Wand ohne Tor | −5 |
| Schuss ohne Ball | −5 |
| Gegner erreicht losen Ball (Episode endet) | −10 |
| Gegner tackelt Agent mit Ball (Episode endet) | −20 |
