"""
Manual control of the football environment.

Controls:
  Arrow keys  — move (up / down / left / right)
  Space or S  — shoot
  R           — reset episode
  Q           — quit
"""

import os
import sys
import tty
import termios
import select

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import FootballEnv


def _read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b'\x1b':
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                ch += os.read(fd, 2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


KEY_TO_ACTION = {
    b'\x1b[A': 0,  # up
    b'\x1b[B': 1,  # down
    b'\x1b[D': 2,  # left
    b'\x1b[C': 3,  # right
    b' ':       4,  # space = shoot
    b's':       4,
    b'S':       4,
}

LEVEL_NAMES = {1: "Level 1 — Shoot from zone", 2: "Level 2 — Dribbling vs. pass", 3: "Level 3 — Opponent"}


def _print_controls():
    print("  Arrow keys: move  |  Space / S: shoot  |  R: reset  |  Q: quit")


def main():
    env = FootballEnv()
    level_label = LEVEL_NAMES.get(env.level, f"Level {env.level}")

    state = env.reset()
    total_reward = 0
    episode = 1

    os.system('clear')
    print(f"=== Football Environment  |  {level_label} ===")
    print()
    env.render()
    print(f"\n  Score: {total_reward:+.0f}  |  Episode: {episode}")
    print()
    _print_controls()

    while True:
        key = _read_key()

        if key in (b'q', b'Q'):
            print("\nBye!\n")
            break

        if key in (b'r', b'R'):
            state = env.reset()
            total_reward = 0
            episode += 1
            os.system('clear')
            print(f"=== Football Environment  |  {level_label} ===")
            print()
            env.render()
            print(f"\n  Score: {total_reward:+.0f}  |  Episode: {episode}")
            print()
            _print_controls()
            continue

        if key not in KEY_TO_ACTION:
            continue

        action = KEY_TO_ACTION[key]
        state, reward, done = env.step(action)
        total_reward += reward

        os.system('clear')
        print(f"=== Football Environment  |  {level_label} ===")
        print()
        env.render()
        print(f"\n  Last reward: {reward:+.0f}  |  Score: {total_reward:+.0f}  |  Episode: {episode}")
        print()

        if done:
            if reward >= 10:
                print("  *** GOAL! ***")
            else:
                print("  --- Episode ended ---")
            print()
            print("  Press R to play again or Q to quit.")
        else:
            _print_controls()


if __name__ == "__main__":
    main()
