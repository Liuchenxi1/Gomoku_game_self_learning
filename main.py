"""Quick demo: two random policies playing a complete game."""
import argparse
import numpy as np
from gomoku import GomokuEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    env = GomokuEnv(board_size=args.size)
    while not env.terminated:
        action = int(rng.choice(env.legal_moves()))
        env.step(action)
    print(env.render())
    print(f"\nMoves: {env.move_count} | Winner: {env.winner} (1=black, -1=white, 0=draw)")


if __name__ == "__main__":
    main()
