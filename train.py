"""Train Gomoku AlphaZero by self-play; start small and scale up."""
import argparse
from alphazero.training import run_training

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=9)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--simulations', type=int, default=50)
    p.add_argument('--games-per-iteration', type=int, default=4)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--updates', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--replay-size', type=int, default=20000)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    p.add_argument('--output', default='checkpoints/latest.pt')
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    if args.resume:
        print('Note: resumed checkpoints restore model/optimizer, but not prior replay samples.')
    run_training(args)

if __name__ == '__main__':
    main()
