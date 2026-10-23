"""Play as human vs locally trained model in the terminal."""
import argparse
from gomoku import GomokuEnv
from alphazero.training import load_checkpoint, choose_device
from alphazero.mcts import MCTS

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='checkpoints/latest.pt')
    p.add_argument('--simulations', type=int, default=100)
    p.add_argument('--human', choices=['black', 'white'], default='black')
    p.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    args = p.parse_args()
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.model, device)
    env = GomokuEnv(model.board_size)
    mcts = MCTS(model, device, args.simulations)
    human = 1 if args.human == 'black' else -1
    while not env.terminated:
        print('\n' + env.render())
        if env.current_player == human:
            try:
                raw = input('Your row,col (or q): ').strip()
                if raw.lower() == 'q':
                    return
                row, col = map(int, raw.split(','))
                action = row * env.size + col
                env.step(action)
            except (ValueError, IndexError) as exc:
                print('Invalid move:', exc)
        else:
            action, _ = mcts.search(env, temperature=0)
            print(f'AI plays: {divmod(action, env.size)}')
            env.step(action)
    print('\n' + env.render())
    print('Draw' if env.winner == 0 else f'{"Black" if env.winner == 1 else "White"} wins')

if __name__ == '__main__':
    main()
