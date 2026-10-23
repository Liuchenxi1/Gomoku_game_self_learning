"""Self-play data generation, policy/value updates, model checkpointing."""
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from gomoku import GomokuEnv
from alphazero.mcts import MCTS
from alphazero.network import PolicyValueNet


def choose_device(name='auto'):
    if name != 'auto':
        if name == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; check your Apple Silicon PyTorch install')
        return torch.device(name)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def self_play(model, device, size=9, simulations=50, temperature_moves=12, seed=None):
    env = GomokuEnv(size)
    tree = MCTS(model, device, simulations=simulations, seed=seed)
    history = []
    while not env.terminated:
        player = env.current_player
        temperature = 1.0 if env.move_count < temperature_moves else 0.0
        action, probs = tree.search(env, temperature=temperature, add_noise=True)
        history.append((env.observation().copy(), probs, player))
        env.step(action)
    # Each position gets terminal outcome from its own player's perspective.
    samples = [(obs, probs, np.float32(0 if env.winner == 0 else (1 if env.winner == player else -1)))
               for obs, probs, player in history]
    return samples, env.winner


def train_step(model, optimizer, samples, device, batch_size=64, updates=4, rng=None):
    if not samples:
        raise ValueError('Replay buffer empty')
    rng = rng or np.random.default_rng()
    model.train()
    losses = []
    for _ in range(updates):
        batch = [samples[i] for i in rng.integers(len(samples), size=batch_size)]
        states = torch.from_numpy(np.stack([x[0] for x in batch])).to(device)
        targets_p = torch.from_numpy(np.stack([x[1] for x in batch])).to(device)
        targets_v = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device)
        logits, values = model(states)
        policy_loss = -(targets_p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values, targets_v)
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append((float(policy_loss.detach()), float(value_loss.detach())))
    return np.mean(losses, axis=0).tolist()


def save_checkpoint(path, model, optimizer=None, iteration=0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {'board_size': model.board_size, 'channels': model.channels, 'blocks': model.blocks,
               'model': model.state_dict(), 'iteration': iteration}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, device, with_optimizer=False):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = PolicyValueNet(payload['board_size'], payload['channels'], payload['blocks']).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, payload


def run_training(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device)
        start = checkpoint['iteration']
    else:
        model = PolicyValueNet(args.size, args.channels, args.blocks).to(device)
        checkpoint, start = {}, 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    replay = deque(maxlen=args.replay_size)
    print(f'Device: {device} | board: {model.board_size} | search: {args.simulations} simulations')
    for iteration in range(start + 1, start + args.iterations + 1):
        winners = []
        for game in range(args.games_per_iteration):
            samples, winner = self_play(model, device, model.board_size, args.simulations,
                                        seed=int(rng.integers(2**31)))
            replay.extend(samples)
            winners.append(winner)
        p_loss, v_loss = train_step(model, optimizer, replay, device, args.batch_size,
                                     args.updates, rng)
        save_checkpoint(args.output, model, optimizer, iteration)
        print(f'iteration {iteration} | positions={len(replay)} | '
              f'B/W/draw={winners.count(1)}/{winners.count(-1)}/{winners.count(0)} | '
              f'policy_loss={p_loss:.4f} value_loss={v_loss:.4f} | saved={args.output}', flush=True)
