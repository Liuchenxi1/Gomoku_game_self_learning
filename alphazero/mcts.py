"""PUCT Monte Carlo tree search with tactical win/block safeguards."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import torch


@dataclass
class Node:
    prior: float = 1.0
    visits: int = 0
    total_value: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def mean_value(self):
        return self.total_value / self.visits if self.visits else 0.0


class MCTS:
    def __init__(self, model, device, simulations=100, c_puct=1.5, seed=None):
        if simulations < 1:
            raise ValueError("simulations must be >= 1")
        self.model, self.device = model, device
        self.simulations, self.c_puct = simulations, c_puct
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def _evaluate(self, env):
        x = torch.from_numpy(env.observation()[None]).to(self.device)
        was_training = self.model.training
        self.model.eval()
        logits, value = self.model(x)
        if was_training:
            self.model.train()
        logits = logits[0].float().cpu().numpy()
        legal = env.legal_moves()
        z = logits[legal] - logits[legal].max()
        p = np.exp(z)
        p /= p.sum()
        return dict(zip(legal.tolist(), p.tolist())), float(value.item())

    @staticmethod
    def _copy_env(env):
        """Copy a Gomoku environment without mutating the live game."""
        sim = env.__class__(env.size, env.win_length)
        sim.board = env.board.copy()
        sim.current_player = env.current_player
        sim.move_count = env.move_count
        sim.terminated = env.terminated
        sim.winner = env.winner
        sim.last_move = env.last_move
        return sim

    @staticmethod
    def _winning_actions(env, player):
        """Return legal moves that win immediately for the specified player."""
        wins = []
        for action in env.legal_moves():
            row, col = divmod(int(action), env.size)
            # A winning move can only be made by the side to move. For the
            # opponent, temporarily place their stone and inspect the rule.
            board_value = env.board[row, col]
            env.board[row, col] = player
            won = env._is_winning_move(row, col, player)
            env.board[row, col] = board_value
            if won:
                wins.append(int(action))
        return wins

    def _tactical_action(self, env):
        """Return an immediate win, or block an opponent's immediate win."""
        current = env.current_player
        own_wins = self._winning_actions(env, current)
        if own_wins:
            return own_wins[0], "win"
        opponent_wins = self._winning_actions(env, -current)
        if opponent_wins:
            return opponent_wins[0], "block"
        return None

    def search(self, env, temperature=1.0, add_noise=False):
        if env.terminated:
            raise ValueError("Cannot search a finished game")

        # Tactical safeguards are used for the actual selected move.
        # Return a one-hot policy so callers receive a valid action distribution.
        tactical = self._tactical_action(env)
        if tactical is not None:
            action, _reason = tactical
            probs = np.zeros(env.size * env.size, dtype=np.float32)
            probs[action] = 1.0
            return action, probs

        root = Node()
        priors, _ = self._evaluate(env)
        if add_noise and len(priors) > 1:
            actions = list(priors)
            noise = self.rng.dirichlet([0.3] * len(actions))
            priors = {
                a: 0.75 * priors[a] + 0.25 * float(n)
                for a, n in zip(actions, noise)
            }
        root.children = {a: Node(prior=p) for a, p in priors.items()}

        for _ in range(self.simulations):
            sim = self._copy_env(env)
            node = root
            path = []
            value = 0.0

            while node.children:
                parent_visits = max(1, node.visits)
                action, child = max(
                    node.children.items(),
                    key=lambda kv: (
                        -kv[1].mean_value
                        + self.c_puct * kv[1].prior
                        * math.sqrt(parent_visits) / (1 + kv[1].visits)
                    ),
                )
                result = sim.step(action)
                path.append((node, child))
                node = child
                if result.terminated:
                    # From the perspective of the player who would move next,
                    # a win for the previous mover is a loss.
                    value = 0.0 if result.winner == 0 else -1.0
                    break
            else:
                if not path:
                    raise RuntimeError("Search root has no legal moves")
                priors, value = self._evaluate(sim)
                node.children = {a: Node(prior=p) for a, p in priors.items()}

            # Back up values with alternating-player perspective.
            for parent, child in reversed(path):
                value = -value
                child.visits += 1
                child.total_value += -value
                parent.visits += 1

        counts = np.zeros(env.size * env.size, dtype=np.float64)
        for action, child in root.children.items():
            counts[action] = child.visits

        if temperature <= 1e-8:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        else:
            counts = counts ** (1.0 / temperature)
            total = counts.sum()
            if total <= 0:
                legal = env.legal_moves()
                probs = np.zeros(env.size * env.size, dtype=np.float64)
                probs[legal] = 1.0 / len(legal)
            else:
                probs = counts / total

        action = int(self.rng.choice(len(probs), p=probs))
        return action, probs.astype(np.float32)
