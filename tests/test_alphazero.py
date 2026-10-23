import unittest
import numpy as np
import torch
from gomoku import GomokuEnv
from alphazero.network import PolicyValueNet
from alphazero.mcts import MCTS
from alphazero.training import self_play, train_step, save_checkpoint, load_checkpoint
import tempfile
from pathlib import Path

class TestAlphaZero(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = PolicyValueNet(5, 8, 1)
        self.device = torch.device('cpu')

    def test_network_shapes(self):
        logits, values = self.model(torch.zeros(2, 3, 5, 5))
        self.assertEqual(logits.shape, (2, 25))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(torch.all(values.abs() <= 1))

    def test_mcts_legal_and_does_not_mutate(self):
        env = GomokuEnv(5)
        env.step(12)
        before = env.board.copy()
        action, probs = MCTS(self.model, self.device, simulations=4, seed=0).search(env)
        self.assertNotEqual(action, 12)
        self.assertEqual(probs[12], 0)
        self.assertAlmostEqual(float(probs.sum()), 1, places=5)
        np.testing.assert_array_equal(env.board, before)

    def test_self_play_training_checkpoint(self):
        samples, winner = self_play(self.model, self.device, size=5, simulations=2, seed=0)
        self.assertGreater(len(samples), 0)
        self.assertTrue(all(s.shape == (3, 5, 5) and p.shape == (25,) and z in (-1, 0, 1) for s,p,z in samples))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        losses = train_step(self.model, optimizer, samples, self.device, batch_size=4, updates=1)
        self.assertTrue(all(np.isfinite(losses)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(path, self.model, optimizer, 1)
            restored, metadata = load_checkpoint(path, self.device)
            self.assertEqual(metadata['iteration'], 1)
            self.assertEqual(restored.board_size, 5)

if __name__ == '__main__':
    unittest.main()
