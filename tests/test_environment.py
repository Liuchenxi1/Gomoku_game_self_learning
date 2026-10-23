import unittest
import numpy as np
from gomoku import GomokuEnv


class TestGomokuEnv(unittest.TestCase):
    def test_reset_observation_and_masks(self):
        env = GomokuEnv(9)
        self.assertEqual(env.observation().shape, (3, 9, 9))
        self.assertEqual(env.observation().dtype, np.float32)
        self.assertEqual(len(env.legal_moves()), 81)
        self.assertTrue(env.observation()[2].all())
        env.step(0)
        self.assertEqual(env.current_player, -1)
        self.assertEqual(env.observation()[1, 0, 0], 1)
        self.assertFalse(env.observation()[2].any())
        self.assertEqual(len(env.legal_moves()), 80)

    def test_occupied_and_out_of_bounds(self):
        env = GomokuEnv(9)
        env.step(0)
        for action in (0, -1, 81, 1.5):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.move_count, 1)

    def test_all_win_directions(self):
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            with self.subTest(direction=(dr, dc)):
                env = GomokuEnv(9)
                row, col = (0, 4) if dc < 0 else (0, 0)
                black = [(row + k * dr) * 9 + col + k * dc for k in range(5)]
                white = [8 * 9 + k for k in range(4)]
                for i, black_action in enumerate(black):
                    result = env.step(black_action)
                    if i < 4:
                        self.assertFalse(result.terminated)
                        env.step(white[i])
                self.assertTrue(result.terminated)
                self.assertEqual(result.winner, 1)
                self.assertEqual(env.legal_moves().size, 0)
                with self.assertRaises(ValueError):
                    env.step(10)

    def test_white_wins(self):
        env = GomokuEnv(9)
        black = [72, 74, 76, 78, 62]
        white = [0, 1, 2, 3, 4]
        for b, w in zip(black, white):
            env.step(b)
            result = env.step(w)
        self.assertEqual(result.winner, -1)

    def test_draw_on_small_board(self):
        # On a 2x2 board with length 2, a filled draw is impossible.
        # Verify a draw by using a known 3x3 full board with no 3-in-a-row.
        env = GomokuEnv(3, 3)
        # X O X
        # X O O
        # O X X
        moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
        for move in moves:
            result = env.step(move)
        self.assertTrue(result.terminated)
        self.assertEqual(result.winner, 0)


if __name__ == "__main__":
    unittest.main()
