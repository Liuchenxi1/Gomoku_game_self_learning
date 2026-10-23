"""Focused tests for the Gomoku MCTS tactical safeguards.

Run from the Gomoku-Zero project root:
    python test_tactics.py
"""
from gomoku import GomokuEnv
from alphazero.mcts import MCTS


class DummyModel:
    """The tactical tests should not need a trained neural network."""
    training = False


def make_env(size=15, win_length=5, current_player=-1):
    env = GomokuEnv(size, win_length)
    env.current_player = current_player
    return env


def put(env, player, coords):
    for row, col in coords:
        env.board[row, col] = player
    env.move_count = int((env.board != 0).sum())


def assert_blocks_line(name, stones, winning_cells, size=15):
    # Black (+1) is threatening; white (-1) is to move.
    # Either open end of a four-in-a-row can be a valid block.
    env = make_env(size=size, current_player=-1)
    put(env, 1, stones)
    valid_actions = {env.size * row + col for row, col in winning_cells}
    ai = MCTS(DummyModel(), device="cpu", simulations=1)
    result = ai._tactical_action(env)
    assert result is not None, f"{name}: no tactical response found"
    chosen, reason = result
    chosen_cell = divmod(chosen, env.size)
    assert reason == "block", f"{name}: expected block, got {reason}"
    assert chosen in valid_actions, (
        f"{name}: expected block at one of {winning_cells}, got {chosen_cell}"
    )
    print(f"PASS: blocks {name} at {chosen_cell}")


def assert_takes_win(name, stones, winning_cells, size=15):
    env = make_env(size=size, current_player=-1)
    put(env, -1, stones)

    valid_actions = {
        env.size * row + col for row, col in winning_cells
    }

    ai = MCTS(DummyModel(), device="cpu", simulations=1)
    result = ai._tactical_action(env)

    assert result is not None, f"{name}: no winning move found"
    chosen, reason = result
    chosen_cell = divmod(chosen, env.size)

    assert reason == "win", f"{name}: expected win, got {reason}"
    assert chosen in valid_actions, (
        f"{name}: expected win at one of {winning_cells}, got {chosen_cell}"
    )

    print(f"PASS: takes {name} win at {chosen_cell}")


def main():
    # Single immediate threats, one per direction. The tests accept either
    # endpoint because either one prevents an immediate five-in-a-row.
    assert_blocks_line(
        "horizontal four", [(7, 3), (7, 4), (7, 5), (7, 6)], [(7, 2), (7, 7)]
    )
    assert_blocks_line(
        "vertical four", [(3, 8), (4, 8), (5, 8), (6, 8)], [(2, 8), (7, 8)]
    )
    assert_blocks_line(
        "diagonal down-right four",
        [(3, 3), (4, 4), (5, 5), (6, 6)], [(2, 2), (7, 7)]
    )
    assert_blocks_line(
        "diagonal down-left four",
        [(3, 11), (4, 10), (5, 9), (6, 8)], [(2, 12), (7, 7)]
    )

    assert_takes_win(
        "horizontal",
        [(7, 3), (7, 4), (7, 5), (7, 6)],
        [(7, 2), (7, 7)]
    )

    assert_takes_win(
        "vertical",
        [(3, 8), (4, 8), (5, 8), (6, 8)],
        [(2, 8), (7, 8)]
    )

    assert_takes_win(
        "diagonal down-right",
        [(3, 3), (4, 4), (5, 5), (6, 6)],
        [(2, 2), (7, 7)]
    )

    assert_takes_win(
        "diagonal down-left",
        [(3, 11), (4, 10), (5, 9), (6, 8)],
        [(2, 12), (7, 7)]
    )

    # If AI can win now, it should win before blocking an opponent's threat.
    env = make_env(size=15, current_player=-1)
    put(env, -1, [(7, 3), (7, 4), (7, 5), (7, 6)])
    put(env, 1, [(3, 8), (4, 8), (5, 8), (6, 8)])
    ai = MCTS(DummyModel(), device="cpu", simulations=1)
    result = ai._tactical_action(env)
    assert result is not None and result[1] == "win", (
        "AI should take its own immediate win before blocking"
    )
    valid_wins = {7 * 15 + 2, 7 * 15 + 7}
    assert result[0] in valid_wins, (
        f"Expected an immediate winning move, got {divmod(result[0], 15)}"
    )
    print("PASS: immediate win takes priority over blocking")

    print("\nAll tactical tests passed.")


if __name__ == "__main__":
    main()
