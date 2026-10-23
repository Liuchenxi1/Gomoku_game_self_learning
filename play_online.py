"""Play Gomoku-Zero against PaperGames.io. Place this file in the project root."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gomoku import GomokuEnv
from alphazero.training import load_checkpoint, choose_device
from alphazero.mcts import MCTS

URL = "https://papergames.io/zh/五子棋"
BOARD_SELECTOR = "div#gomoku table.table-board"
CELL_SELECTOR = f"{BOARD_SELECTOR} td"


def read_board(page, size, timeout_ms=15000):
    """Read a complete board as X=Black, O=White, .=Empty."""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        try:
            board = page.locator(CELL_SELECTOR).evaluate_all("""
                cells => cells.map(cell => {
                    if (cell.querySelector('circle.circle-dark')) return 'X';
                    if (cell.querySelector('circle.circle-light')) return 'O';
                    return '.';
                })
            """)
            if len(board) == size * size:
                return board
        except Exception:
            pass
        page.wait_for_timeout(250)
    raise RuntimeError(f"Could not read the full {size}x{size} board.")


def detect_player_color(page, player_name="LIU"):
    """Detect the local player's color; return None if the page structure differs."""
    try:
        result = page.locator("div.players-container").evaluate("""
            (container, targetName) => {
                const columns = Array.from(
                    container.querySelectorAll(":scope > .row > .col-6")
                );
                for (const column of columns) {
                    const nameElement = Array.from(
                        column.querySelectorAll("span, div")
                    ).find(el => el.children.length === 0 &&
                        (el.innerText || "").trim().toLowerCase() === targetName);
                    if (!nameElement) continue;
                    const symbol = column.querySelector("app-player-symbol circle");
                    return {found: true, colorClass: symbol?.getAttribute("class") || ""};
                }
                return {found: false, colorClass: ""};
            }
        """, player_name.strip().lower())
    except Exception:
        return None

    color_class = result.get("colorClass", "")
    if result.get("found") and "circle-dark" in color_class:
        return 1, "Black"
    if result.get("found") and "circle-light" in color_class:
        return -1, "White"
    return None


def find_winner(board, size):
    """Return 1 for Black, -1 for White, None if there is no five-in-a-row."""
    values = {"X": 1, "O": -1, ".": 0}
    grid = np.asarray([values[cell] for cell in board], dtype=np.int8).reshape(size, size)
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    winners = set()
    for row in range(size):
        for col in range(size):
            player = int(grid[row, col])
            if not player:
                continue
            for dr, dc in directions:
                end_r, end_c = row + 4 * dr, col + 4 * dc
                if not (0 <= end_r < size and 0 <= end_c < size):
                    continue
                if all(int(grid[row + i * dr, col + i * dc]) == player for i in range(5)):
                    winners.add(player)
    if len(winners) > 1:
        raise RuntimeError("Both colors appear to have winning lines; stopping safely.")
    return next(iter(winners)) if winners else None


def make_env(board, size):
    """Build the model environment using standard Black-first Gomoku turns."""
    mapping = {"X": 1, "O": -1, ".": 0}
    env = GomokuEnv(size)
    env.board = np.asarray([mapping[cell] for cell in board], dtype=np.int8).reshape(size, size)
    black = int(np.count_nonzero(env.board == 1))
    white = int(np.count_nonzero(env.board == -1))

    if black not in (white, white + 1):
        raise RuntimeError(
            f"Unexpected stone counts: Black={black}, White={white}. "
            "The game may not have started normally."
        )

    env.move_count = black + white
    env.current_player = 1 if black == white else -1
    env.last_move = None
    env.winner = None
    env.terminated = False

    winner = find_winner(board, size)
    if winner is not None:
        env.winner = winner
        env.terminated = True
    elif env.move_count == size * size:
        env.winner = 0
        env.terminated = True
    return env, black, white


def immediate_winning_moves(env, player, size):
    wins = []
    for row in range(size):
        for col in range(size):
            if env.board[row, col] != 0:
                continue
            env.board[row, col] = player
            try:
                if env._is_winning_move(row, col, player):
                    wins.append(row * size + col)
            finally:
                env.board[row, col] = 0
    return wins


def choose_action(env, mcts, ai_player, size):
    own_wins = immediate_winning_moves(env, ai_player, size)
    if own_wins:
        return own_wins[0], "win"

    opponent_wins = immediate_winning_moves(env, -ai_player, size)
    if len(opponent_wins) == 1:
        return opponent_wins[0], "block"

    action, _ = mcts.search(env, temperature=0)
    return int(action), "MCTS"


def print_board(board, size):
    print("\nFinal board (X=Black, O=White):")
    print("   " + " ".join(f"{col:2}" for col in range(size)))
    for row in range(size):
        cells = board[row * size:(row + 1) * size]
        print(f"{row:2} " + " ".join(f"{cell:2}" for cell in cells))


def print_result(env):
    if env.winner == 1:
        print("Game over: Black wins.")
    elif env.winner == -1:
        print("Game over: White wins.")
    elif env.winner == 0:
        print("Game over: draw.")


def main():
    parser = argparse.ArgumentParser(description="Play Gomoku-Zero on PaperGames.io.")
    parser.add_argument("--model", default="checkpoints/15x15_tactics.pt")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--auto", action="store_true", help="Calculate without waiting for Enter on AI turns.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        print(f"Model checkpoint not found: {model_path}")
        return 1

    device = choose_device(args.device)
    print(f"Gomoku-Zero | Model: {model_path.name} | Device: {device} | MCTS: {args.simulations}")
    model, _ = load_checkpoint(str(model_path), device)
    size = model.board_size
    mcts = MCTS(model, device, simulations=args.simulations)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.locator(BOARD_SELECTOR).wait_for(state="visible", timeout=60000)
            input("Join/start a game in the browser, then press Enter here... ")

            board = read_board(page, size)
            detected = detect_player_color(page, "LIU")
            if detected:
                ai_player, ai_color = detected
                print(f"AI color detected: {ai_color}.")
            else:
                initial_black = board.count("X")
                initial_white = board.count("O")
                print(f"Current stones: Black {initial_black}, White {initial_white}")
                color_input = input("Which color should the AI play? Enter B or W: ").strip().upper()
                if color_input not in ("B", "W"):
                    print("Cancelled.")
                    return 1
                ai_player = 1 if color_input == "B" else -1
                ai_color = "Black" if ai_player == 1 else "White"
                print(f"AI color: {ai_color}.")

            waiting = False
            while True:
                board = read_board(page, size)
                env, black, white = make_env(board, size)

                if env.terminated:
                    print_result(env)
                    print_board(board, size)
                    break

                if env.current_player != ai_player:
                    if not waiting:
                        opponent = "Black" if env.current_player == 1 else "White"
                        print(f"Waiting for {opponent} to move...")
                        waiting = True
                    # Poll the DOM directly so an opponent move is detected even
                    # if it happens between two browser waits.
                    page.wait_for_timeout(400)
                    continue

                waiting = False
                if not args.auto:
                    answer = input("AI turn: press Enter to calculate, or q to quit: ").strip().lower()
                    if answer == "q":
                        print("Stopped by user.")
                        break
                    # The opponent might have moved while the prompt was open.
                    fresh_board = read_board(page, size)
                    if fresh_board != board:
                        continue
                    board = fresh_board
                    env, black, white = make_env(board, size)
                    if env.terminated:
                        print_result(env)
                        print_board(board, size)
                        break
                    if env.current_player != ai_player:
                        continue

                print("AI thinking...")
                action, reason = choose_action(env, mcts, ai_player, size)
                row, col = divmod(int(action), size)
                if env.board[row, col] != 0:
                    print("AI selected an occupied cell; retrying.")
                    continue

                print(f"AI move: row={row}, col={col} ({reason})")
                target = page.locator(f"{BOARD_SELECTOR} td.cell-{row}-{col}")
                if target.count() != 1:
                    raise RuntimeError("Could not identify the target cell.")
                if read_board(page, size)[row * size + col] != ".":
                    print("Board changed before the click; checking again.")
                    continue

                target.click()
                expected_selector = "circle.circle-dark" if ai_player == 1 else "circle.circle-light"
                try:
                    target.locator(expected_selector).wait_for(state="attached", timeout=8000)
                except PlaywrightTimeoutError:
                    print("Move was not confirmed by the website; stopping safely.")
                    break

                print("Move verified.")
                page.wait_for_timeout(300)

        except KeyboardInterrupt:
            print("Stopped by user.")
        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}")
            return 1
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
