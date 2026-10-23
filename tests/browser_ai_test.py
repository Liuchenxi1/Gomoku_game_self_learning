
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gomoku import GomokuEnv
from alphazero.training import load_checkpoint, choose_device
from alphazero.mcts import MCTS


BOARD_SIZE = 15
URL = "https://papergames.io/zh/五子棋"
BOARD_SELECTOR = "div#gomoku table.table-board"
CELL_SELECTOR = f"{BOARD_SELECTOR} td"
MODEL_PATH = ROOT / "checkpoints" / "15x15_tactics.pt"


def read_board(page):
    """Read the board as X=Black, O=White, .=Empty."""
    return page.locator(CELL_SELECTOR).evaluate_all("""
        cells => cells.map(cell => {
            if (cell.querySelector('circle.circle-dark')) return 'X';
            if (cell.querySelector('circle.circle-light')) return 'O';
            return '.';
        })
    """)


def detect_player_color(page, player_name="LIU"):
    """Detect a player's stone color from the adjacent player symbol."""
    player_name = player_name.strip().lower()

    result = page.locator(
        "div.players-container"
    ).evaluate("""
        (container, targetName) => {
            const columns = Array.from(
                container.querySelectorAll(":scope > .row > .col-6")
            );

            for (const column of columns) {
                const nameElements = Array.from(
                    column.querySelectorAll("span, div")
                );

                const nameElement = nameElements.find(el =>
                    el.children.length === 0 &&
                    (el.innerText || "").trim().toLowerCase() === targetName
                );

                if (!nameElement) continue;

                const symbol = column.querySelector(
                    "app-player-symbol circle"
                );

                if (!symbol) {
                    return {
                        found: true,
                        colorClass: null
                    };
                }

                return {
                    found: true,
                    colorClass: symbol.getAttribute("class")
                };
            }

            return {
                found: false,
                colorClass: null
            };
        }
    """, player_name)

    if not result["found"]:
        raise RuntimeError(
            f"Could not find player {player_name} in the player panel."
        )

    color_class = result["colorClass"] or ""

    if "circle-dark" in color_class:
        color = "Black"
    elif "circle-light" in color_class:
        color = "White"
    else:
        raise RuntimeError(
            f"Could not determine the color for {player_name}. "
            f"Detected class: {color_class}"
        )

    print(f"Detected player: {player_name}")
    print(f"Detected stone color: {color}")

    return color



def inspect_board_dom(page):
    """Inspect actual stones and ignore board star points."""
    results = page.locator(
        "div#gomoku table.table-board td"
    ).evaluate_all("""
        cells => cells
            .map(cell => ({
                cellClass: cell.className,
                stones: Array.from(
                    cell.querySelectorAll(
                        "circle.circle-dark, circle.circle-light"
                    )
                ).map(circle => ({
                    className: circle.getAttribute("class"),
                    fill: getComputedStyle(circle).fill
                }))
            }))
            .filter(item => item.stones.length > 0)
    """)

    print("\n--- Actual stone DOM inspection ---")
    for item in results:
        print(item)

    if not results:
        print("No stones detected on the board.")



def print_board(board):
    print("\nBoard state: X=Black, O=White, .=Empty\n")
    print("   " + " ".join(f"{c:2}" for c in range(BOARD_SIZE)))

    for row in range(BOARD_SIZE):
        cells = board[row * BOARD_SIZE:(row + 1) * BOARD_SIZE]
        print(f"{row:2} " + " ".join(f"{cell:2}" for cell in cells))


def create_env(board):
    """Convert the web board into the local game environment."""
    if len(board) != BOARD_SIZE * BOARD_SIZE:
        raise ValueError(f"Expected 225 cells, got {len(board)}")

    mapping = {"X": 1, "O": -1, ".": 0}
    env = GomokuEnv(BOARD_SIZE)
    env.board = np.array(
        [mapping[cell] for cell in board], dtype=np.int8
    ).reshape(BOARD_SIZE, BOARD_SIZE)

    black_count = int(np.count_nonzero(env.board == 1))
    white_count = int(np.count_nonzero(env.board == -1))

    if black_count not in (white_count, white_count + 1):
        raise ValueError(
            f"Unexpected stone counts: Black={black_count}, "
            f"White={white_count}. Stopping without clicking."
        )

    env.move_count = black_count + white_count
    env.current_player = 1 if black_count == white_count else -1
    env.last_move = None

    winners = set()
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            player = int(env.board[row, col])
            if player and env._is_winning_move(row, col, player):
                winners.add(player)

    if len(winners) > 1:
        raise ValueError("Both colors have winning lines.")

    if winners:
        env.winner = winners.pop()
        env.terminated = True
    elif env.move_count == BOARD_SIZE * BOARD_SIZE:
        env.winner = 0
        env.terminated = True

    return env, black_count, white_count



def main():
    if not MODEL_PATH.exists():
        print(f"Model checkpoint not found: {MODEL_PATH}")
        return

    device = choose_device("mps")
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_PATH}")

    model, _ = load_checkpoint(str(MODEL_PATH), device)

    if model.board_size != BOARD_SIZE:
        print(f"Expected a {BOARD_SIZE}x{BOARD_SIZE} model.")
        return

    mcts = MCTS(model, device, simulations=100)
    print("Model loaded successfully.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            page.goto(
                URL,
                wait_until="domcontentloaded",
                timeout=60000,
            )

            input(
                "Start or join a Gomoku game. "
                "When the board is visible, press Enter here..."
            )

            page.locator(BOARD_SELECTOR).wait_for(
                state="visible",
                timeout=30000,
            )

            print("\n--- Player color detection diagnostics ---")

            player_details = page.locator("body").evaluate("""
                body => {
                    return ["liu", "Paper Man"].map(name => {
                        const elements = Array.from(
                            body.querySelectorAll("*")
                        ).filter(el =>
                            el.children.length === 0 &&
                            (el.innerText || "").trim().toLowerCase() ===
                                name.toLowerCase()
                        );

                        const element = elements[0];

                        if (!element) {
                            return {
                                player: name,
                                found: false
                            };
                        }

                        let node = element;
                        const ancestors = [];

                        for (let i = 0; i < 5 && node; i++) {
                            ancestors.push({
                                tag: node.tagName,
                                className: String(node.className),
                                html: node.outerHTML.slice(0, 2000)
                            });
                            node = node.parentElement;
                        }

                        return {
                            player: name,
                            found: true,
                            ancestors: ancestors
                        };
                    });
                }
            """)

            for player in player_details:
                print(player)

            ai_color = detect_player_color(page, "LIU")
            ai_player = 1 if ai_color == "Black" else -1

            print(f"AI will play as {ai_color}.")
            print("The script will ask before every AI move.")

            previous_board = None

            while True:
                board = read_board(page)

                try:
                    env, black_count, white_count = create_env(board)
                except ValueError as exc:
                    print(f"\nBoard validation failed: {exc}")
                    print("Stopping to avoid an incorrect move.")
                    print_board(board)
                    break

                print("\n--- Current board ---")
                print_board(board)
                print(f"Black stones: {black_count}")
                print(f"White stones: {white_count}")

                if env.terminated:
                    if env.winner == 1:
                        print("Game over. Black wins.")
                    elif env.winner == -1:
                        print("Game over. White wins.")
                    else:
                        print("Game over. Draw.")
                    break

                current_player = env.current_player
                current_color = (
                    "Black" if current_player == 1 else "White"
                )

                print(f"Next color based on board counts: {current_color}")

                if current_player != ai_player:
                    print(
                        f"It is {current_color}'s turn, "
                        "not the AI's turn."
                    )
                    print(
                        "Waiting for the opponent to make a move "
                        "on the website..."
                    )

                    starting_board = board
                    move_detected = False

                    for _ in range(120):
                        page.wait_for_timeout(1000)
                        updated_board = read_board(page)

                        if updated_board != starting_board:
                            move_detected = True
                            print("Board change detected.")
                            break

                    if not move_detected:
                        print(
                            "No board change detected within 120 seconds."
                        )
                        print(
                            "Check whose turn it is and whether "
                            "the opponent is still connected."
                        )

                        confirmation = input(
                            "Type RETRY to keep waiting, "
                            "or STOP to end: "
                        ).strip().upper()

                        if confirmation == "RETRY":
                            continue

                        print("Stopping the game loop.")
                        break

                    previous_board = starting_board
                    continue

                confirmation = input(
                    f"\nAI is ready to play {ai_color}. "
                    "Type YES to calculate a move, or STOP to quit: "
                ).strip().upper()

                if confirmation == "STOP":
                    print("Stopped by user.")
                    break

                if confirmation != "YES":
                    print("No move made. Returning to board check.")
                    continue

                print("Running MCTS search...")
                action, _ = mcts.search(env, temperature=0)
                row, col = divmod(action, BOARD_SIZE)

                if env.board[row, col] != 0:
                    print("ERROR: The selected cell is occupied.")
                    print("Stopping to avoid an invalid move.")
                    break

                print("\n--- Proposed AI move ---")
                print(f"Color: {ai_color}")
                print(f"Row: {row}")
                print(f"Column: {col}")

                confirmation = input(
                    "Type CLICK to place the move, "
                    "or anything else to cancel: "
                ).strip().upper()

                if confirmation != "CLICK":
                    print("Move cancelled.")
                    continue

                cell = page.locator(
                    f"{BOARD_SELECTOR} td.cell-{row}-{col}"
                )

                if cell.count() != 1:
                    print("ERROR: Could not identify the target cell.")
                    break

                board_before_click = read_board(page)

                if board_before_click != board:
                    print(
                        "The board changed during calculation. "
                        "No click will be made; refreshing the state."
                    )
                    continue

                if board_before_click[row * BOARD_SIZE + col] != ".":
                    print("ERROR: The target cell is occupied.")
                    continue

                cell.click()

                try:
                    cell.locator(
                        "circle.circle-dark, circle.circle-light"
                    ).wait_for(
                        state="attached",
                        timeout=10000,
                    )
                except Exception as exc:
                    print("WARNING: No stone detected after clicking.")
                    print(f"Error: {type(exc).__name__}: {exc}")
                    print("Stopping for inspection.")
                    print(f"Cell HTML: {cell.inner_html()}")
                    page.screenshot(
                        path=str(
                            ROOT / "tests" / "browser_click_debug.png"
                        ),
                        full_page=True,
                    )
                    break

                board_after_click = read_board(page)
                index = row * BOARD_SIZE + col
                actual_stone = board_after_click[index]
                expected_stone = "X" if ai_player == 1 else "O"

                print("\n--- Move verification ---")
                print(f"Expected stone: {expected_stone}")
                print(f"Actual stone: {actual_stone}")

                if actual_stone != expected_stone:
                    print(
                        "WARNING: The placed color does not match "
                        "the selected AI color. Stopping."
                    )
                    print_board(board_after_click)
                    break

                changed_cells = [
                    i
                    for i, (before, after) in enumerate(
                        zip(board_before_click, board_after_click)
                    )
                    if before != after
                ]

                print(f"Changed cells: {len(changed_cells)}")

                if len(changed_cells) != 1:
                    print(
                        "WARNING: The board changed in an unexpected "
                        "way. Stopping for inspection."
                    )
                    print_board(board_after_click)
                    break

                print("AI move verified successfully.")
                print_board(board_after_click)

                previous_board = board_after_click

        except KeyboardInterrupt:
            print("\nStopped by user.")

        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}")

        finally:
            input("Press Enter to close the browser...")
            browser.close()

if __name__ == "__main__":
    main()