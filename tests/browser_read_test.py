
from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"
BOARD_SELECTOR = "div#gomoku table.table-board"
CELL_SELECTOR = f"{BOARD_SELECTOR} td"
BOARD_SIZE = 15


def read_board(page):
    """Read the board as a list: X=Black, O=White, .=Empty."""
    return page.locator(CELL_SELECTOR).evaluate_all("""
        cells => cells.map(cell => {
            if (cell.querySelector('circle.circle-dark')) {
                return 'X';
            }
            if (cell.querySelector('circle.circle-light')) {
                return 'O';
            }
            return '.';
        })
    """)


def print_board(board):
    """Print the board with row and column coordinates."""
    print("\\nBoard state: X=Black, O=White, .=Empty\\n")
    print("   " + " ".join(f"{col:2}" for col in range(BOARD_SIZE)))

    for row in range(BOARD_SIZE):
        start = row * BOARD_SIZE
        row_cells = board[start:start + BOARD_SIZE]
        print(f"{row:2} " + " ".join(f"{cell:2}" for cell in row_cells))

    print(f"\\nTotal cells read: {len(board)}")
    print(f"Black stones: {board.count('X')}")
    print(f"White stones: {board.count('O')}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)

            input(
                "Enter a Gomoku game in the browser. "
                "Once the board is visible, press Enter here..."
            )

            board_locator = page.locator(BOARD_SELECTOR)
            board_locator.wait_for(state="visible", timeout=30000)

            cell_count = page.locator(CELL_SELECTOR).count()

            if cell_count != BOARD_SIZE * BOARD_SIZE:
                print(
                    f"Unexpected board size: found {cell_count} cells, "
                    f"expected {BOARD_SIZE * BOARD_SIZE}."
                )
                return

            board = read_board(page)

            if len(board) != BOARD_SIZE * BOARD_SIZE:
                print(f"Failed to read the complete board: {len(board)} cells.")
                return

            print_board(board)

            input("\\nInspection complete. Press Enter to close the browser...")

        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}")
            print(
                "Make sure the Gomoku game is open and the board is visible."
            )

            input("Press Enter to close the browser...")

        finally:
            browser.close()


if __name__ == "__main__":
    main()
