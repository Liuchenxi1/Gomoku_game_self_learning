
from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

def inspect_board(page):
    cells = page.locator("#gomoku table.table-board td")
    print(f"\nTotal cells: {cells.count()}")
    print("Occupied cells:")

    found = 0

    for i in range(cells.count()):
        cell = cells.nth(i)
        symbol = cell.locator("svg.symbol")

        if symbol.count() > 0:
            row, col = divmod(i, 15)
            svg_class = symbol.get_attribute("class")
            circles = symbol.locator("circle").evaluate_all(
                "(els) => els.map(e => e.getAttribute('class') || '')"
            )

            print(
                f"({row}, {col}) "
                f"svg={svg_class}, circles={circles}"
            )
            found += 1

    if found == 0:
        print("No stones detected. Make sure a move has been played.")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")

    input(
        "Enter a game. Wait until at least one stone appears, "
        "then press Enter here..."
    )

    inspect_board(page)

    input(
        "\nPlay another move manually and wait for the stone to appear. "
        "Press Enter here to inspect again..."
    )

    inspect_board(page)

    input("\nPress Enter to close the browser...")
    browser.close()

from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")

    input("Enter a game and wait for the board, then press Enter...")

    cells = page.locator("#gomoku table.table-board td")
    print("Total cells:", cells.count())

    print("\nOccupied cells and their SVG classes:")

    for i in range(cells.count()):
        cell = cells.nth(i)
        symbol = cell.locator("svg.symbol")

        if symbol.count() > 0:
            row = i // 15
            col = i % 15
            svg_class = symbol.get_attribute("class")
            circle_classes = symbol.locator("circle").evaluate_all(
                "(els) => els.map(e => e.getAttribute('class'))"
            )

            print(
                f"({row}, {col}) "
                f"SVG={svg_class}, circles={circle_classes}"
            )

    input("\nPress Enter to close...")
    browser.close()

from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

def read_board(page):
    cells = page.locator("#gomoku table.table-board td")
    if cells.count() != 225:
        raise RuntimeError(
            f"Expected 225 cells, found {cells.count()}"
        )

    board = []

    for r in range(15):
        row = []
        for c in range(15):
            cell = page.locator(f"#gomoku td.cell-{r}-{c}")
            symbol = cell.locator("svg.symbol")
            if symbol.count() == 0:
                row.append(".")
            else:
                classes = symbol.locator("circle").evaluate_all(
                    "(els) => els.map(e => e.getAttribute('class') || '')"
                )

                if any("circle-dark" in x for x in classes):
                    row.append("X")
                elif any("circle-light" in x for x in classes):
                    row.append("O")
                else:
                    row.append("?")

        board.append(row)

    return board

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")

    input("Enter a game and wait until the board is visible, then press Enter...")

    board = read_board(page)

    print("\nBoard state (X=dark, O=light, .=empty):")
    print("   " + " ".join(f"{c:2}" for c in range(15)))

    for r, row in enumerate(board):
        print(f"{r:2} " + "  ".join(row))

    input("\nInspection complete. Press Enter to close...")
    browser.close()

