
from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")

    input("Enter a game and wait for the board, then press Enter...")

    board = page.locator("#gomoku")
    cells = board.locator("table.table-board td")

    print("Total cells:", cells.count())

    # Inspect a cell before a move.
    cell = board.locator("td.cell-7-7")
    print("\nCenter cell BEFORE move:")
    print("Class:", cell.get_attribute("class"))
    print("HTML:", cell.inner_html())

    input(
        "\nIn the browser, make ONE legal move manually. "
        "Wait for the stone to appear, then press Enter here..."
    )

    print("\n--- Occupied cells ---")
    for i in range(cells.count()):
        current = cells.nth(i)
        html = current.inner_html()
        classes = current.get_attribute("class") or ""

        if html.strip() not in ("", "<!----><!---->") or "clickable" not in classes:
            print(
                "Cell:", classes,
                "\nHTML:", html[:500],
                "\n"
            )

    print("\n--- Center cell AFTER move ---")
    print("Class:", cell.get_attribute("class"))
    print("HTML:", cell.inner_html())

    input("Press Enter to close...")
    browser.close()

from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")

    input("Enter a game and wait for the board, then press Enter...")

    board = page.locator("#gomoku")
    print("Board found:", board.count())

    table = board.locator("table.table-board")
    print("Board tables:", table.count())

    rows = table.locator("tr")
    print("Rows:", rows.count())

    for r in range(min(rows.count(), 3)):
        row = rows.nth(r)
        cells = row.locator("td")
        print(f"\nRow {r}: {cells.count()} cells")

        for c in range(min(cells.count(), 4)):
            cell = cells.nth(c)
            print(
                f"  Cell ({r}, {c}):",
                "class =", cell.get_attribute("class"),
                "| HTML =", cell.inner_html()[:250],
            )

    print("\n--- Board event listeners ---")
    print(
        "DOM inspection completed. Browser event listeners are not "
        "directly exposed by this script."
    )

    input("Press Enter to close the browser...")
    browser.close()

