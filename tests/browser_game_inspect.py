
from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)

        input(
            "Join a Gomoku game and wait until the board is visible. "
            "Press Enter here to inspect the page..."
        )

        print("\n--- Page title ---")
        print(page.title())

        print("\n--- Visible page text ---")
        print(page.locator("body").inner_text()[:12000])

        print("\n--- Buttons ---")
        buttons = page.locator("button").evaluate_all("""
            elements => elements.map(button => ({
                text: button.innerText,
                ariaLabel: button.getAttribute('aria-label'),
                title: button.getAttribute('title'),
                disabled: button.disabled
            }))
        """)
        for button in buttons:
            print(button)

        print("\n--- Board-related elements ---")
        print("Board cells:", page.locator("div#gomoku table.table-board td").count())

        input("\nInspection complete. Press Enter to close the browser.")

    finally:
        browser.close()
