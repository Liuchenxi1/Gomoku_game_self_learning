
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(URL, wait_until="domcontentloaded")

        print("Page title:", page.title())
        input(
            "Enter a game in the browser. Wait until the board appears, "
            "then return here and press Enter."
        )

        print("\n--- Frames ---")
        for i, frame in enumerate(page.frames):
            print(f"\nFrame {i}")
            print("URL:", frame.url[:300])
            try:
                print("Title:", frame.title())
                print("Canvas:", frame.locator("canvas").count())
                print("SVG:", frame.locator("svg").count())
                print("Buttons:", frame.locator("button").count())
                print("Images:", frame.locator("img").count())
            except Exception as e:
                print("Inspection error:", str(e)[:200])

        Path("debug").mkdir(exist_ok=True)
        page.screenshot(path="../debug/papergames.png", full_page=True)
        print("\nScreenshot saved: debug/papergames.png")
        input("Press Enter to close the browser...")
        browser.close()

if __name__ == "__main__":
    main()
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        page.goto(URL, wait_until="domcontentloaded")
        print("Page title:", page.title())
        print("Current URL:", page.url)

        input(
            "\nIn the opened browser, manually enter a Gomoku game "
            "and wait until the board appears. Then press Enter here..."
        )

        print("Canvas elements:", page.locator("canvas").count())
        print("Iframe elements:", page.locator("iframe").count())
        print("SVG elements:", page.locator("svg").count())
        print("Button elements:", page.locator("button").count())

        Path("debug").mkdir(exist_ok=True)
        page.screenshot(path="../debug/papergames.png", full_page=True)

        print("Screenshot saved to debug/papergames.png")
        browser.close()

if __name__ == "__main__":
    main()
