
from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded")

    input("Enter a game and wait for the board, then press Enter here...")

    for i, frame in enumerate(page.frames):
        print(f"\n{'=' * 50}")
        print(f"FRAME {i}")
        print("URL:", frame.url[:250])

        try:
            print("Title:", frame.title())
            print("SVG:", frame.locator("svg").count())
            print("Images:", frame.locator("img").count())
            print("Buttons:", frame.locator("button").count())
            print("Elements with role=button:",
                  frame.locator('[role="button"]').count())
            print("Elements with role=img:",
                  frame.locator('[role="img"]').count())

            # Print a small sample of visible text for non-ad frames.
            if "about:blank" not in frame.url:
                print("Text sample:",
                      frame.locator("body").inner_text(timeout=2000)[:400]
                      .replace("\n", " | "))
        except Exception as e:
            print("Could not inspect frame:", str(e)[:150])

    input("\nInspection finished. Press Enter to close...")
    browser.close()

