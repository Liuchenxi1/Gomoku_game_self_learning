from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://papergames.io/zh/五子棋")
    print("Page title:", page.title())
    input("网页打开了。检查一下浏览器，然后按 Enter 退出...")
    browser.close()
