
from playwright.sync_api import sync_playwright

URL = "https://papergames.io/zh/五子棋"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")

    input("Enter a game and wait for the empty board, then press Enter...")

    print("\n--- Elements near the board ---")

    # Find likely board containers by looking for elements with many children.
    candidates = page.locator("div, section, table, svg").evaluate_all("""
        els => els.map(el => ({
            tag: el.tagName,
            id: el.id,
            cls: typeof el.className === 'string' ? el.className : '',
            children: el.children.length,
            rect: (() => {
                const r = el.getBoundingClientRect();
                return {
                    x: Math.round(r.x), y: Math.round(r.y),
                    w: Math.round(r.width), h: Math.round(r.height)
                };
            })()
        }))
        .filter(x => x.rect.w > 350 && x.rect.h > 350)
        .sort((a, b) => (b.rect.w * b.rect.h) - (a.rect.w * a.rect.h))
        .slice(0, 20)
    """)

    for item in candidates:
        print(item)

    print("\n--- SVG elements ---")
    svgs = page.locator("svg").evaluate_all("""
        els => els.map((el, i) => {
            const r = el.getBoundingClientRect();
            return {
                i,
                cls: el.getAttribute('class'),
                viewBox: el.getAttribute('viewBox'),
                width: Math.round(r.width),
                height: Math.round(r.height),
                children: el.children.length
            };
        })
    """)
    for item in svgs:
        print(item)

    input("\nInspection finished. Press Enter to close...")
    browser.close()

