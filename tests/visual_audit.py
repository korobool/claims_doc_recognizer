"""Visual audit: open the remote app in a real browser and capture
high-resolution screenshots of every meaningful UI surface so I can review
layout, spacing, typography, and button hierarchy.

Not a pass/fail suite — the goal is screenshots I can eyeball. Each step
advances the app through a state and snaps it. Runs against whatever the
remote deployment is currently serving.

Usage:
    .venv/bin/python tests/visual_audit.py
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

REMOTE_URL = "http://108.181.157.13:8011"
OUT_DIR = Path(__file__).parent / "audit"
VIEWPORT = {"width": 1600, "height": 1000}

# Sample files available on disk.
SAMPLES_DIR = Path.home() / "Downloads" / "RE_ AI доставчици"
IMAGE_SCREENSHOTS = [
    Path.home() / "Desktop" / "Screenshot 2026-04-15 at 08.26.55.png",
    Path.home() / "Desktop" / "Screenshot 2026-04-15 at 08.27.14.png",
    Path.home() / "Desktop" / "Screenshot 2026-04-15 at 08.27.31.png",
]
REAL_PDF = SAMPLES_DIR / "фактура.pdf"
REAL_DOCX = SAMPLES_DIR / "AI Readiness Questionaire for Axiom BG.docx"


async def snap(page, idx: int, name: str):
    path = OUT_DIR / f"{idx:02d}_{name}.png"
    await page.screenshot(path=str(path), full_page=True)
    print(f"  captured {path.name}")


async def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    print(f"Visual audit → {OUT_DIR}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=80)
        ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = await ctx.new_page()

        try:
            # 1. Fresh home (empty state)
            await page.goto(REMOTE_URL, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(1200)
            await snap(page, 1, "home_empty")

            # 2. Domain tab (empty sidebar / focus on center column)
            await page.click("#domainTabBtn")
            await page.wait_for_selector("#domainTab.active")
            await page.wait_for_timeout(800)
            await snap(page, 2, "domain_tab_empty_library")

            # 3. Domain with a specific one selected
            health_item = page.locator("#domainList .schema-item", has_text="Health Insurance").first
            if await health_item.count():
                await health_item.click()
                await page.wait_for_timeout(600)
                await snap(page, 3, "domain_health_selected")

            # 4. Templates tab
            await page.click("#schemasTabBtn")
            await page.wait_for_selector("#schemasTab.active")
            await page.wait_for_timeout(800)
            await snap(page, 4, "templates_tab")

            # 5. Settings tab (shows ollama models)
            await page.click("#settingsTabBtn")
            await page.wait_for_selector("#settingsTab.active")
            await page.wait_for_timeout(1500)  # give checkLlmStatus() time
            await snap(page, 5, "settings_tab")

            # 6. Viewer tab — upload 3 images to populate the library
            await page.click("#viewerTabBtn")
            await page.wait_for_selector("#viewerTab.active")
            await page.set_input_files(
                "#fileInput",
                [str(p) for p in IMAGE_SCREENSHOTS if p.exists()],
            )
            await page.wait_for_function(
                "() => document.querySelectorAll('.image-item').length >= 3",
                timeout=30000,
            )
            await page.wait_for_timeout(800)
            await snap(page, 6, "viewer_after_image_upload")

            # 7. Upload a scanned PDF and a DOCX
            if REAL_PDF.exists():
                await page.set_input_files("#fileInput", str(REAL_PDF))
                await page.wait_for_function(
                    "() => document.querySelectorAll('.image-item').length >= 4",
                    timeout=60000,
                )
                await page.wait_for_timeout(800)
                await snap(page, 7, "viewer_after_pdf_upload")

            if REAL_DOCX.exists():
                await page.set_input_files("#fileInput", str(REAL_DOCX))
                await page.wait_for_function(
                    "() => document.querySelectorAll('.image-item').length >= 5",
                    timeout=60000,
                )
                await page.wait_for_timeout(800)
                await snap(page, 8, "viewer_after_docx_upload")

            # 9. Upload 3 screenshots as a multipage document
            await page.evaluate("() => { window.prompt = () => 'Audit Multipage'; window.confirm = () => true; }")
            await page.set_input_files(
                "#fileInputMultipage",
                [str(p) for p in IMAGE_SCREENSHOTS if p.exists()],
            )
            await page.wait_for_function(
                "() => Object.values(state.documents).some(d => d.source_kind === 'multipage')",
                timeout=60000,
            )
            await page.wait_for_timeout(800)
            await snap(page, 9, "viewer_after_multipage_upload")

            # 10. Multipage doc selected, page 1 (navigator visible)
            doc_id = await page.evaluate(
                "() => Object.values(state.documents).find(d => d.source_kind === 'multipage')?.doc_id"
            )
            if doc_id:
                await page.evaluate(f"selectDocument('{doc_id}', 1)")
                await page.wait_for_timeout(800)
                await snap(page, 10, "multipage_page1_navigator")

                # 11. Page 2 via navigator click
                await page.click("#pageNavigator .page-next")
                await page.wait_for_function(
                    "() => document.querySelector('#pageNavigator .page-indicator').textContent.includes('Page 2 of 3')"
                )
                await page.wait_for_timeout(600)
                await snap(page, 11, "multipage_page2_navigator")

                # 12. Page 3
                await page.click("#pageNavigator .page-next")
                await page.wait_for_function(
                    "() => document.querySelector('#pageNavigator .page-indicator').textContent.includes('Page 3 of 3')"
                )
                await page.wait_for_timeout(600)
                await snap(page, 12, "multipage_page3_navigator")

            # 13. Full sidebar populated — scroll to show all docs
            await snap(page, 13, "sidebar_populated")

        finally:
            await page.wait_for_timeout(1500)
            await browser.close()
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
