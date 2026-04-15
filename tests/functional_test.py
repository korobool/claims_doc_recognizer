"""Functional test suite for claims_doc_recognizer (remote deployment).

Drives a real Chromium browser against http://108.181.157.13:8011/ and walks
through every major user flow, verifying:

    1. Core tabs render, active classes toggle correctly
    2. Domain tab lists both shipped domains, Health Insurance is active
    3. Domain selection + Set Active flips the active pointer
    4. Create + Save + Delete a throwaway domain
    5. Upload three screenshots via the file input, verify sidebar populates
    6. Run OCR + SigLIP classification on each
    7. Process with LLM (gemma4:31b) under Health Insurance domain
    8. Swap active domain to Motor Insurance and re-process the same image
       — verify the system prompt truly picks up the new domain description
    9. Template tab: create a new template via Generate-with-AI, verify the
       schema output reflects the active domain (motor terminology)

Screenshots are saved to tests/screenshots/ with a step number so the user can
review the visual trace after the run.

Run:
    .venv/bin/python tests/functional_test.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import List

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    expect,
)

REMOTE_URL = "http://108.181.157.13:8011"
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"

# Images (existing Desktop screenshots) — used as vanilla single-file uploads
# AND stapled together as a 3-page multipage document.
IMAGE_PATHS = [
    Path.home() / "Desktop" / "Screenshot 2026-04-15 at 08.26.55.png",
    Path.home() / "Desktop" / "Screenshot 2026-04-15 at 08.27.14.png",
    Path.home() / "Desktop" / "Screenshot 2026-04-15 at 08.27.31.png",
]

# Real-world test documents: one scanned PDF invoice and one DOCX
# questionnaire. These exercise the OCR-fallback path and the DOCX
# text-layer fast path respectively.
_SAMPLE_DIR = Path.home() / "Downloads" / "RE_ AI доставчици"
SCANNED_PDF_PATH = _SAMPLE_DIR / "фактура.pdf"              # 1-page scanned (no text layer)
DOCX_PATH = _SAMPLE_DIR / "AI Readiness Questionaire for Axiom BG.docx"  # DOCX with native text
MULTIPAGE_PDF_PATH = _SAMPLE_DIR / "Епикриза.pdf"            # 2-page PDF for multipage nav check

# Synthesized PDF-with-text-layer (covers the fast-path branch that the
# real samples don't exercise).
TEXT_LAYER_PDF_PATH = Path(__file__).parent / "assets" / "text_layer_sample.pdf"

OBSERVE_MS = int(os.environ.get("OBSERVE_MS", "1200"))  # visual pause per step

STEP = 0


def log(msg: str, level: str = "info") -> None:
    tag = {"info": "•", "pass": "✓", "fail": "✗", "step": "»"}.get(level, "•")
    print(f"  {tag} {msg}")


async def snap(page: Page, name: str) -> None:
    """Save a screenshot for the visual trace."""
    global STEP
    STEP += 1
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / f"{STEP:02d}_{name}.png"
    await page.screenshot(path=str(path), full_page=True)
    log(f"screenshot → {path.name}")


async def observe(page: Page, label: str = "") -> None:
    """Deliberate pause so the watching user can see what happened."""
    if label:
        log(label)
    await page.wait_for_timeout(OBSERVE_MS)


# ---------------------------------------------------------------------------
# Test steps
# ---------------------------------------------------------------------------


async def step_load_home(page: Page) -> None:
    log("Step 1: Load home", "step")
    await page.goto(REMOTE_URL, wait_until="networkidle", timeout=30000)
    title = await page.title()
    assert "Document Recognition" in title, f"unexpected title: {title}"
    log(f"title: {title}", "pass")
    await snap(page, "home")
    await observe(page, "home loaded")


async def step_verify_tabs(page: Page) -> None:
    log("Step 2: Verify all 5 main tabs exist", "step")
    for tab_id, label in [
        ("viewerTabBtn", "Document Viewer"),
        ("llmTabBtn", "LLM Results"),
        ("schemasTabBtn", "Templates"),
        ("domainTabBtn", "Domain"),
        ("settingsTabBtn", "Settings"),
    ]:
        el = page.locator(f"#{tab_id}")
        await expect(el).to_be_visible()
        text = (await el.text_content() or "").strip()
        assert label.lower() in text.lower(), f"{tab_id} text mismatch: {text!r}"
        log(f"{tab_id}: {text!r}", "pass")


async def step_open_domain_tab(page: Page) -> None:
    log("Step 3: Open Domain tab", "step")
    await page.click("#domainTabBtn")
    await page.wait_for_selector("#domainTab.active", timeout=5000)
    await page.wait_for_function(
        """() => {
            const list = document.getElementById('domainList');
            return list && list.children.length >= 2;
        }""",
        timeout=10000,
    )
    items = await page.locator("#domainList .schema-item").all()
    names = [
        (await i.locator(".schema-item-title").text_content() or "").strip()
        for i in items
    ]
    log(f"domains in sidebar: {names}")
    assert any("Health Insurance" in n for n in names), "Health Insurance missing"
    assert any("Motor Insurance" in n for n in names), "Motor Insurance missing"
    # Health should be the active one (it has the "Active" badge).
    active_label = await page.locator("#activeDomainLabel").text_content()
    assert active_label and "Health" in active_label, f"active label = {active_label!r}"
    log(f"active label: {active_label}", "pass")
    await snap(page, "domain_tab_initial")
    await observe(page, "domain tab open with 2 domains")


async def step_switch_to_motor_and_activate(page: Page) -> None:
    log("Step 4: Select motor_insurance and Set Active", "step")
    # Click the motor item by its visible title
    motor_item = page.locator(
        "#domainList .schema-item", has_text="Motor Insurance"
    ).first
    await motor_item.click()
    await page.wait_for_selector("#domainEditForm:visible", timeout=5000)
    yaml_value = await page.locator("#domainYamlEditor").input_value()
    assert "motor_insurance" in yaml_value, "motor YAML not loaded"
    assert "VIN" in yaml_value, "motor YAML missing VIN (content check failed)"
    log("motor YAML loaded, contains VIN", "pass")
    await snap(page, "motor_yaml_loaded")
    await observe(page, "motor YAML visible in editor")

    # Set Active button should be enabled
    set_active = page.locator("#setActiveDomainBtn")
    assert not await set_active.is_disabled(), "Set Active button should be enabled"
    await set_active.click()
    # Wait for the label to reflect the switch
    await page.wait_for_function(
        """() => {
            const el = document.getElementById('activeDomainLabel');
            return el && el.textContent.includes('Motor');
        }""",
        timeout=5000,
    )
    log("active label now reads Motor Insurance", "pass")
    await snap(page, "motor_is_active")
    await observe(page, "motor insurance is now active")


async def step_switch_back_to_health(page: Page) -> None:
    log("Step 5: Switch active back to Health Insurance", "step")
    health_item = page.locator(
        "#domainList .schema-item", has_text="Health Insurance"
    ).first
    await health_item.click()
    await page.wait_for_selector("#domainEditForm:visible", timeout=5000)
    await page.click("#setActiveDomainBtn")
    await page.wait_for_function(
        """() => {
            const el = document.getElementById('activeDomainLabel');
            return el && el.textContent.includes('Health');
        }""",
        timeout=5000,
    )
    log("active label back to Health Insurance", "pass")
    await snap(page, "health_is_active_again")


async def step_domain_crud_roundtrip(page: Page) -> None:
    log("Step 6: Create + Save + Delete a throwaway domain", "step")
    await page.click("#newDomainBtn")
    # A default template should be visible
    await page.wait_for_selector("#domainEditForm:visible", timeout=5000)
    yaml_val = await page.locator("#domainYamlEditor").input_value()
    assert "new_domain" in yaml_val, "new-domain template missing"

    # Replace with a test domain
    test_yaml = (
        "domain_id: qa_test_domain\n"
        "display_name: QA Test Domain\n"
        "description: |\n"
        "  A disposable domain written by the functional test suite to verify\n"
        "  create / save / delete work end-to-end.\n"
    )
    await page.fill("#domainYamlEditor", test_yaml)
    # Accept the "Schema saved" / "Domain saved" alert dialogue.
    page.once("dialog", lambda d: asyncio.create_task(d.accept()))
    await page.click("#saveDomainBtn")
    await page.wait_for_function(
        """() => Array.from(document.querySelectorAll('#domainList .schema-item-title'))
                       .some(el => el.textContent.includes('QA Test Domain'))""",
        timeout=5000,
    )
    log("QA Test Domain appears in sidebar", "pass")
    await snap(page, "qa_domain_created")
    await observe(page, "throwaway domain visible in sidebar")

    # Delete it
    page.once("dialog", lambda d: asyncio.create_task(d.accept()))  # confirm
    await page.click("#deleteDomainBtn")
    await page.wait_for_function(
        """() => !Array.from(document.querySelectorAll('#domainList .schema-item-title'))
                         .some(el => el.textContent.includes('QA Test Domain'))""",
        timeout=5000,
    )
    log("QA Test Domain removed", "pass")
    await snap(page, "qa_domain_deleted")


def ensure_text_layer_pdf() -> Path:
    """Synthesize a small text-layer PDF that we can upload to exercise the
    'PDF has native text, skip OCR' fast path. Regenerates whenever the file
    is missing so the test is self-contained."""
    TEXT_LAYER_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TEXT_LAYER_PDF_PATH.exists():
        return TEXT_LAYER_PDF_PATH
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(TEXT_LAYER_PDF_PATH), pagesize=letter)
    for page_num in range(1, 4):
        c.setFont("Helvetica-Bold", 22)
        c.drawString(72, 720, f"Synthetic Claim Report — Page {page_num}")
        c.setFont("Helvetica", 13)
        c.drawString(72, 695, "This PDF has an embedded text layer and should skip OCR.")
        c.drawString(72, 675, f"Claim number: QA-{page_num:03d}  |  Policy: POL-2025-{page_num:04d}")
        c.drawString(72, 655, "Patient: Test User  |  Date: 2026-04-15  |  Amount: 127.50 BGN")
        c.showPage()
    c.save()
    return TEXT_LAYER_PDF_PATH


async def step_upload_images(page: Page) -> List[str]:
    """Upload three screenshots as THREE separate documents."""
    log("Step 7: Upload 3 screenshots as independent documents", "step")
    await page.click("#viewerTabBtn")
    await page.wait_for_selector("#viewerTab.active", timeout=5000)

    for p in IMAGE_PATHS:
        assert p.exists(), f"missing {p}"

    await page.set_input_files("#fileInput", [str(p) for p in IMAGE_PATHS])
    # Three uploaded images produce three separate .image-item rows.
    await page.wait_for_function(
        """() => document.querySelectorAll('.image-item').length >= 3""",
        timeout=30000,
    )
    # The sidebar now groups by doc, so we grab doc IDs.
    doc_ids = await page.evaluate(
        "() => Array.from(document.querySelectorAll('.image-item')).map(i => i.dataset.docId).filter(Boolean)"
    )
    page_ids = await page.evaluate(
        "() => Array.from(document.querySelectorAll('.image-item')).map(i => i.dataset.id).filter(Boolean)"
    )
    assert len(doc_ids) >= 3, f"expected 3 docs, got {len(doc_ids)}"
    log(f"sidebar shows {len(doc_ids)} documents", "pass")
    await snap(page, "images_uploaded")
    await observe(page, "3 single-image documents in sidebar")
    return page_ids


async def step_upload_pdf_with_text(page: Page) -> None:
    """Synthetic PDF with a text layer should skip OCR on every page."""
    log("Step 7b: Upload PDF with native text layer (fast path)", "step")
    pdf_path = ensure_text_layer_pdf()

    # Capture the doc count before so we can pick out the newly added one.
    before = await page.evaluate("() => Object.keys(state.documents).length")
    await page.set_input_files("#fileInput", str(pdf_path))
    await page.wait_for_function(
        f"() => Object.keys(state.documents).length > {before}", timeout=60000
    )
    new_doc = await page.evaluate(
        """() => {
            const docs = Object.values(state.documents);
            const d = docs[docs.length - 1];
            return d ? {doc_id: d.doc_id, source_kind: d.source_kind, page_count: d.page_count, has_text_layer: d.has_text_layer, filename: d.filename} : null;
        }"""
    )
    assert new_doc and new_doc["source_kind"] == "pdf", f"bad doc: {new_doc}"
    assert new_doc["page_count"] == 3, f"expected 3 pages, got {new_doc['page_count']}"
    assert new_doc["has_text_layer"] is True, "text layer should be present"
    log(f"pdf-with-text: {new_doc['page_count']} pages, has_text_layer={new_doc['has_text_layer']}", "pass")
    await snap(page, "pdf_with_text_uploaded")

    # Select it and recognize page 1 — should skip OCR (used_text_layer true).
    await page.evaluate(f"selectDocument('{new_doc['doc_id']}')")
    await observe(page, "selected synthetic PDF")
    await snap(page, "pdf_with_text_selected")
    nav_visible = await page.is_visible("#pageNavigator")
    assert nav_visible, "page navigator should be visible for 3-page PDF"
    log("page navigator visible for multipage PDF", "pass")

    await page.click("#recognizeBtn")
    await page.wait_for_function(
        """() => state.ocrResult && state.ocrResult.used_text_layer === true""",
        timeout=60_000,
    )
    log("recognize returned used_text_layer=true (OCR skipped)", "pass")
    await snap(page, "pdf_with_text_recognized")


async def step_upload_scanned_pdf(page: Page) -> None:
    """Real scanned PDF → OCR fallback path."""
    log("Step 7c: Upload scanned PDF (OCR fallback)", "step")
    if not SCANNED_PDF_PATH.exists():
        log(f"sample missing: {SCANNED_PDF_PATH} — skipping", "fail")
        return
    before = await page.evaluate("() => Object.keys(state.documents).length")
    await page.set_input_files("#fileInput", str(SCANNED_PDF_PATH))
    await page.wait_for_function(
        f"() => Object.keys(state.documents).length > {before}", timeout=60000
    )
    new_doc = await page.evaluate(
        """() => {
            const docs = Object.values(state.documents);
            const d = docs[docs.length - 1];
            return d ? {doc_id: d.doc_id, source_kind: d.source_kind, page_count: d.page_count, has_text_layer: d.has_text_layer} : null;
        }"""
    )
    assert new_doc and new_doc["source_kind"] == "pdf", f"bad doc: {new_doc}"
    assert new_doc["has_text_layer"] is False, "scanned PDF should have no text layer"
    log(f"scanned pdf: {new_doc['page_count']} pages, no text layer", "pass")

    await page.evaluate(f"selectDocument('{new_doc['doc_id']}')")
    await page.click("#recognizeBtn")
    # Wait for OCR to complete (text overlay populated).
    await page.wait_for_function(
        """() => state.ocrResult && Array.isArray(state.ocrResult.text_lines) && state.ocrResult.text_lines.length > 0 && state.ocrResult.used_text_layer === false""",
        timeout=180_000,
    )
    lines = await page.evaluate("() => state.ocrResult.text_lines.length")
    log(f"scanned pdf OCR: {lines} text lines", "pass")
    await snap(page, "scanned_pdf_recognized")


async def step_upload_docx(page: Page) -> None:
    """Real DOCX → text-layer fast path."""
    log("Step 7d: Upload DOCX (text-layer fast path)", "step")
    if not DOCX_PATH.exists():
        log(f"sample missing: {DOCX_PATH} — skipping", "fail")
        return
    before = await page.evaluate("() => Object.keys(state.documents).length")
    await page.set_input_files("#fileInput", str(DOCX_PATH))
    await page.wait_for_function(
        f"() => Object.keys(state.documents).length > {before}", timeout=60000
    )
    new_doc = await page.evaluate(
        """() => {
            const docs = Object.values(state.documents);
            const d = docs[docs.length - 1];
            return d ? {doc_id: d.doc_id, source_kind: d.source_kind, has_text_layer: d.has_text_layer} : null;
        }"""
    )
    assert new_doc and new_doc["source_kind"] == "docx", f"bad doc: {new_doc}"
    assert new_doc["has_text_layer"] is True, "DOCX should always have text layer"
    log("docx uploaded with text layer", "pass")

    await page.evaluate(f"selectDocument('{new_doc['doc_id']}')")
    await page.click("#recognizeBtn")
    await page.wait_for_function(
        """() => state.ocrResult && state.ocrResult.used_text_layer === true""",
        timeout=60_000,
    )
    log("recognize used text layer (OCR skipped)", "pass")
    await snap(page, "docx_recognized")


async def step_upload_multipage_from_screenshots(page: Page) -> None:
    """Upload the three screenshots again, this time stapled into a single
    3-page multipage document via the new Upload Multipage button."""
    log("Step 7e: Upload 3 screenshots as one multipage document", "step")

    # Replace the native prompt so we don't hang on the name dialog.
    await page.evaluate(
        "() => { window.prompt = () => 'QA Multipage Bundle'; window.confirm = () => true; }"
    )
    before = await page.evaluate("() => Object.keys(state.documents).length")
    await page.set_input_files("#fileInputMultipage", [str(p) for p in IMAGE_PATHS])
    await page.wait_for_function(
        f"() => Object.keys(state.documents).length > {before}", timeout=60000
    )
    new_doc = await page.evaluate(
        """() => {
            const docs = Object.values(state.documents);
            const d = docs[docs.length - 1];
            return d ? {doc_id: d.doc_id, source_kind: d.source_kind, page_count: d.page_count, filename: d.filename} : null;
        }"""
    )
    assert new_doc and new_doc["source_kind"] == "multipage", f"bad doc: {new_doc}"
    assert new_doc["page_count"] == 3, f"expected 3 pages, got {new_doc['page_count']}"
    log(f"multipage doc: {new_doc['filename']} with {new_doc['page_count']} pages", "pass")
    await snap(page, "multipage_uploaded")

    await page.evaluate(f"selectDocument('{new_doc['doc_id']}')")
    # Verify page navigator is visible and walks through pages 1 → 2 → 3.
    await page.wait_for_selector("#pageNavigator:visible", timeout=5000)
    indicator = await page.locator("#pageNavigator .page-indicator").text_content()
    assert "Page 1 of 3" in (indicator or ""), f"bad indicator: {indicator}"
    log(f"indicator: {indicator}", "pass")
    await snap(page, "multipage_page1")

    await page.click("#pageNavigator .page-next")
    await page.wait_for_function(
        "() => document.querySelector('#pageNavigator .page-indicator').textContent.includes('Page 2 of 3')"
    )
    log("navigated to page 2", "pass")
    await snap(page, "multipage_page2")

    await page.click("#pageNavigator .page-next")
    await page.wait_for_function(
        "() => document.querySelector('#pageNavigator .page-indicator').textContent.includes('Page 3 of 3')"
    )
    log("navigated to page 3", "pass")
    await snap(page, "multipage_page3")

    # Next should be disabled on last page.
    assert await page.locator("#pageNavigator .page-next").is_disabled()
    log("next button disabled on last page", "pass")


async def step_recognize(page: Page, image_ids: List[str]) -> None:
    log("Step 8: Run OCR + SigLIP classification on the first image", "step")
    # Click the first image to select it (if not auto-selected by upload)
    first_item = page.locator(".image-item").first
    await first_item.click()
    await page.wait_for_timeout(500)
    recognize = page.locator("#recognizeBtn")
    await expect(recognize).to_be_enabled()
    await recognize.click()
    # Wait for OCR to finish by checking for text overlay
    await page.wait_for_function(
        """() => {
            const ov = document.getElementById('textOverlay');
            return ov && ov.children.length > 0;
        }""",
        timeout=180_000,
    )
    count = await page.evaluate(
        "() => document.getElementById('textOverlay').children.length"
    )
    log(f"text overlay rendered {count} spans", "pass")
    await snap(page, "ocr_done")
    await observe(page, "OCR text overlay rendered")


async def step_process_llm(page: Page, label: str) -> str:
    log(f"Step 9 ({label}): Process with LLM — gemma4:31b", "step")
    await page.click("#llmTabBtn")
    await page.wait_for_selector("#llmMainTab.active", timeout=5000)
    process_btn = page.locator("#processLlmBtnMain")
    await expect(process_btn).to_be_enabled()
    await process_btn.click()
    # Wait up to 3 min for the result area to become visible
    await page.wait_for_selector("#llmResultMain:visible", timeout=180_000)
    text = await page.locator("#llmResultMain").text_content() or ""
    log(f"LLM result length: {len(text)} chars", "pass")
    await snap(page, f"llm_result_{label}")
    await observe(page, f"LLM result ({label}) displayed")
    return text


async def step_switch_active_domain_via_api(page: Page, domain_id: str) -> None:
    log(f"Step 10: Switch active domain to {domain_id}", "step")
    result = await page.evaluate(
        """async (id) => {
            const r = await fetch('/api/domain/active', {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({domain_id: id}),
            });
            return {status: r.status, body: await r.json()};
        }""",
        domain_id,
    )
    assert result["status"] == 200, f"switch failed: {result}"
    log(f"active domain → {result['body']['display_name']}", "pass")


async def step_recognize_and_llm_multipage(page: Page) -> None:
    """Assumes the currently-selected document is the multipage screenshot
    bundle. Recognizes each page in turn, then runs LLM on the whole
    document so the server aggregates text across pages."""
    log("Step 12: Recognize + LLM on multipage bundle", "step")
    doc = await page.evaluate(
        """() => {
            const d = state.documents[state.selectedDocId];
            return d ? {doc_id: d.doc_id, page_count: d.page_count, filename: d.filename} : null;
        }"""
    )
    if not doc or doc["page_count"] < 2:
        log("selected doc is not multipage — skipping", "fail")
        return

    # Walk through pages and recognize each one.
    for i in range(1, doc["page_count"] + 1):
        await page.evaluate(f"selectDocument('{doc['doc_id']}', {i})")
        await page.click("#recognizeBtn")
        await page.wait_for_function(
            """() => state.ocrResult && Array.isArray(state.ocrResult.text_lines) && state.ocrResult.text_lines.length > 0""",
            timeout=180_000,
        )
        log(f"recognized page {i}", "pass")
    await snap(page, "multipage_all_recognized")

    # Go back to page 1 and trigger LLM on the whole document.
    await page.evaluate(f"selectDocument('{doc['doc_id']}', 1)")
    await page.click("#llmTabBtn")
    await page.wait_for_selector("#llmMainTab.active", timeout=5000)
    process_btn = page.locator("#processLlmBtnMain")
    await expect(process_btn).to_be_enabled()
    await process_btn.click()
    await page.wait_for_selector("#llmResultMain:visible", timeout=240_000)
    await snap(page, "multipage_llm_result")
    log("multipage LLM result displayed", "pass")


async def step_generate_schema(page: Page) -> None:
    log("Step 11: Generate a template via AI with motor domain active", "step")
    # Visit Settings first so state.llmStatus is populated (the generate
    # button is gated on state.llmStatus.ollamaAvailable + selectedModel).
    await page.click("#settingsTabBtn")
    await page.wait_for_selector("#settingsTab.active", timeout=5000)
    # Give checkLlmStatus() a moment to fetch /api/llm/status.
    await page.wait_for_function(
        """() => state.llmStatus && state.llmStatus.ollamaAvailable && state.llmStatus.selectedModel""",
        timeout=15000,
    )
    model_name = await page.evaluate("() => state.llmStatus.selectedModel")
    log(f"selected model = {model_name}")

    await page.click("#schemasTabBtn")
    await page.wait_for_selector("#schemasTab.active", timeout=5000)
    await page.click("#newSchemaBtn")
    await page.wait_for_selector("#schemaEditForm:visible", timeout=5000)
    await page.fill(
        "#schemaDescription",
        "A claim form that a driver fills out after a minor accident. Includes vehicle details, driver details, accident location, and damage description.",
    )
    await observe(page, "schema description entered")
    generate_btn = page.locator("#generateSchemaBtn")
    if await generate_btn.is_disabled():
        log("generate button disabled — gemma4 not available in UI state", "fail")
        await snap(page, "schema_generate_disabled")
        return
    await generate_btn.click()
    # Wait for the completion marker the frontend sets on the stream's `done`
    # event ("(generated)" gets appended to the title). Using value-length as
    # the gate races the still-streaming response.
    await page.wait_for_function(
        """() => {
            const title = document.getElementById('schemaEditorTitle');
            return title && title.textContent.includes('(generated)');
        }""",
        timeout=240_000,
    )
    # Small settle so the final data.yaml replacement lands before we read.
    await page.wait_for_timeout(300)
    yaml_val = await page.locator("#schemaYamlEditor").input_value()
    log(f"generated YAML length: {len(yaml_val)}")
    lower = yaml_val.lower()
    domain_hits = sum(
        1 for kw in ("vehicle", "vin", "license", "plate", "driver", "accident")
        if kw in lower
    )
    log(f"motor-domain keyword hits in YAML: {domain_hits}/6", "pass" if domain_hits >= 3 else "fail")
    await snap(page, "schema_generated_motor")
    await observe(page, "generated schema visible — domain context confirmed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def main() -> int:
    print("\n=== claims_doc_recognizer functional test suite ===")
    print(f"target: {REMOTE_URL}")
    print(f"screenshot dir: {SCREENSHOT_DIR}")
    if SCREENSHOT_DIR.exists():
        shutil.rmtree(SCREENSHOT_DIR)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(headless=False, slow_mo=120)
        context: BrowserContext = await browser.new_context(
            viewport={"width": 1440, "height": 900},
        )
        page: Page = await context.new_page()

        async def safe(coro, name: str):
            try:
                return await coro
            except Exception as e:
                failures.append(f"{name}: {e}")
                print(f"  ✗ FAIL — {name}: {e}")
                try:
                    await snap(page, f"fail_{name}")
                except Exception:
                    pass
                return None

        try:
            await safe(step_load_home(page), "load_home")
            await safe(step_verify_tabs(page), "verify_tabs")
            await safe(step_open_domain_tab(page), "open_domain_tab")
            await safe(step_switch_to_motor_and_activate(page), "motor_active")
            await safe(step_switch_back_to_health(page), "health_active")
            await safe(step_domain_crud_roundtrip(page), "domain_crud")

            # --- Baseline image flow: upload 3 screenshots, OCR, LLM ---
            ids = await safe(step_upload_images(page), "upload")
            if ids:
                await safe(step_recognize(page, ids), "recognize")
                health_text = await safe(step_process_llm(page, "health"), "llm_health")
                await safe(
                    step_switch_active_domain_via_api(page, "motor_insurance"),
                    "switch_to_motor",
                )
                motor_text = await safe(step_process_llm(page, "motor"), "llm_motor")
                await safe(
                    step_switch_active_domain_via_api(page, "health_insurance"),
                    "switch_back_health",
                )
                if health_text and motor_text:
                    log(
                        f"health chars={len(health_text)} motor chars={len(motor_text)}",
                        "info",
                    )

            # --- Schema generation with motor domain active ---
            await safe(
                step_switch_active_domain_via_api(page, "motor_insurance"),
                "switch_to_motor_for_schema",
            )
            await safe(step_generate_schema(page), "schema_generate")
            await safe(
                step_switch_active_domain_via_api(page, "health_insurance"),
                "restore_health",
            )

            # --- New upload paths: PDF (text layer + scanned), DOCX, multipage ---
            await safe(step_upload_pdf_with_text(page), "upload_pdf_text_layer")
            await safe(step_upload_scanned_pdf(page), "upload_scanned_pdf")
            await safe(step_upload_docx(page), "upload_docx")
            await safe(step_upload_multipage_from_screenshots(page), "upload_multipage")

            # --- Final: run LLM on the multipage doc to verify aggregation ---
            await safe(step_recognize_and_llm_multipage(page), "multipage_llm")
        finally:
            print("\n=== test run complete ===")
            if failures:
                print(f"✗ {len(failures)} failures:")
                for f in failures:
                    print(f"  - {f}")
            else:
                print("✓ all steps passed")
            await page.wait_for_timeout(2000)
            await browser.close()

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
