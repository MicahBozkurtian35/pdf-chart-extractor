#!/usr/bin/env python3
"""
PDF Chart Data Extractor Core Module
Upstage-only detector & crops, OpenRouter VLM extraction, no detection fallback.
"""

import os
import re
import io
import json
import math
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import cv2  # still used for helpers like resizing/Hough but NOT for region detection
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Small safety helpers
# ----------------------
def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, inter_x1 - inter_x0), max(0, inter_y1 - inter_y0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _safe_bbox(bbox, W, H, pad_px: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """
    Clamp bbox to image bounds and handle normalized inputs.
    bbox: (x0, y0, x1, y1) either absolute pixels or normalized [0..1].
    Returns ints or None if invalid.
    """
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = map(float, bbox)
    # If normalized, scale to px
    if 0.0 <= min(x0, y0, x1, y1) and max(x0, y0, x1, y1) <= 1.0:
        x0 *= W; x1 *= W; y0 *= H; y1 *= H
    x0 -= pad_px; y0 -= pad_px; x1 += pad_px; y1 += pad_px

    x0 = int(max(0, math.floor(x0)))
    y0 = int(max(0, math.floor(y0)))
    x1 = int(min(W, math.ceil(x1)))
    y1 = int(min(H, math.ceil(y1)))

    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)

def _safe_pil_from_numpy_rgb(img: np.ndarray) -> Image.Image:
    """
    Accepts numpy RGB or grayscale, returns PIL Image (RGB or L).
    Ensures dtype, contiguity, and channel count are PIL-safe.
    """
    if img is None or img.size == 0:
        raise ValueError("Empty image for PIL conversion")
    arr = img
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # If 3+ channels, keep first 3 as RGB
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3].copy(order="C")  # assume already RGB from pdf2image
        return Image.fromarray(arr)
    if arr.ndim == 2:
        return Image.fromarray(arr)
    raise ValueError(f"Unsupported image shape for PIL: {arr.shape}")

def _pil_save_safe(pil_img: Image.Image, path: Path):
    """Write via buffer first to avoid partial state errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    with open(path, "wb") as f:
        f.write(buf.getvalue())

# ----------------------
# Extractor
# ----------------------

class PDFChartExtractor:
    """Main extractor class implementing the full pipeline (Upstage-only detection)."""
    def _call_upstage_document_digitization(self, pdf_path: str) -> dict:
        """
        POST to Upstage Document Digitization with the minimal, known-good shape.
        - endpoint: self.upstage_url (defaults to base)
        - file field: 'document'
        - form: {'model': 'document-parse'}
        """
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise RuntimeError("UPSTAGE_API_KEY not set")

        url = getattr(self, "upstage_url", None) or "https://api.upstage.ai/v1/document-digitization"
        headers = {"Authorization": f"Bearer {api_key}"}

        # Minimal payload that we verified works (curl + py_min_post)
        data = {"model": "document-parse"}

        # Trace exactly what we’re about to send
        try:
            fsz = os.path.getsize(pdf_path)
        except Exception:
            fsz = -1
        logging.info(
            "TRACE Upstage POST url=%s data_keys=%s file_field=%s file_size=%s",
            url, list(data.keys()), "document", fsz
        )

        with open(pdf_path, "rb") as f:
            files = {"document": f}
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)

        try:
            resp.raise_for_status()
        except requests.HTTPError:
            logging.error("Upstage HTTP error %s body=%s", resp.status_code, resp.text[:1000])
            raise

        j = resp.json()
        # (optional) quick schema peek
        content = j.get("content", {})
        pages = j.get("pages", [])
        logging.info(
            "TRACE Upstage JSON keys=%s content_keys=%s pages_len=%s",
            list(j.keys()),
            (list(content.keys()) if isinstance(content, dict) else type(content).__name__),
            (len(pages) if isinstance(pages, list) else "n/a"),
        )
        return j



    def _log_upstage_payload(self, data, files, url):
        try:
            dk = [k for (k, _) in data]
            fk = list(files.keys())
            logger.info("Upstage payload", extra={
                "event": "upstage_payload",
                "data_keys": dk,
                "file_keys": fk,
                "url": url
            })
        except Exception:
            pass


    def __init__(self):
        # API Keys
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.upstage_key = os.getenv("UPSTAGE_API_KEY", "")

        # Models
        self.llm_model = os.getenv("LLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")
        self.openrouter_site = os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173")
        self.openrouter_app = os.getenv("OPENROUTER_APP_NAME", "pdf-extractor")

        # Upstage
        # Keep your previous endpoint but fix payload to be API-friendly
        self.upstage_url = os.getenv("UPSTAGE_URL", "https://api.upstage.ai/v1/document-digitization")
        # Optional "model" if your account expects one; otherwise leave blank
        self.upstage_model = os.getenv("UPSTAGE_MODEL", "document-parse")  # e.g., "document-parse"
        self.upstage_force_ocr = os.getenv("UPSTAGE_FORCE_OCR", "auto")
        self.upstage_use_base64 = os.getenv("UPSTAGE_USE_BASE64_CROPS", "false").lower() == "true"
        self.upstage_timeout = int(os.getenv("UPSTAGE_TIMEOUT", "240"))
        # STRICT: no fallback; raise on any error
        self.upstage_strict = True

        # PDF render
        self.poppler_path = os.getenv("POPPLER_PATH") or None
        self.pdf_dpi = int(os.getenv("PDF_DPI", "350"))
        self.upscale_factor = float(os.getenv("UPSCALE_FACTOR", "1.0"))

        # Perf / debug
        self.fast_dev = os.getenv("FAST_DEV", "false").lower() == "true"
        self.fast_dev_scale = float(os.getenv("FAST_DEV_IMAGE_SCALE", "0.8"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "3"))
        self.debug_timing = os.getenv("DEBUG_TIMING", "true").lower() == "true"

        # LLM timeouts
        self.llm_timeout = int(os.getenv("LLM_TIMEOUT", "120"))

        # Dirs
        self.dirs = {
            "temp": Path("temp_images"),
            "enhanced": Path("enhanced_images"),
            "thumbnails": Path("thumbnails"),
            "upstage_cache": Path("upstage_cache"),
            "llm_cache": Path("llm_cache"),
            "debug": Path("debug"),
        }
        for d in self.dirs.values():
            d.mkdir(exist_ok=True)

        # Session
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # ------------- Public helpers (server uses this) -------------

    def _filter_regions(self, regions: List[Dict]) -> List[Dict]:
        """
        Keep likely chart regions; prefer explicit chart markers and 'chart' category,
        enforce minimum size/aspect ratio, and dedupe by IoU keeping the best-scoring box.
        """
        MIN_W, MIN_H = 320, 220
        MIN_AR = 1.1     # width / height
        IOU_DROP = 0.60  # overlap threshold for dedupe

        def w(r): x0, y0, x1, y1 = r["bbox"]; return max(0, x1 - x0)
        def h(r): x0, y0, x1, y1 = r["bbox"]; return max(0, y1 - y0)

        candidates = []
        for r in regions:
            x0, y0, x1, y1 = r["bbox"]
            ww, hh = w(r), h(r)
            ar = (ww / hh) if hh > 0 else 0
            cat = (r.get("category") or "").lower()
            has_marker = bool(r.get("has_chart_marker", False))

            # Hard screen: require reasonable size AND (explicit chart marker OR category==chart/figure)
            if ww < MIN_W or hh < MIN_H:
                r["_drop_reason"] = f"too_small({ww}x{hh})"
                continue
            if not (has_marker or cat in ("chart", "figure")):
                r["_drop_reason"] = f"bad_cat({cat})"
                continue
            if ar < MIN_AR:  # many charts are wider than tall; avoid tall text columns
                r["_drop_reason"] = f"bad_ar({ar:.2f})"
                continue

            # Score: explicit chart marker > category chart > figure; then prefer larger (more likely full chart)
            score = 0.0
            if has_marker: score += 2.0
            if cat == "chart": score += 1.0
            score += min(1.0, (ww * hh) / (1400 * 900))  # mild area bonus, capped
            r["_score"] = score
            candidates.append(r)

        # Sort by score desc, then area desc (keep best first)
        candidates.sort(key=lambda r: (r.get("_score", 0.0), (w(r) * h(r))), reverse=True)

        kept: List[Dict] = []
        for r in candidates:
            rb = r["bbox"]
            if any(_iou(rb, k["bbox"]) >= IOU_DROP for k in kept):
                r["_drop_reason"] = "dedup_iou"
                continue
            kept.append(r)

        # Logging: what happened
        dropped = [r for r in regions if r not in kept]
        logging.info("TRACE Filtered regions: kept=%d of %d", len(kept), len(regions))
        for idx, r in enumerate(kept[:12]):
            x0, y0, x1, y1 = r["bbox"]; ww, hh = x1 - x0, y1 - y0
            logging.info("KEPT %2d: cat=%s marker=%s score=%.2f size=%dx%d bbox=%s",
                        idx, r.get("category"), bool(r.get("has_chart_marker")), r.get("_score", 0.0), ww, hh, r["bbox"])
        if dropped:
            logging.info("DROPPED %d regions (first 12 reasons):", len(dropped))
            for r in dropped[:12]:
                x0, y0, x1, y1 = r["bbox"]; ww, hh = x1 - x0, y1 - y0
                logging.info("drop: cat=%s marker=%s size=%dx%d reason=%s bbox=%s",
                            r.get("category"), bool(r.get("has_chart_marker")), ww, hh, r.get("_drop_reason",""), r["bbox"])
        return kept



    def generate_thumbnails_for_pdf(self, pdf_path: str, stamped_name: str) -> int:
        """Render all pages to thumbs with name like: <pdf>_page_001_thumb.png"""
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        for p in range(1, page_count + 1):
            page_img = self._render_page(pdf_path, p, stamped_name)
            # render_page already writes a thumb for the page
        return page_count

    def process_page(self, pdf_path: str, page_number: int) -> Dict[str, Any]:
        """
        Process a single page from a PDF and return:
        { tables: [...], debug_raw: [...] }
        """
        t0 = time.time()

        with fitz.open(pdf_path) as doc:
            if not (1 <= page_number <= len(doc)):
                raise ValueError(f"Page {page_number} out of range 1..{len(doc)}")

        base = Path(pdf_path).name
        page_image = self._render_page(pdf_path, page_number, base)

        regions = self._detect_regions_upstage(pdf_path, page_number, page_image)
        if self.debug_timing:
            logger.info("Detected %d regions on page %d", len(regions), page_number)

        # --- NEW: log what we’re about to feed the LLM
        logger.info("FEEDING LLM: %d regions -> categories & sizes", len(regions))
        for i, r in enumerate(regions[:50]):  # cap for noise
            x0, y0, x1, y1 = r["bbox"]
            logger.info("region %d: cat=%s size=%dx%d", i, r.get("category"), x1 - x0, y1 - y0)

        tables: List[Dict] = []
        debug_raw: List[Dict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {
                pool.submit(self._process_region, page_image, region, page_number, idx): (region, idx)
                for idx, region in enumerate(regions)
            }
            for fut in as_completed(futs):
                region, idx = futs[fut]
                try:
                    result = fut.result(timeout=self.llm_timeout + 30)
                    if result:
                        tables.append(result["table"])
                        debug_raw.append(result["debug"])
                except Exception as e:
                    logger.exception("Region %s failed", idx)
                    tables.append({
                        "page": page_number,
                        "region": idx,
                        "error": str(e),
                        "note": f"Region failed: {e}",
                    })

        if self.debug_timing:
            logger.info("Page %d processed in %.2fs", page_number, time.time() - t0)

        return {"tables": tables, "debug_raw": debug_raw}


    # ------------- Rendering & detection -------------

    def _render_page(self, pdf_path: str, page_number: int, stamped_name: str) -> np.ndarray:
        """Render page and write a thumb named <pdf>_page_001_thumb.png to match frontend."""
        pages = convert_from_path(
            pdf_path, first_page=page_number, last_page=page_number,
            dpi=self.pdf_dpi, poppler_path=self.poppler_path
        )
        if not pages:
            raise RuntimeError(f"Failed to render page {page_number}")

        page_pil = pages[0]  # RGB
        page_np = np.array(page_pil)  # RGB ndarray

        if self.upscale_factor > 1.0:
            h, w = page_np.shape[:2]
            page_np = cv2.resize(page_np, (int(w * self.upscale_factor), int(h * self.upscale_factor)), interpolation=cv2.INTER_CUBIC)

        # Write thumb with the exact naming pattern your UI requests
        thumb_name = f"{stamped_name}_page_{page_number:03d}_thumb.png"
        thumb_path = self.dirs["thumbnails"] / thumb_name
        # Keep aspect ratio, width ~ 200
        w = 200
        scale = w / page_np.shape[1]
        thumb = cv2.resize(page_np, (w, max(1, int(page_np.shape[0] * scale))))
        # pdf2image -> RGB; OpenCV expects BGR to write
        cv2.imwrite(str(thumb_path), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        return page_np

    def _detect_regions_upstage(self, pdf_path: str, page_number: int, page_image: np.ndarray) -> List[Dict]:
        width, height = int(page_image.shape[1]), int(page_image.shape[0])

        # ---- cache or call ----
        cache_file = self.dirs["upstage_cache"] / (Path(pdf_path).name + ".upstage.json")
        up_json = None
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    up_json = json.load(f)
            except Exception:
                up_json = None
        if up_json is None:
            up_json = self._call_upstage_document_digitization(pdf_path)
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(up_json, f)
            except Exception:
                pass
        if not up_json:
            raise RuntimeError("Empty response from Upstage")

        def _to_px_box(coords_or_box):
            if isinstance(coords_or_box, (list, tuple)) and coords_or_box and isinstance(coords_or_box[0], dict):
                xs = [float(pt.get("x", 0)) for pt in coords_or_box]
                ys = [float(pt.get("y", 0)) for pt in coords_or_box]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                if 0.0 <= min(x0, x1, y0, y1) and max(x0, x1, y0, y1) <= 1.0:
                    return [int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)]
                return [int(x0), int(y0), int(x1), int(y1)]
            if isinstance(coords_or_box, (list, tuple)) and len(coords_or_box) == 4:
                x0, y0, x1, y1 = map(float, coords_or_box)
                if 0.0 <= min(x0, x1, y0, y1) and max(x0, x1, y0, y1) <= 1.0:
                    return [int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)]
                return [int(x0), int(y0), int(x1), int(y1)]
            return None

        regions: List[Dict] = []

        # ---- primary: elements ----
        els = up_json.get("elements") or up_json.get("data") or []
        if not isinstance(els, list):
            pages = up_json.get("pages", [])
            for pg in pages:
                try:
                    if int(pg.get("page", -1)) == int(page_number):
                        els = pg.get("elements", [])
                        break
                except Exception:
                    continue

        for el in els:
            try:
                p = int(el.get("page", 1))
            except Exception:
                p = 1
            if p != page_number:
                continue

            cat = (el.get("category") or el.get("type") or "").lower() or "figure"
            bbox = None
            if el.get("coordinates"):
                bbox = _to_px_box(el["coordinates"])
            if not bbox and (el.get("bounding_box") or el.get("bbox")):
                bbox = _to_px_box(el.get("bounding_box") or el.get("bbox"))
            if not bbox or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            # detect marker "Chart Type:" inside HTML content
            has_marker = False
            try:
                html_snip = ((el.get("content") or {}).get("html") or "")[:400]
                if "Chart Type:" in html_snip:
                    has_marker = True
            except Exception:
                pass

            crop_bytes = None
            if self.upstage_use_base64:
                img_obj = el.get("image") or el.get("preview") or (el.get("content") or {}).get("image")
                if isinstance(img_obj, dict):
                    b64 = img_obj.get("base64")
                    url = img_obj.get("url")
                    if isinstance(b64, str):
                        try:
                            import base64 as _b64
                            crop_bytes = _b64.b64decode(b64)
                        except Exception:
                            crop_bytes = None
                    elif isinstance(url, str) and url.startswith(("http://", "https://")):
                        try:
                            rimg = self.session.get(url, timeout=20)
                            if rimg.ok:
                                crop_bytes = rimg.content
                        except Exception:
                            pass

            regions.append({
                "bbox": bbox,
                "category": cat,
                "confidence": float(el.get("confidence", 0.5)) if isinstance(el.get("confidence", 0.5), (int, float)) else 0.5,
                "crop_bytes": crop_bytes,
                "text_tokens": el.get("text", []),
                "has_chart_marker": has_marker,
            })

        # ---- fallback: parse HTML <img data-coord="..."> with float coords ----
        html = (((up_json or {}).get("content") or {}).get("html")) or ""
        if html:
            import re
            for m in re.finditer(
                r'data-coord\s*=\s*[\'"]\s*top-left:\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*;\s*bottom-right:\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*[\'"]',
                html, flags=re.I
            ):
                x0, y0, x1, y1 = map(float, m.groups())
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                if x1 <= x0 + 8 or y1 <= y0 + 8:
                    continue
                dup = any(_iou([x0, y0, x1, y1], r["bbox"]) >= 0.9 for r in regions)
                if not dup:
                    regions.append({
                        "bbox": [x0, y0, x1, y1],
                        "category": "figure",
                        "confidence": 0.5,
                        "crop_bytes": None,
                        "text_tokens": [],
                        "has_chart_marker": True,  # these <figure> blocks include Chart Type in HTML
                    })

        # ---- sort, log raw count, then filter ----
        regions.sort(key=lambda r: (r["bbox"][1] // 100, r["bbox"][0]))
        logging.info("Upstage detected %d regions on page %d", len(regions), page_number)

        regions = self._filter_regions(regions)
        try:
            preview = [r["bbox"] for r in regions[:10]]
        except Exception:
            preview = []
        logging.info("TRACE Regions page=%s kept=%s preview_bboxes=%s", page_number, len(regions), preview)
        return regions




    # ------------- Region processing -------------

    def _process_region(self, page_image: np.ndarray, region: Dict, page_number: int, region_idx: int) -> Optional[Dict]:
        t0 = time.time()
        try:
            crops = self._generate_crops(page_image, region, page_number, region_idx)
            if not crops or "main" not in crops:
                raise RuntimeError("No usable crop produced")

            extraction_result = self._extract_with_llm(crops, page_number, region_idx)

            if extraction_result and extraction_result.get("data"):
                table_entry = {
                    "page": page_number,
                    "region": region_idx,
                    "image": crops["main"]["filename"],
                    "data": extraction_result.get("data", []),
                    "series_meta": extraction_result.get("series_meta", []),
                    "chart_type": extraction_result.get("chart_type", "unknown"),
                    "confidence": extraction_result.get("confidence", "low"),
                    "note": extraction_result.get("note", ""),
                    "category": region.get("category", "figure"),
                    "series_hints": extraction_result.get("series_hints", []),
                    "extras_used": list(crops.keys()),
                }
                debug_entry = {
                    "page": page_number,
                    "region": region_idx,
                    "image": crops["main"]["filename"],
                    "extras": [crops[k]["filename"] for k in crops if k != "main"],
                    "raw": extraction_result.get("raw_response", "")[:500],
                    "raw_fix": extraction_result.get("retry_response", "")[:500],
                }
                if self.debug_timing:
                    logger.info("Region %d processed in %.2fs", region_idx, time.time() - t0)
                return {"table": table_entry, "debug": debug_entry}

            # No LLM data—return a stub with the crop so the UI still shows something
            return {
                "table": {
                    "page": page_number,
                    "region": region_idx,
                    "image": crops["main"]["filename"],
                    "data": [],
                    "note": "LLM extraction returned no data",
                },
                "debug": {"page": page_number, "region": region_idx, "raw": ""},
            }
        except Exception as e:
            logger.exception("Error processing region %s", region_idx)
            return {
                "table": {
                    "page": page_number,
                    "region": region_idx,
                    "error": str(e),
                    "note": f"Processing failed: {e}",
                },
                "debug": {"page": page_number, "region": region_idx, "error": str(e)},
            }

    def _generate_crops(self, page_image: np.ndarray, region: Dict, page_number: int, region_idx: int) -> Dict:
        """
        If Upstage gave us a crop image, use it AS-IS.
        Otherwise crop by bbox with strict clamping and safe PIL save.
        """
        timestamp = int(time.time() * 1000)
        crops: Dict[str, Dict] = {}
        H, W = page_image.shape[:2]

        # 1) Try Upstage-provided crop bytes
        cb = region.get("crop_bytes")
        if cb:
            try:
                pil = Image.open(io.BytesIO(cb)).convert("RGB")
                main_img = np.array(pil)
                logger.info("Region %s: using Upstage crop image", region_idx)
            except Exception as e:
                logger.warning("Region %s: failed to decode Upstage crop (%s), falling back to bbox", region_idx, e)
                main_img = None
        else:
            main_img = None

        # 2) Crop by bbox with clamping
        if main_img is None:
            raw_bbox = region.get("bbox")
            bbox = _safe_bbox(raw_bbox, W, H, pad_px=0)
            if bbox is None:
                raise RuntimeError(f"Invalid bbox after clamping: {raw_bbox}")
            x0, y0, x1, y1 = bbox
            main_img = page_image[y0:y1, x0:x1].copy(order="C")  # RGB slice
            logger.info("Region %s: bbox=%s -> crop size=%dx%d", region_idx, bbox, main_img.shape[1], main_img.shape[0])

        if self.fast_dev and self.fast_dev_scale < 1.0:
            new_w = max(1, int(main_img.shape[1] * self.fast_dev_scale))
            new_h = max(1, int(main_img.shape[0] * self.fast_dev_scale))
            main_img = cv2.resize(main_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save main
        main_filename = f"page_{page_number}_region_{region_idx}_{timestamp}_crop.png"
        main_path = self.dirs["enhanced"] / main_filename
        _pil_save_safe(_safe_pil_from_numpy_rgb(main_img), main_path)
        crops["main"] = {"path": main_path, "filename": main_filename, "image": main_img}

        # Axis strips (only if wide enough)
        if main_img.shape[1] > 240:
            left_strip = main_img[:, :120]
            lf = f"page_{page_number}_region_{region_idx}_{timestamp}_axisL.png"
            lp = self.dirs["enhanced"] / lf
            _pil_save_safe(_safe_pil_from_numpy_rgb(left_strip), lp)
            crops["axis_left"] = {"path": lp, "filename": lf, "image": left_strip}

            right_strip = main_img[:, -120:]
            rf = f"page_{page_number}_region_{region_idx}_{timestamp}_axisR.png"
            rp = self.dirs["enhanced"] / rf
            _pil_save_safe(_safe_pil_from_numpy_rgb(right_strip), rp)
            crops["axis_right"] = {"path": rp, "filename": rf, "image": right_strip}

        # Optional plot focus (guarded)
        plot_box = self._find_plot_box_safe(main_img)
        if plot_box:
            px0, py0, px1, py1 = plot_box
            plot_img = main_img[py0:py1, px0:px1]
            pf = f"page_{page_number}_region_{region_idx}_{timestamp}_plotfocus.png"
            pp = self.dirs["enhanced"] / pf
            _pil_save_safe(_safe_pil_from_numpy_rgb(plot_img), pp)
            crops["plot_focus"] = {"path": pp, "filename": pf, "image": plot_img}

        # Color emphasis
        color_enhanced = self._enhance_colors(main_img)
        cf = f"page_{page_number}_region_{region_idx}_{timestamp}_color.png"
        cp = self.dirs["enhanced"] / cf
        _pil_save_safe(_safe_pil_from_numpy_rgb(color_enhanced), cp)
        crops["color_emphasis"] = {"path": cp, "filename": cf, "image": color_enhanced}

        return crops

    # ------------- Visual helpers -------------

    def _find_plot_box_safe(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find inner plot-ish rectangle; clamp to avoid out-of-bounds. Returns None if unsure."""
        try:
            H, W = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
            if lines is None:
                return None

            h_lines, v_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if angle < 10 or angle > 170:  # horizontal
                    h_lines.append(int((y1 + y2) / 2))
                elif 80 < angle < 100:  # vertical
                    v_lines.append(int((x1 + x2) / 2))

            if not h_lines or not v_lines:
                return None
            h_lines.sort()
            v_lines.sort()
            left = v_lines[0] if v_lines[0] > W * 0.1 else int(W * 0.15)
            right = v_lines[-1] if v_lines[-1] < W * 0.9 else int(W * 0.85)
            top = h_lines[0] if h_lines[0] > H * 0.1 else int(H * 0.15)
            bottom = h_lines[-1] if h_lines[-1] < H * 0.9 else int(H * 0.85)

            # clamp and validate
            left = max(0, min(left, W - 2))
            right = max(left + 1, min(right, W))
            top = max(0, min(top, H - 2))
            bottom = max(top + 1, min(bottom, H))
            if right - left < 4 or bottom - top < 4:
                return None
            return (left, top, right, bottom)
        except Exception:
            return None

    def _enhance_colors(self, image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image)
        pil = ImageEnhance.Color(pil).enhance(1.5)
        pil = ImageEnhance.Contrast(pil).enhance(1.2)
        return np.array(pil)

    # ------------- LLM extraction -------------

    def _extract_with_llm(self, crops: Dict, page_number: int, region_idx: int) -> Optional[Dict]:
        main_hash = self._hash_image(crops["main"]["image"])
        model_safe = (self.llm_model or "model").replace("/", "_").replace("\\", "_").replace(":", "_")
        cache_key = f"{main_hash}_{model_safe}.json"
        cache_file = self.dirs["llm_cache"] / cache_key
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        prompt = self._build_extraction_prompt()

        images_b64 = []
        for key in ("main", "axis_left", "plot_focus", "color_emphasis"):
            if key in crops:
                images_b64.append(self._image_to_base64(crops[key]["image"]))

        try:
            response = self._call_openrouter(prompt, images_b64)
            logger.info("LLM RESP region=%s bytes=%s", region_idx, (len(response) if response else 0))

            if response:
                extracted = self._parse_llm_response(response)
                if not self._validate_extraction(extracted):
                    retry = self._build_retry_prompt(extracted)
                    retry_resp = self._call_openrouter(retry, images_b64)
                    logger.info("LLM RETRY RESP region=%s bytes=%s", region_idx, (len(retry_resp) if retry_resp else 0))
                    if retry_resp:
                        retry_extracted = self._parse_llm_response(retry_resp)
                        if self._validate_extraction(retry_extracted):
                            retry_extracted["retry_response"] = retry_resp[:500]
                            extracted = retry_extracted
                if extracted:
                    extracted["raw_response"] = (response or "")[:500]
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(extracted, f)
                    return extracted
        except Exception as e:
            logger.error("LLM extraction failed: %s", e)
        return None


    def _build_extraction_prompt(self) -> str:
        return (
            "You are analyzing a chart image to extract numeric data values.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Extract the ACTUAL DATA POINTS only - the lines, bars, or plotted values\n"
            "2. EXCLUDE all overlays, reference lines, targets, averages, bands, or annotations\n"
            "3. Return precise NUMERIC Y-values by reading from the Y-axis scale\n"
            "4. Interpolate Y-values based on pixel position relative to Y-axis ticks\n\n"
            "Return a JSON object with this EXACT structure:\n"
            "{\n"
            '  "chart_type": "line|bar|stacked_bar|dual_axis|combo|other",\n'
            '  "x_values": ["label1", "label2", ...],\n'
            '  "series": [\n'
            '    {"name": "Series Name", "y_values": [123.4, 567.8, ...], "axis": "left|right|none", "render": "line|bar"}\n'
            "  ],\n"
            '  "y_axis_range": {"min": 0, "max": 1000},\n'
            '  "confidence": "high|medium|low",\n'
            '  "notes": "Any relevant observations"\n'
            "}\n\n"
            "RULES:\n"
            "- y_values MUST be numeric (not strings)\n"
            "- y_values MUST match x_values length exactly\n"
            "- Exclude overlays/targets/averages\n"
            "- Look at all provided images\n"
        )

    def _build_retry_prompt(self, previous_result: Dict) -> str:
        issues: List[str] = []
        if not previous_result.get("series"):
            issues.append("No data series found")
        return "Retry focusing on accurate numeric y_values and proper series separation. Return the same JSON structure."

    def _call_openrouter(self, prompt: str, images_b64: List[str]) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "HTTP-Referer": self.openrouter_site,
            "X-Title": self.openrouter_app,
            "Content-Type": "application/json",
        }
        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        data = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        r = self.session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=self.llm_timeout)
        logger.info("LLM HTTP status=%s", r.status_code)
        if not r.ok:
            if r.status_code == 429:
                time.sleep(5)
                return None
            logger.error("LLM HTTP error %s body_head=%s", r.status_code, r.text[:600])
            r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]


    def _parse_llm_response(self, response: str) -> Dict:
        try:
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                return {}
            data = json.loads(m.group())
            out: Dict[str, Any] = {
                "chart_type": data.get("chart_type", "unknown"),
                "confidence": data.get("confidence", "low"),
                "note": data.get("notes", ""),
            }
            x_vals = data.get("x_values", [])
            series = data.get("series", [])
            if x_vals and series:
                rows: List[Dict[str, Any]] = []
                meta: List[Dict[str, Any]] = []
                for i, xv in enumerate(x_vals):
                    row = {"X": xv}
                    for s in series:
                        name = s.get("name", "Series")
                        yv = s.get("y_values", [])
                        if i < len(yv):
                            row[name] = yv[i]
                    rows.append(row)
                for s in series:
                    meta.append({
                        "name": s.get("name", "Series"),
                        "axis": s.get("axis", "left"),
                        "render": s.get("render", "line"),
                    })
                out["data"] = rows
                out["series_meta"] = meta
            return out
        except Exception:
            return {}

    def _validate_extraction(self, extracted: Dict) -> bool:
        if not extracted or not extracted.get("data"):
            return False
        # at least one numeric column
        for row in extracted["data"]:
            for k, v in row.items():
                if k != "X" and isinstance(v, (int, float)):
                    return True
        return False

    # ------------- misc utils -------------

    def _hash_image(self, image: np.ndarray) -> str:
        img_bytes = cv2.imencode(".png", image)[1].tobytes()
        return hashlib.md5(img_bytes).hexdigest()

    def _image_to_base64(self, image: np.ndarray) -> str:
        import base64
        buf = io.BytesIO()
        _safe_pil_from_numpy_rgb(image).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
