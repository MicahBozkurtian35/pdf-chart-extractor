#!/usr/bin/env python3
"""
PDF Chart Data Extractor — Upstage-only detection + strict chart-only cropping
- Upstage Document Digitization for region detection
- Strong region refinement (quality-scored + NMS) so only charts/graphs get through
- OpenRouter VLM extraction
"""

import os
import io
import re
import json
import math
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd  # kept for potential future table post-processing
import requests
import cv2
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("pdf-chart-extractor")

# -----------------------------------------------------------------------------
# Small helpers (geometry, PIL safety)
# -----------------------------------------------------------------------------

def _iou_xyxy(a, b) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0); ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    denom = (max(0, ax1-ax0)*max(0, ay1-ay0) + max(0, bx1-bx0)*max(0, by1-by0) - inter) + 1e-6
    return inter / denom

def _safe_bbox(bbox, W, H, pad_px: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """Clamp bbox to image bounds; accepts pixel or normalized [0..1] coords."""
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = map(float, bbox)
    if 0.0 <= min(x0, y0, x1, y1) and max(x0, y0, x1, y1) <= 1.0:  # normalized
        x0 *= W; x1 *= W; y0 *= H; y1 *= H
    x0 -= pad_px; y0 -= pad_px; x1 += pad_px; y1 += pad_px
    x0 = int(max(0, math.floor(x0))); y0 = int(max(0, math.floor(y0)))
    x1 = int(min(W, math.ceil(x1)));  y1 = int(min(H, math.ceil(y1)))
    if x1 - x0 < 2 or y1 - y0 < 2: return None
    return (x0, y0, x1, y1)

def _safe_pil_from_numpy_rgb(img: np.ndarray) -> Image.Image:
    if img is None or img.size == 0:
        raise ValueError("Empty image for PIL conversion")
    arr = img
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return Image.fromarray(arr[:, :, :3].copy(order="C"))  # RGB
    if arr.ndim == 2:
        return Image.fromarray(arr)
    raise ValueError(f"Unsupported image shape for PIL: {arr.shape}")

def _pil_save_safe(pil_img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    with open(path, "wb") as f:
        f.write(buf.getvalue())

# -----------------------------------------------------------------------------
# Quality scoring + NMS + debug overlay
# -----------------------------------------------------------------------------
def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float((edges > 0).mean())

def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _white_ratio(gray: np.ndarray) -> float:
    _, th = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    return float((th == 255).mean())

def _qscore(img_rgb: np.ndarray, bbox: List[int]) -> float:
    x0, y0, x1, y1 = map(int, bbox)
    crop = img_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return -1.0
    h, w = crop.shape[:2]
    if max(h, w) > 900:
        s = 900.0 / max(h, w)
        crop = cv2.resize(crop, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    ed = _edge_density(gray)
    lv = _lap_var(gray)
    ws = _white_ratio(gray)
    # Higher is better; strong penalty for big blank/white crops
    return (ed * 1.8) + (min(lv, 250.0)/250.0 * 1.2) - (max(0.0, ws - 0.85) * 2.0)

def _nms(dets: List[Dict], iou_thr: float = 0.35, key: str = "qscore") -> List[Dict]:
    dets = sorted(dets, key=lambda d: d.get(key, 0.0), reverse=True)
    kept: List[Dict] = []
    for d in dets:
        if all(_iou_xyxy(d["bbox"], k["bbox"]) < iou_thr for k in kept):
            kept.append(d)
    return kept

def _overlay_debug(page_img_rgb: np.ndarray, kept: List[Dict], dropped_lowq: List[Dict], dropped_hard: List[Dict], out_path: Path):
    img = page_img_rgb.copy()
    def draw(b, color, txt):
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, txt, (x0, max(0, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for d in dropped_hard:
        draw(d["bbox"], (0, 255, 255), d.get("_drop_reason", "hard"))
    for d in dropped_lowq:
        draw(d["bbox"], (0, 0, 255), f"q={d.get('qscore',0):.2f}")
    for i, d in enumerate(kept):
        draw(d["bbox"], (0, 200, 0), f"{i+1}:{d.get('qscore',0):.2f}")
    try:
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        logger.warning("Failed saving debug overlay: %s", e)

# -----------------------------------------------------------------------------
# Main Extractor
# -----------------------------------------------------------------------------
class PDFChartExtractor:
    """
    Full pipeline (Upstage-only detection, strict region refinement, VLM extraction).
    """

    # -------------------- init & config --------------------
    def __init__(self):
        # Keys
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.upstage_key = os.getenv("UPSTAGE_API_KEY", "")

        # Models / OpenRouter app headers
        self.llm_model = os.getenv("LLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")
        self.openrouter_site = os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173")
        self.openrouter_app = os.getenv("OPENROUTER_APP_NAME", "pdf-extractor")

        # Upstage config
        self.allowed_categories = {
            c.strip().lower() for c in os.getenv("ALLOWED_UPSTAGE_CATEGORIES", "chart,table").split(",")
            if c.strip()
        }
        self.upstage_url = os.getenv("UPSTAGE_URL", "https://api.upstage.ai/v1/document-digitization")
        self.upstage_model = os.getenv("UPSTAGE_MODEL", "document-parse")
        self.upstage_timeout = int(os.getenv("UPSTAGE_TIMEOUT", "240"))
        self.upstage_use_base64 = os.getenv("UPSTAGE_USE_BASE64_CROPS", "false").lower() == "true"

        # Rendering
        self.poppler_path = os.getenv("POPPLER_PATH") or None
        self.pdf_dpi = int(os.getenv("PDF_DPI", "350"))
        self.upscale_factor = float(os.getenv("UPSCALE_FACTOR", "1.0"))

        # Perf / debug
        self.fast_dev = os.getenv("FAST_DEV", "false").lower() == "true"
        self.fast_dev_scale = float(os.getenv("FAST_DEV_IMAGE_SCALE", "0.8"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "3"))
        self.debug_timing = os.getenv("DEBUG_TIMING", "true").lower() == "true"
        self.debug_overlay = os.getenv("DEBUG_OVERLAY", "false").lower() == "true"

        # LLM
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

        # HTTP Session
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _layout_sort_and_tag(self, regions: List[Dict]) -> List[Dict]:
        """
        Sort regions visually (top->bottom, then left->right) using row buckets,
        and assign stable layout_order + human-friendly labels.
        """
        def key_fn(r: Dict):
            # be robust if bbox is malformed
            try:
                x0, y0, x1, y1 = r["bbox"]
            except Exception:
                return (1 << 30, 1 << 30)
            # bucket by Y to reduce jitter, then X within row
            return (int(y0 / 64) * 64, x0)

        # keep only items with bbox for ordering, append any weird ones at end
        with_bbox = [r for r in regions if r.get("bbox")]
        without_bbox = [r for r in regions if not r.get("bbox")]

        regions_sorted = sorted(with_bbox, key=key_fn) + without_bbox

        for i, r in enumerate(regions_sorted, start=1):
            r["layout_order"] = i
            cat = (r.get("category") or "").lower()
            r["label"] = f"{'Chart' if cat == 'chart' else 'Table'} {i:02d}"
        return regions_sorted


    # -------------------- public entrypoints --------------------
    def generate_thumbnails_for_pdf(self, pdf_path: str, stamped_name: str) -> int:
        """Render all pages to thumbnails (<name>_page_###_thumb.png) used by UI."""
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
        for p in range(1, page_count + 1):
            _ = self._render_page(pdf_path, p, stamped_name)  # writes thumb
        return page_count

    def process_page(self, pdf_path: str, page_number: int) -> Dict[str, Any]:
        logger.warning("PATCH_MARKER: process_page v2025-08-26C")  # <-- sanity beacon in logs
        t0 = time.time()
        with fitz.open(pdf_path) as doc:
            if not (1 <= page_number <= len(doc)):
                raise ValueError(f"Page {page_number} out of range 1..{len(doc)}")

        base = Path(pdf_path).name
        page_image = self._render_page(pdf_path, page_number, base)

        # Detect/Refine -> regions
        regions = self._detect_regions_upstage(pdf_path, page_number, page_image)

        # Filter to allowed categories FIRST
        allowed = getattr(self, "allowed_categories", {"chart", "table"})
        regions = [r for r in regions if (r.get("category") or "").lower() in allowed]

        # Stable visual order + labels (top→bottom, then left→right using row buckets)
        regions = self._layout_sort_and_tag(regions)

        # ----- ORDER TRACE (intended visual order) -----
        order_trace = " | ".join(
            [f"{r.get('layout_order', i+1):02d}:{r.get('label','?')}@{tuple(r['bbox'])}"
            for i, r in enumerate(regions) if r.get('bbox')]
        )
        logger.warning("ORDER TRACE: %s", order_trace)

        tables: List[Dict[str, Any]] = []
        debug_raw: List[Dict[str, Any]] = []

        # Parallelize region processing BUT assemble results in the same visual order
        pool_size = max(1, min(self.max_workers, len(regions), 8))
        finish_order_idx: List[int] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            fut_by_idx = {
                idx: pool.submit(self._process_region, page_image, region, page_number, idx)
                for idx, region in enumerate(regions)
            }
            results_by_idx = {idx: None for idx in fut_by_idx}
            for fut in as_completed(list(fut_by_idx.values())):
                idx = next(k for k, v in fut_by_idx.items() if v is fut)
                finish_order_idx.append(idx)
                try:
                    results_by_idx[idx] = fut.result(timeout=self.llm_timeout + 30)
                except Exception as e:
                    logger.exception("Region %s failed", idx)
                    results_by_idx[idx] = {
                        "table": {
                            "page": page_number,
                            "region": idx,
                            "layout_order": regions[idx].get("layout_order", idx + 1),
                            "label": regions[idx].get("label") or f"Chart {idx+1:02d}",
                            "data": [],
                            "row_count": 0, "total_rows": 0, "visible_rows": 0, "hidden_rows": 0,
                            "truncated": False, "has_more_rows": False, "remaining_rows": 0, "more_rows": 0,
                            "error": str(e),
                        },
                        "debug": {"page": page_number, "region": idx, "error": str(e)},
                    }

        # Place results strictly in the same order as 'regions'
        for idx, region in enumerate(regions):
            res = results_by_idx.get(idx)
            if not res:
                continue
            if "table" in res and isinstance(res["table"], dict):
                res["table"]["layout_order"] = region.get("layout_order", idx + 1)
                res["table"]["label"] = region.get("label") or res["table"].get("label") or f"Chart {idx+1:02d}"
                tables.append(res["table"])
            else:
                tables.append({
                    "page": page_number,
                    "region": idx,
                    "layout_order": region.get("layout_order", idx + 1),
                    "label": region.get("label") or f"Chart {idx+1:02d}",
                    "data": [],
                    "row_count": 0, "total_rows": 0, "visible_rows": 0, "hidden_rows": 0,
                    "truncated": False, "has_more_rows": False, "remaining_rows": 0, "more_rows": 0,
                })
            debug_raw.append((res or {}).get("debug", {}))

        # Final safety: sort by layout_order (should already be correct)
        tables.sort(key=lambda t: t.get("layout_order", 10**9))

        # ----- RESULT TRACE (actual payload order we return) -----
        result_trace = " | ".join([f"{t.get('layout_order','?'):02d}:{t.get('label','?')}" for t in tables])
        logger.warning("RESULT TRACE: %s", result_trace)

        # ----- FINISH TRACE (the order workers actually finished) -----
        logger.info("FINISH TRACE (worker completion order): %s", " -> ".join(str(i) for i in finish_order_idx))

        if self.debug_timing:
            logger.info("Page %d processed in %.2fs", page_number, time.time() - t0)

        return {"tables": tables, "debug_raw": debug_raw}

    # -------------------- rendering & detection --------------------
    def _render_page(self, pdf_path: str, page_number: int, stamped_name: str) -> np.ndarray:
        pages = convert_from_path(
            pdf_path, first_page=page_number, last_page=page_number,
            dpi=self.pdf_dpi, poppler_path=self.poppler_path
        )
        if not pages:
            raise RuntimeError(f"Failed to render page {page_number}")

        page_pil = pages[0]  # RGB
        page_np = np.array(page_pil)

        if self.upscale_factor > 1.0:
            h, w = page_np.shape[:2]
            page_np = cv2.resize(page_np, (int(w * self.upscale_factor), int(h * self.upscale_factor)), interpolation=cv2.INTER_CUBIC)

        # Write UI thumb
        thumb_name = f"{stamped_name}_page_{page_number:03d}_thumb.png"
        thumb_path = self.dirs["thumbnails"] / thumb_name
        w = 200
        scale = w / page_np.shape[1]
        thumb = cv2.resize(page_np, (w, max(1, int(page_np.shape[0] * scale))))
        cv2.imwrite(str(thumb_path), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        return page_np

    def _call_upstage_document_digitization(self, pdf_path: str) -> dict:
        """
        Minimal, known-good request to Upstage Document Digitization.
        Do NOT send extra fields; the API rejects unknown params.
        """
        api_key = self.upstage_key
        if not api_key:
            raise RuntimeError("UPSTAGE_API_KEY not set")

        url = self.upstage_url  # "https://api.upstage.ai/v1/document-digitization"
        headers = {"Authorization": f"Bearer {api_key}"}

        # Minimal form — ONLY 'model' is reliable here.
        data = {"model": self.upstage_model or "document-parse"}

        with open(pdf_path, "rb") as f:
            files = {"document": f}
            resp = self.session.post(url, headers=headers, files=files, data=data, timeout=self.upstage_timeout)

        try:
            resp.raise_for_status()
        except requests.HTTPError:
            logger.error("Upstage HTTP error %s body=%s", resp.status_code, resp.text[:1000])
            raise
        return resp.json()



    def _detect_regions_upstage(self, pdf_path: str, page_number: int, page_image: np.ndarray) -> List[Dict]:
        """
        Upstage-only detection for charts + tables:
        - categories: {'chart','table'} only
        - coords: normalized [0..1], PDF points, or pixels -> convert to rendered pixels
        - prefer Upstage base64 crops when provided
        - light pre-filter to drop tiny fragments; heavy lifting still done by _refine_detections(...)
        """
        H, W = page_image.shape[:2]

        # --- get the page size in PDF points (for points->pixels conversion)
        try:
            with fitz.open(pdf_path) as _doc_conv:
                rect = _doc_conv[page_number - 1].rect
                page_w_pt, page_h_pt = float(rect.width), float(rect.height)
        except Exception:
            # sensible defaults if something odd happens; most PDFs are ~612x792 pt (US Letter)
            page_w_pt, page_h_pt = 612.0, 792.0

        sx = W / max(1.0, page_w_pt)  # points -> pixels scale (x)
        sy = H / max(1.0, page_h_pt)  # points -> pixels scale (y)

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

        # pull elements (flat or per-page)
        elements = up_json.get("elements") or up_json.get("data") or []
        if not isinstance(elements, list):
            elements = []
            for pg in (up_json.get("pages") or []):
                try:
                    if int(pg.get("page", -1)) == int(page_number):
                        elements = pg.get("elements", [])
                        break
                except Exception:
                    continue

        # allowed categories (default to chart+table if not set on the instance)
        allowed = getattr(self, "allowed_categories", None)
        if not allowed:
            allowed = {"chart", "table"}

        # Helper: coords -> rendered pixels (detect normalized vs points vs pixels)
        def _to_px_box(coords_or_box):
            if isinstance(coords_or_box, (list, tuple)) and coords_or_box and isinstance(coords_or_box[0], dict):
                xs = [float(pt.get("x", 0)) for pt in coords_or_box]
                ys = [float(pt.get("y", 0)) for pt in coords_or_box]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            elif isinstance(coords_or_box, (list, tuple)) and len(coords_or_box) == 4:
                x0, y0, x1, y1 = map(float, coords_or_box)
            else:
                return None

            mn = min(x0, y0, x1, y1)
            mx = max(x0, y0, x1, y1)

            if 0.0 <= mn and mx <= 1.0:
                # normalized
                x0, x1 = x0 * W, x1 * W
                y0, y1 = y0 * H, y1 * H
            elif (-2.0 <= mn) and (mx <= max(page_w_pt, page_h_pt) + 2.0):
                # PDF points (with a small tolerance)
                x0, x1 = x0 * sx, x1 * sx
                y0, y1 = y0 * sy, y1 * sy
            else:
                # assume already pixels
                pass

            # clamp to image
            x0 = int(max(0, min(W, math.floor(x0))))
            x1 = int(max(0, min(W, math.ceil(x1))))
            y0 = int(max(0, min(H, math.floor(y0))))
            y1 = int(max(0, min(H, math.ceil(y1))))
            if x1 - x0 < 2 or y1 - y0 < 2:
                return None
            return [x0, y0, x1, y1]

        # --- collect regions (charts + tables only), prefer base64 crops
        regions: List[Dict] = []
        for el in elements:
            try:
                if int(el.get("page", 1)) != page_number:
                    continue
            except Exception:
                continue

            cat = (el.get("category") or el.get("type") or "").lower()
            if cat not in allowed:
                continue

            # Prefer Upstage-provided crop; bbox is fallback
            crop_bytes = None
            if self.upstage_use_base64:
                img_obj = el.get("image") or el.get("preview") or (el.get("content") or {}).get("image")
                if isinstance(img_obj, dict):
                    b64 = img_obj.get("base64")
                    url = img_obj.get("url")
                    if isinstance(b64, str):
                        import base64 as _b64
                        try:
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

            bbox = None
            if not crop_bytes:
                if el.get("coordinates"):
                    bbox = _to_px_box(el["coordinates"])
                if not bbox and (el.get("bounding_box") or el.get("bbox")):
                    bbox = _to_px_box(el.get("bounding_box") or el.get("bbox"))
                if not bbox:
                    # if no crop image and no usable bbox, skip
                    continue

            regions.append({
                "bbox": bbox,
                "category": cat,  # 'chart' or 'table'
                "confidence": float(el.get("confidence", 0.5)) if isinstance(el.get("confidence", 0.5), (int, float)) else 0.5,
                "crop_bytes": crop_bytes,
                "has_chart_marker": False,  # not used anymore
            })

        # --- NO HTML <img data-coord> fallback here -- it caused false positives

        # Light pre-filter to drop tiny fragments before refine (relative to page)
        MIN_SIDE = int(min(W, H) * 0.16)           # ~16% of min(page side)
        MIN_AREA = int((W * H) * 0.020)            # ~2.0% of page area
        filtered = []
        for r in regions:
            if r["bbox"] is None:
                continue
            x0, y0, x1, y1 = r["bbox"]
            ww, hh = x1 - x0, y1 - y0
            if ww < MIN_SIDE or hh < MIN_SIDE or (ww * hh) < MIN_AREA:
                r["_drop_reason"] = f"prefilter_small({ww}x{hh})"
                continue
            filtered.append(r)

        # Stable sort (top->bottom, then left->right)
        filtered.sort(key=lambda r: (r["bbox"][1] // 64, r["bbox"][0]))

        logging.info(
            "Upstage (allowed=%s) found=%d  after_prefilter=%d  on page %d",
            ",".join(sorted(allowed)), len(regions), len(filtered), page_number
        )

        # Final refine (NMS / scoring / ordering)
        refined = self._refine_detections(filtered, page_image)
        return refined["kept"]




    def _refine_detections(self, regions: List[Dict], page_img_rgb: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Refine Upstage detections:
        - HARD filter by size/aspect & allowed categories (chart/table only)
        - Cluster nearby boxes to absorb legend/axis fragments
        - Quality score, threshold, NMS, stable sort
        """
        H, W = page_img_rgb.shape[:2]
        allowed = getattr(self, "allowed_categories", {"chart", "table"})

        # ---- HARD filters (relative to page size)
        MIN_SIDE = int(min(W, H) * 0.16)     # ~16% of short side
        MIN_AREA = (W * H) * 0.020           # ~2.0% of page
        AR_MIN, AR_MAX = 0.30, 3.5           # avoid extreme slivers

        def _pass_hard(r):
            cat = (r.get("category") or "").lower()
            if cat not in allowed:
                r["_drop_reason"] = f"cat({cat})"; return False
            x0, y0, x1, y1 = r["bbox"]
            w, h = x1 - x0, y0 - y1 if y1 < y0 else y1 - y0
            # correct w,h
            w = x1 - x0; h = y1 - y0
            if w < MIN_SIDE or h < MIN_SIDE or (w * h) < MIN_AREA:
                r["_drop_reason"] = f"small({w}x{h})"; return False
            ar = w / float(h) if h > 0 else 0.0
            if ar < AR_MIN or ar > AR_MAX:
                r["_drop_reason"] = f"aspect({ar:.2f})"; return False
            return True

        first_pass, dropped_hard = [], []
        for r in regions:
            if r.get("bbox") and _pass_hard(r):
                first_pass.append(r)
            else:
                dropped_hard.append(r)

        if not first_pass:
            return {"kept": [], "dropped_lowq": [], "dropped_hard": dropped_hard}

        # ---- Cluster nearby fragments (merge legend/axis with chart/table)
        def _center(b):
            return ((b[0]+b[2])*0.5, (b[1]+b[3])*0.5)

        used = [False] * len(first_pass)
        clusters: List[List[Dict]] = []
        for i, ri in enumerate(first_pass):
            if used[i]: continue
            bx = ri["bbox"]; cx, cy = _center(bx)
            ci = [ri]; used[i] = True
            for j, rj in enumerate(first_pass):
                if used[j]: continue
                bj = rj["bbox"]
                iou = _iou_xyxy(bx, bj)
                dx, dy = _center(bj)
                dist = math.hypot(dx - cx, dy - cy)
                # generous criteria to absorb small neighbors
                if iou > 0.18 or dist < min(W, H) * 0.12:
                    ci.append(rj); used[j] = True
            clusters.append(ci)

        merged: List[Dict] = []
        for group in clusters:
            xs0, ys0, xs1, ys1 = [], [], [], []
            cat = "chart" if any((g.get("category") == "chart") for g in group) else "table"
            for g in group:
                x0, y0, x1, y1 = g["bbox"]
                xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
            merged.append({
                "bbox": [min(xs0), min(ys0), max(xs1), max(ys1)],
                "category": cat,
                "confidence": max(g.get("confidence", 0.5) for g in group),
            })

        # ---- Quality score + threshold + NMS
        for r in merged:
            r["qscore"] = _qscore(page_img_rgb, r["bbox"])

        QS_MIN = 0.20  # gentle; we already filtered hard by size and allowed cats
        qual_pass = [r for r in merged if r["qscore"] >= QS_MIN]
        dropped_lowq = [r for r in merged if r["qscore"] < QS_MIN]
        if not qual_pass:
            return {"kept": [], "dropped_lowq": dropped_lowq, "dropped_hard": dropped_hard}

        dedup = _nms(qual_pass, iou_thr=0.35, key="qscore")

        # ---- stable sort: top-to-bottom, then left-to-right
        def ltr_ttb_key(b):
            x0, y0, x1, y1 = b
            return (int(y0 / 64) * 64, x0)

        dedup.sort(key=lambda d: ltr_ttb_key(d["bbox"]))

        logger.info("Refined regions: kept=%d  dropped_lowq=%d  dropped_hard=%d",
                    len(dedup), len(dropped_lowq), len(dropped_hard))

        # Optional debug overlay
        if getattr(self, "debug_overlay", False):
            dbg = self.dirs["debug"] / "debug_refine.png"
            _overlay_debug(page_img_rgb, dedup, dropped_lowq, dropped_hard, dbg)

        return {"kept": dedup, "dropped_lowq": dropped_lowq, "dropped_hard": dropped_hard}


    # -------------------- region processing --------------------
    def _process_region(self, page_image: np.ndarray, region: Dict, page_number: int, region_idx: int) -> Optional[Dict]:
        t0 = time.time()
        try:
            allowed = getattr(self, "allowed_categories", {"chart", "table"})
            if (region.get("category") or "").lower() not in allowed:
                return None

            lo = int(region.get("layout_order", region_idx + 1))
            cat = (region.get("category") or "chart").lower()
            label = region.get("label") or (f"Chart {lo:02d}" if cat == "chart" else f"Table {lo:02d}")
            label_fs = label.replace(" ", "_")

            crops = self._generate_crops(page_image, region, page_number, region_idx)
            if not crops or "main" not in crops:
                raise RuntimeError("No usable crop produced")

            extraction_result = self._extract_with_llm(crops, page_number, region_idx)
            rows = (extraction_result or {}).get("data", []) or []

            # Persist full data (optional)
            csv_path = None
            json_path = None
            try:
                if rows:
                    df = pd.DataFrame(rows)
                    csv_name = f"page_{page_number:03d}_{label_fs}_{lo:02d}_data.csv"
                    json_name = f"page_{page_number:03d}_{label_fs}_{lo:02d}_data.json"
                    csv_path = self.dirs["enhanced"] / csv_name
                    json_path = self.dirs["enhanced"] / json_name
                    df.to_csv(csv_path, index=False)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(rows, f, ensure_ascii=False)
            except Exception as _e:
                logger.warning("Failed to persist full data for %s: %s", label, _e)

            n = len(rows)
            table_entry = {
                "page": page_number,
                "region": region_idx,
                "layout_order": lo,
                "label": label,
                "image": crops["main"]["filename"],
                "category": (region.get("category") or "unknown"),

                # FULL dataset — no slicing
                "data": rows,

                # Be explicit to suppress any “… more rows” UI hint
                "row_count": n, "total_rows": n, "visible_rows": n, "hidden_rows": 0,
                "truncated": False, "has_more_rows": False,
                "remaining_rows": 0, "more_rows": 0, "preview_count": n, "ui_has_more_rows": False,

                # chart metadata
                "chart_type": (extraction_result or {}).get("chart_type", "unknown"),
                "confidence": (extraction_result or {}).get("confidence", "low"),
                "note": (extraction_result or {}).get("note", ""),
                "series_meta": (extraction_result or {}).get("series_meta", []),
                "series_hints": (extraction_result or {}).get("series_hints", []),
                "extras_used": list(crops.keys()),
            }
            if csv_path: table_entry["data_csv"] = str(csv_path)
            if json_path: table_entry["data_json"] = str(json_path)

            debug_entry = {
                "page": page_number,
                "region": region_idx,
                "layout_order": lo,
                "label": label,
                "image": crops["main"]["filename"],
                "extras": [crops[k]["filename"] for k in crops if k != "main"],
                "raw": (extraction_result or {}).get("raw_response", "")[:500],
                "raw_fix": (extraction_result or {}).get("retry_response", "")[:500],
            }

            if self.debug_timing:
                logger.info("%s rows=%d processed in %.2fs", label, n, time.time() - t0)

            return {"table": table_entry, "debug": debug_entry}

        except Exception as e:
            logger.exception("Error processing %s (region %s)", region.get("label") or f"Region {region_idx}", region_idx)
            return {
                "table": {
                    "page": page_number,
                    "region": region_idx,
                    "layout_order": int(region.get("layout_order", region_idx + 1)),
                    "label": region.get("label") or f"Chart {int(region.get('layout_order', region_idx + 1)):02d}",
                    "error": str(e),
                    "data": [],
                    "row_count": 0, "total_rows": 0, "visible_rows": 0, "hidden_rows": 0,
                    "truncated": False, "has_more_rows": False,
                    "remaining_rows": 0, "more_rows": 0, "preview_count": 0, "ui_has_more_rows": False,
                },
                "debug": {"page": page_number, "region": region_idx, "error": str(e)},
            }

    def _generate_crops(self, page_image: np.ndarray, region: Dict, page_number: int, region_idx: int) -> Dict:
        """
        Use Upstage-provided crop if available; otherwise crop from bbox.
        Also create side strips + color emphasis + optional plot focus.
        """
        timestamp = int(time.time() * 1000)
        crops: Dict[str, Dict] = {}
        H, W = page_image.shape[:2]

        # 1) Upstage crop bytes
        cb = region.get("crop_bytes")
        main_img = None
        if cb:
            try:
                pil = Image.open(io.BytesIO(cb)).convert("RGB")
                main_img = np.array(pil)
                logger.info("Region %s: using Upstage crop image", region_idx)
            except Exception as e:
                logger.warning("Region %s: failed to decode Upstage crop (%s), falling back to bbox", region_idx, e)

        # 2) BBox crop as fallback
        if main_img is None:
            bbox = _safe_bbox(region.get("bbox"), W, H, pad_px=0)
            if bbox is None:
                raise RuntimeError(f"Invalid bbox after clamping: {region.get('bbox')}")
            x0, y0, x1, y1 = bbox
            main_img = page_image[y0:y1, x0:x1].copy(order="C")
            logger.info("Region %s: bbox=%s -> crop %dx%d", region_idx, bbox, main_img.shape[1], main_img.shape[0])

        if self.fast_dev and self.fast_dev_scale < 1.0:
            new_w = max(1, int(main_img.shape[1] * self.fast_dev_scale))
            new_h = max(1, int(main_img.shape[0] * self.fast_dev_scale))
            main_img = cv2.resize(main_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save main (use layout order & label if available)
        lo = int(region.get("layout_order", region_idx + 1))
        lbl = (region.get("label") or f"Region {lo}").replace(" ", "_")
        main_filename = f"page_{page_number:03d}_{lbl}_{lo:02d}_crop.png"
        main_path = self.dirs["enhanced"] / main_filename
        _pil_save_safe(_safe_pil_from_numpy_rgb(main_img), main_path)
        crops["main"] = {"path": main_path, "filename": main_filename, "image": main_img}


        # Optional axis strips (use layout order + label in names)
        lo = int(region.get("layout_order", region_idx + 1))
        lbl = (region.get("label") or f"Region {lo}").replace(" ", "_")

        if main_img.shape[1] > 240:
            # left strip
            left_strip = main_img[:, :120]
            lf = f"page_{page_number:03d}_{lbl}_{lo:02d}_axisL.png"
            lp = self.dirs["enhanced"] / lf
            _pil_save_safe(_safe_pil_from_numpy_rgb(left_strip), lp)
            crops["axis_left"] = {"path": lp, "filename": lf, "image": left_strip}

            # right strip
            right_strip = main_img[:, -120:]
            rf = f"page_{page_number:03d}_{lbl}_{lo:02d}_axisR.png"
            rp = self.dirs["enhanced"] / rf
            _pil_save_safe(_safe_pil_from_numpy_rgb(right_strip), rp)
            crops["axis_right"] = {"path": rp, "filename": rf, "image": right_strip}


        # Plot-focus (guarded; only when we detect a rectangular plot)
        plot_box = self._find_plot_box_safe(main_img)
        if plot_box:
            px0, py0, px1, py1 = plot_box
            plot_img = main_img[py0:py1, px0:px1]
            pf = f"page_{page_number}_region_{region_idx}_{timestamp}_plot.png"
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

    # -------------------- visual helpers --------------------
    def _find_plot_box_safe(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Heuristic: try to find inner plot rectangle (returns None if unsure)."""
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
                ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if ang < 10 or ang > 170:  # horizontal
                    h_lines.append(int((y1 + y2) / 2))
                elif 80 < ang < 100:      # vertical
                    v_lines.append(int((x1 + x2) / 2))
            if not h_lines or not v_lines:
                return None
            h_lines.sort(); v_lines.sort()
            left  = v_lines[0] if v_lines[0] > W * 0.1 else int(W * 0.15)
            right = v_lines[-1] if v_lines[-1] < W * 0.9 else int(W * 0.85)
            top    = h_lines[0] if h_lines[0] > H * 0.1 else int(H * 0.15)
            bottom = h_lines[-1] if h_lines[-1] < H * 0.9 else int(H * 0.85)
            left = max(0, min(left, W - 2)); right = max(left + 1, min(right, W))
            top  = max(0, min(top, H - 2));  bottom = max(top + 1, min(bottom, H))
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

    # -------------------- LLM extraction --------------------
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
            logger.info("LLM HTTP done region=%s len=%s", region_idx, (len(response) if response else 0))
            if response:
                extracted = self._parse_llm_response(response)
                if not self._validate_extraction(extracted):
                    retry = self._build_retry_prompt(extracted)
                    retry_resp = self._call_openrouter(retry, images_b64)
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
            "1) Extract the ACTUAL DATA POINTS only (lines, bars, plotted values)\n"
            "2) EXCLUDE overlays, reference lines, targets, averages, bands, annotations\n"
            "3) Return numeric Y-values using the Y-axis scale (interpolate if needed)\n\n"
            "Return JSON with EXACT structure:\n"
            "{\n"
            '  "chart_type": "line|bar|stacked_bar|dual_axis|combo|other",\n'
            '  "x_values": ["label1", "label2", ...],\n'
            '  "series": [ {"name":"Series Name","y_values":[123.4,...],"axis":"left|right|none","render":"line|bar"} ],\n'
            '  "y_axis_range": {"min": 0, "max": 1000},\n'
            '  "confidence": "high|medium|low",\n'
            '  "notes": "Any relevant observations"\n'
            "}\n"
            "Rules: y_values MUST be numbers and match x_values length.\n"
        )

    def _build_retry_prompt(self, previous_result: Dict) -> str:
        return "Retry. Focus on accurate numeric y_values (no overlays). Return the same JSON structure."

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
        if not r.ok:
            if r.status_code == 429:
                time.sleep(5)
                return None
            logger.error("LLM HTTP error %s head=%s", r.status_code, r.text[:600])
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
        for row in extracted["data"]:
            for k, v in row.items():
                if k != "X" and isinstance(v, (int, float)):
                    return True
        return False

    # -------------------- misc utils --------------------
    def _hash_image(self, image: np.ndarray) -> str:
        img_bytes = cv2.imencode(".png", image)[1].tobytes()
        return hashlib.md5(img_bytes).hexdigest()

    def _image_to_base64(self, image: np.ndarray) -> str:
        import base64
        buf = io.BytesIO()
        _safe_pil_from_numpy_rgb(image).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

# -----------------------------------------------------------------------------
# End module
# -----------------------------------------------------------------------------
