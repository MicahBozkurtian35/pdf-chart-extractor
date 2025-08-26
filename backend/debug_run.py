# debug_run.py
import os, json, logging, sys, time
from PDF_Data_Extract import PDFChartExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("debug")

def main(pdf_path, page=1):
    os.environ["DEBUG_TRACE"] = "true"   # turn on extra logging in extractor (below)
    t0 = time.time()
    logger.info("=== DEBUG START === pdf=%s page=%s", pdf_path, page)
    ex = PDFChartExtractor()

    # sanity: show env the extractor will read
    for k in ("UPSTAGE_API_KEY","UPSTAGE_URL","UPSTAGE_MODEL","UPSTAGE_OCR","UPSTAGE_OUTPUT_FORMAT"):
        logger.info("env %s=%s", k, os.getenv(k, ""))

    # run the same pipeline the server uses
    out = ex.process_page(pdf_path, int(page))
    logger.info("RESULT tables=%d debug_raw=%d", len(out.get("tables", [])), len(out.get("debug_raw", [])))
    print(json.dumps(out, indent=2)[:1500])
    logger.info("=== DEBUG END (%.2fs) ===", time.time() - t0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python debug_run.py <PDF_PATH> [PAGE]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 1)
