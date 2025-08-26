import os, requests
from PDF_Data_Extract import PDFChartExtractor

PDF = r".\Sample3.pdf"

def call_probe_style(pdf_path):
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {os.getenv('UPSTAGE_API_KEY','')}"}
    data = {
        "model": "document-parse",
        "ocr": "force",
        "base64_encoding": "['table']",
        "output_format": "json",
    }
    with open(pdf_path, "rb") as f:
        files = {"document": f}
        r = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    return {
        "who": "probe",
        "url": url,
        "data": data,
        "status": r.status_code,
        "body": r.text[:800],
    }

def call_extractor_style(pdf_path):
    ex = PDFChartExtractor()
    url = getattr(ex, "upstage_url", None)
    headers = {"Authorization": f"Bearer {os.getenv('UPSTAGE_API_KEY','')}"}
    data = {
        "model": os.getenv("UPSTAGE_MODEL", "document-parse") or "document-parse",
        "ocr": os.getenv("UPSTAGE_OCR", "force") or "force",
        "base64_encoding": os.getenv("UPSTAGE_BASE64", "['table']") or "['table']",
        "output_format": os.getenv("UPSTAGE_OUTPUT_FORMAT", "json") or "json",
    }
    with open(pdf_path, "rb") as f:
        files = {"document": f}
        r = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    return {
        "who": "extractor",
        "url": url,
        "data": data,
        "status": r.status_code,
        "body": r.text[:800],
    }

def show(x):
    print(f"=== {x['who']} ===")
    print("url:", x["url"])
    print("data keys:", list(x["data"].keys()))
    print("data:", x["data"])
    print("status:", x["status"])
    print("body_head:", x["body"].replace("\n", "\\n")[:800])
    print()

def main():
    a = call_probe_style(PDF)
    b = call_extractor_style(PDF)
    show(a)
    show(b)

    diffs = []
    keys = set(a["data"].keys()).union(b["data"].keys())
    for k in keys:
        if a["data"].get(k) != b["data"].get(k):
            diffs.append((k, a["data"].get(k), b["data"].get(k)))
    if diffs:
        print("=== DATA DIFFS (probe vs extractor) ===")
        for k, av, bv in diffs:
            print(f"{k}: probe={av!r}  extractor={bv!r}")
    else:
        print("=== DATA match exactly ===")

if __name__ == "__main__":
    main()
