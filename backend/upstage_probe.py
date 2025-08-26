import os, requests

API_KEY = "up_Uosj99CTGdZYQgIkjT5CWpAvL1E2o"  # your real key
PDF = r"C:\Users\Mbomm\Projects\pdf-chart-extractor\backend\Sample3.pdf"  # full path to the PDF

with open(PDF, "rb") as f:
    resp = requests.post(
        "https://api.upstage.ai/v1/document-digitization",
        headers={"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"},
        files={"document": ("Sample3.pdf", f, "application/pdf")},
        data={
            "model": "document-parse",
            # keep it minimal first; we can add these after a 200:
            # "ocr": "auto",
            # "coordinates": "true",
            # "return_image": "true",
            # "output_formats": "['html']",
        },
        timeout=180,
    )
print(resp.status_code)
print(resp.text[:2000])
