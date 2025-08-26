import requests
url = "https://api.upstage.ai/v1/document-digitization"
headers = {"Authorization": "Bearer up_Uosj99CTGdZYQgIkjT5CWpAvL1E2o"}
pdf = r"C:\Users\Mbomm\Projects\pdf-chart-extractor\backend\Sample3.pdf"
with open(pdf, "rb") as f:
    files = {"document": f}
    data = {"model": "document-parse"}  # minimal and known-good
    r = requests.post(url, headers=headers, files=files, data=data, timeout=120)
print("Status:", r.status_code)
print(r.text[:800])
