#!/usr/bin/env python3
"""
Flask API server for PDF Chart Data Extractor
- Upstage-only extractor
- Structured JSON logging
- No duplicate Werkzeug banners
"""

import os
import json
import logging
import traceback
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pythonjsonlogger import jsonlogger

# --- Load env early
load_dotenv()
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# --- Import extractor
from PDF_Data_Extract import PDFChartExtractor

# ---------------------------
# Silence Werkzeug banners/log spam (dev server only)
# ---------------------------
try:
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
    _logging.getLogger("werkzeug").propagate = False

    # Monkey-patch the low-level serving logger that prints the banners
    from werkzeug import serving as _serving
    # No request line logs
    _serving.WSGIRequestHandler.log_request = lambda *a, **k: None
    _serving.WSGIRequestHandler.log_message = lambda *a, **k: None
    # No internal _log (covers the “* Serving Flask app” banners)
    _serving._log = lambda *a, **k: None
except Exception:
    pass

# ---------------------------
# Structured JSON logging
# ---------------------------
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(ts)s %(levelname)s %(message)s %(event)s %(dur_ms)s %(page)s %(region)s',
    timestamp='ts'
)
logHandler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# ---------------------------
# Flask app + CORS
# ---------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------
# Folders / config
# ---------------------------
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB default
ALLOWED_EXTENSIONS = {'pdf'}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
THUMBNAIL_FOLDER = BASE_DIR / 'thumbnails'
ENHANCED_FOLDER = BASE_DIR / 'enhanced_images'
DEBUG_FOLDER = BASE_DIR / 'debug'

for folder in [UPLOAD_FOLDER, THUMBNAIL_FOLDER, ENHANCED_FOLDER, DEBUG_FOLDER]:
    folder.mkdir(exist_ok=True)

# ---------------------------
# Extractor
# ---------------------------
extractor = PDFChartExtractor()

# ---------------------------
# Helpers
# ---------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def error_response(code: int, error_code: str, message: str, details=None):
    resp = {'error_code': error_code, 'message': message}
    if details is not None:
        resp['details'] = details
    logger.error('API error', extra={
        'event': 'api_error',
        'error_code': error_code,
        'details': details,
        'ts': time.time()
    })
    return jsonify(resp), code

# ---------------------------
# Routes
# ---------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Accept PDF upload, save it, and return metadata
    Response: { pdf_path, page_count, thumbnail }
    """
    t0 = time.time()

    try:
        if 'file' not in request.files:
            return error_response(400, 'NO_FILE', 'No file part in request')

        file = request.files['file']
        if file.filename == '':
            return error_response(400, 'NO_FILENAME', 'No file selected')

        if not allowed_file(file.filename):
            return error_response(400, 'INVALID_TYPE', 'Only PDF files are accepted')

        # Size check
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > MAX_FILE_SIZE:
            return error_response(
                400, 'FILE_TOO_LARGE',
                f'File size {file_size} exceeds maximum {MAX_FILE_SIZE} bytes'
            )

        # Save with timestamp prefix (matches your logs/UI)
        filename = secure_filename(file.filename)
        stamped = f"{int(time.time())}_{filename}"
        pdf_path = UPLOAD_FOLDER / stamped
        file.save(pdf_path)

        # Generate thumbnails for ALL pages via extractor
        try:
            page_count = extractor.generate_thumbnails_for_pdf(str(pdf_path), stamped)
            first_thumb = f"{stamped}_page_001_thumb.png" if page_count >= 1 else None
        except Exception as e:
            logger.error("PDF processing error: %s", str(e), extra={
                'event': 'pdf_process_error',
                'stack': traceback.format_exc(),
                'ts': time.time()
            })
            return error_response(422, 'PDF_PARSE_ERROR', f'Failed to process PDF: {e}')

        dur_ms = int((time.time() - t0) * 1000)
        logger.info('PDF uploaded', extra={
            'event': 'pdf_upload',
            'pdf_filename': stamped,
            'page_count': page_count,
            'dur_ms': dur_ms,
            'ts': time.time()
        })

        return jsonify({
            'pdf_path': stamped,
            'page_count': page_count,
            'thumbnail': first_thumb
        })

    except Exception as e:
        logger.error("Upload error: %s", str(e), extra={
            'event': 'upload_error',
            'stack': traceback.format_exc(),
            'ts': time.time()
        })
        return error_response(500, 'INTERNAL_ERROR', f'Upload failed: {e}')

@app.route('/thumbnail/<path:filename>', methods=['GET'])
def serve_thumbnail(filename):
    try:
        return send_from_directory(THUMBNAIL_FOLDER, filename)
    except FileNotFoundError:
        return error_response(404, 'THUMBNAIL_NOT_FOUND', f'Thumbnail {filename} not found')

@app.route('/process_page', methods=['POST'])
def process_page():
    """
    Body: { pdf_path: string, page_number: int }
    Returns extractor output: { tables: [...], debug_raw: [...] }
    """
    t0 = time.time()
    try:
        data = request.get_json(silent=True) or {}
        pdf_path = data.get('pdf_path')
        page_number = data.get('page_number')

        if not pdf_path:
            return error_response(400, 'MISSING_PDF_PATH', 'pdf_path is required')
        if page_number is None:
            return error_response(400, 'MISSING_PAGE_NUMBER', 'page_number is required')

        full_pdf_path = UPLOAD_FOLDER / pdf_path
        if not full_pdf_path.exists():
            return error_response(404, 'PDF_NOT_FOUND', f'PDF {pdf_path} not found')

        logger.info('Processing page', extra={
            'event': 'process_start',
            'page': page_number,
            'ts': time.time()
        })

        try:
            result = extractor.process_page(str(full_pdf_path), int(page_number))
            dur_ms = int((time.time() - t0) * 1000)
            logger.info('Page processed successfully', extra={
                'event': 'process_complete',
                'page': page_number,
                'regions': len(result.get('tables', [])),
                'dur_ms': dur_ms,
                'ts': time.time()
            })
            return jsonify(result)
        except ValueError as e:
            return error_response(400, 'INVALID_PAGE', str(e), {'page': page_number})
        except Exception as e:
            logger.error("Processing error: %s", str(e), extra={
                'event': 'process_error',
                'page': page_number,
                'stack': traceback.format_exc(),
                'ts': time.time()
            })
            return error_response(500, 'PROCESS_ERROR', f'Failed to process page: {e}', {'page': page_number})

    except Exception as e:
        logger.error("Request error: %s", str(e), extra={
            'event': 'request_error',
            'stack': traceback.format_exc(),
            'ts': time.time()
        })
        return error_response(500, 'INTERNAL_ERROR', f'Request failed: {e}')

@app.route('/images/enhanced/<path:filename>', methods=['GET'])
def serve_enhanced_image(filename):
    try:
        return send_from_directory(ENHANCED_FOLDER, filename)
    except FileNotFoundError:
        return error_response(404, 'IMAGE_NOT_FOUND', f'Image {filename} not found')

@app.route('/debug/<path:filename>', methods=['GET'])
def serve_debug_file(filename):
    try:
        return send_from_directory(DEBUG_FOLDER, filename)
    except FileNotFoundError:
        return error_response(404, 'DEBUG_FILE_NOT_FOUND', f'Debug file {filename} not found')

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    # Prevent the dev server reloader from spawning a second process
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

    logger.info('Starting server', extra={
        'event': 'server_start',
        'port': port,
        'debug': debug,
        'ts': time.time()
    })

    # Force no reloader (avoids double-run) and silence request logs
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)
