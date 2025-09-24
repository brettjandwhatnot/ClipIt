import webview
import os
import shlex
import re
import threading
import uuid
import requests
import csv
import json
import logging
import subprocess
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from urllib.parse import urljoin, quote, unquote
import yt_dlp

from clipping_logic import run_clipping_process, get_ffmpeg_path

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)
UI_FOLDER = os.path.dirname(os.path.abspath(__file__))
jobs = {}

class Api:
    def __init__(self):
        self.window = None
    def set_window(self, window):
        self.window = window
    def select_folder(self):
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        return result[0] if result else None
    def select_file(self, file_types=('All files (*.*)',)):
        result = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)
        if not result: return None
        return result[0]

    def save_session_dialog(self, data):
        try:
            file_path = self.window.create_file_dialog(webview.SAVE_DIALOG, directory=os.path.expanduser('~'), save_filename='session.clipit')
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(data)
                return {'success': True, 'path': file_path}
            return {'success': False, 'error': 'Save cancelled by user.'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def load_session_dialog(self):
        try:
            file_types = ('Clipit Session (*.clipit)',)
            result = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)
            if not result:
                return {'success': False, 'error': 'Load cancelled by user.'}
            
            file_path = result[0]
            with open(file_path, 'r') as f:
                content = f.read()
            return {'success': True, 'data': content, 'path': file_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}

def _parse_curl_command(curl_string: str):
    cleaned_string = re.sub(r'\\\s*\n?', ' ', curl_string.strip())
    try:
        args = shlex.split(cleaned_string)
        url = next((arg for arg in args if arg.startswith('http')), None)
        if not url: raise ValueError("Could not find a URL in the cURL command.")
        headers = {}
        i = 0
        while i < len(args):
            if args[i] in ('-H', '--header'):
                key, value = args[i + 1].split(':', 1)
                headers[key.strip()] = value.strip()
                i += 2
            else:
                i += 1
        return url, headers
    except Exception as e:
        raise ValueError(f"Could not parse cURL command: {e}")

@app.route('/')
@app.route('/index.html')
def serve_index():
    return send_from_directory(UI_FOLDER, 'index.html')

@app.route('/editor.html')
def serve_editor():
    return send_from_directory(UI_FOLDER, 'editor.html')

@app.route('/api/video_info')
def get_video_info():
    curl_command = request.args.get('curl')
    if not curl_command: return jsonify({'error': 'Missing cURL parameter'}), 400
    try:
        url, headers = _parse_curl_command(curl_command)
        ydl_opts = {'quiet': True, 'http_headers': headers, 'skip_download': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info("Attempting to get video info for URL: %s", url)
            info_dict = ydl.extract_info(url, download=False)
            logging.info("Successfully retrieved video info.")
            duration = info_dict.get('duration')
        if duration is None: raise ValueError("Failed to determine video duration.")
        return jsonify({'duration': duration})
    except Exception as e:
        logging.error("Failed to get video info: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/hls_proxy')
def hls_proxy():
    curl_command = request.args.get('curl')
    target_url_encoded = request.args.get('target')
    if not curl_command or not target_url_encoded: return "Missing cURL or target URL parameters", 400
    try:
        _, headers = _parse_curl_command(curl_command)
        target_url = unquote(target_url_encoded)
        if target_url.strip().lower().startswith('curl'):
            target_url, _ = _parse_curl_command(target_url)
        
        logging.info("Proxying request to target URL: %s", target_url)
        r = requests.get(target_url, headers=headers, stream=True)
        r.raise_for_status()
        logging.info("Request to target URL successful (Status: %s).", r.status_code)
        
        content_type = r.headers.get('Content-Type', '')
        if 'mpegurl' in content_type or target_url.endswith('.m3u8'):
            manifest_content = r.text
            lines = manifest_content.splitlines()
            new_lines = []
            base_url_for_resolving = target_url.rsplit('/', 1)[0] + '/'
            for line in lines:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    absolute_line_url = urljoin(base_url_for_resolving, stripped_line)
                    encoded_line_url = quote(absolute_line_url)
                    encoded_curl = quote(curl_command)
                    new_lines.append(f"/api/hls_proxy?target={encoded_line_url}&curl={encoded_curl}")
                else:
                    new_lines.append(line)
            rewritten_manifest = "\n".join(new_lines)
            return Response(rewritten_manifest, mimetype='application/vnd.apple.mpegurl')
        else:
            return Response(r.iter_content(chunk_size=8192), content_type=content_type)
    except Exception as e:
        logging.exception("HLS Proxy Error for target %s", target_url_encoded)
        return str(e), 500

@app.route('/api/clip', methods=['POST'])
def start_clip_job():
    params = request.json
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'pending', 'params': params, 'progress': 'Job queued'}
    thread = threading.Thread(target=run_clipping_process, args=(job_id, params, jobs))
    thread.daemon = True
    thread.start()
    return jsonify({'message': 'Job started.', 'job_id': job_id}), 202


@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)

@app.route('/api/preview/<job_id>/<path:filename>')
def serve_preview(job_id, filename):
    job = jobs.get(job_id)
    if not job or 'output_folder' not in job:
        return "Job not found or output folder not specified.", 404
    output_folder = job['output_folder']
    if not os.path.isdir(output_folder):
        return "Output directory not found.", 404
    try:
        return send_from_directory(output_folder, filename, as_attachment=False)
    except FileNotFoundError:
        return "File not found.", 404
    except Exception as e:
        return "Error serving file.", 500

@app.route('/api/waveform/<job_id>/<path:filename>')
def get_waveform_data(job_id, filename):
    job = jobs.get(job_id)
    if not job or 'output_folder' not in job:
        return jsonify({'error': 'Job not found or output folder not specified.'}), 404
    
    video_path = os.path.join(job['output_folder'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found.'}), 404

    try:
        ffmpeg_path = get_ffmpeg_path()
        command = [
            ffmpeg_path,
            '-i', video_path,
            '-ac', '1',
            '-filter:a', 'aresample=8000',
            '-map', '0:a',
            '-c:a', 'pcm_s16le',
            '-f', 'data',
            '-'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr.decode('utf-8', 'ignore')}")

        audio_data = np.frombuffer(stdout, dtype=np.int16)
        normalized_data = audio_data / 32768.0
        
        num_samples = 500
        step = len(normalized_data) // num_samples
        if step == 0:
             return jsonify(np.abs(normalized_data).tolist())

        peaks = [np.max(np.abs(normalized_data[i:i+step])) for i in range(0, len(normalized_data), step)]
        
        return jsonify(peaks)
    except Exception as e:
        logging.error(f"Waveform generation failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/register_job_folder', methods=['POST'])
def register_job_folder():
    data = request.json
    job_id = data.get('job_id')
    output_folder = data.get('output_folder')
    if job_id and output_folder:
        if job_id not in jobs:
            jobs[job_id] = {}
        jobs[job_id]['output_folder'] = output_folder
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Missing job_id or output_folder'}), 400

def _read_csv_data(path, file_type):
    data = []
    with open(path, mode='r', encoding='utf-8-sig') as csvfile:
        if file_type == 'transcript':
            reader = csv.reader(csvfile)
            try: next(reader)
            except StopIteration: csvfile.seek(0)
            for row in reader:
                if not row: continue
                line = row[0]
                match = re.match(r'"?(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\s*-->\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?"?\s*(.*)', line)
                if match:
                    start_time, text = match.groups()
                    cleaned_text = text.strip().strip('"')
                    data.append({'start': start_time.strip(), 'text': cleaned_text})
        else:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append({str(k).lower().strip() if k else f'unknown_col_{i}': v for i, (k, v) in enumerate(row.items())})
    return data

@app.route('/api/csv_data')
def get_csv_data():
    path = request.args.get('path')
    file_type = request.args.get('type')
    if not path: return jsonify({'error': 'Missing path parameter'}), 400
    try:
        data = _read_csv_data(path, file_type)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.json
    analysis_text = (
        "This frame shows a presenter holding up what appears to be a stylish handbag. "
        "The product is well-lit and in focus against a neutral background. "
        "The composition is strong, making this a potentially good thumbnail or opening shot for an ad."
    )
    return jsonify({'analysis': analysis_text})

@app.route('/api/chat', methods=['POST'])
def ai_chat():
    data = request.json
    user_message = data.get('message', '').lower()
    if 'thumbnail' in user_message or 'shot' in user_message:
        response_text = "Yes, I believe this would make an excellent thumbnail due to the clear product visibility and good lighting."
    elif 'product' in user_message or 'handbag' in user_message or 'item' in user_message:
        response_text = "The product appears to be a women's handbag, possibly made of leather. It has a classic design with metallic hardware."
    elif 'lighting' in user_message or 'background' in user_message:
        response_text = "The lighting is soft and even, highlighting the product's features without harsh shadows. The background is simple, ensuring the product is the main focus."
    else:
        response_text = "I can analyze the current video frame for you. Ask me about the product, its presentation, or if it would make a good thumbnail."
    return jsonify({'response': response_text})

@app.route('/api/ai_find_moments', methods=['POST'])
def ai_find_moments():
    data = request.json
    prompt = data.get('prompt', '').lower()
    image_data = data.get('image')
    sales_path = data.get('sales_csv_path')

    if not sales_path:
        return jsonify({'error': 'Sales CSV path is required.'}), 400

    try:
        sales_data = _read_csv_data(sales_path, 'sales')
        sold_items = [item for item in sales_data if item.get('status', '').upper() == 'SOLD' and item.get('is_pending_payment', '').lower() == 'false' and item.get('sold_price_amount')]

        found_products = []
        if image_data:
            logging.info("Image data received, simulating visual search...")
            specific_item = next((item for item in sold_items if 'leather jacket' in item.get('product_name', '').lower()), None)
            if specific_item:
                found_products = [specific_item]
            else:
                found_products = sold_items[1:2] if len(sold_items) > 1 else sold_items
        elif 'expensive' in prompt or 'highest' in prompt:
            sold_items.sort(key=lambda x: int(x.get('sold_price_amount', 0)), reverse=True)
            found_products = sold_items[:3]
        elif 'first' in prompt or 'early' in prompt:
            def to_seconds_for_sort(time_str):
                try:
                    parts = str(time_str).strip().replace(',', '.').split(':')
                    if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
                    return float('inf')
                except: return float('inf')
            
            sold_items.sort(key=lambda x: to_seconds_for_sort(x.get('time')))
            found_products = sold_items[:3]
        else:
            found_products = [item for item in sold_items if item.get('product_name') and prompt in item.get('product_name').lower()][:5]

        return jsonify(found_products)
    except Exception as e:
        logging.error(f"AI search failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    api_instance = Api()
    window = webview.create_window('Clip It!', app, js_api=api_instance, width=1600, height=900, min_size=(1200, 700))
    api_instance.set_window(window)
    webview.start(debug=True)

