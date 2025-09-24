import webview
import os
import shlex
import re
import threading
import uuid
import requests
import csv
import json
import traceback # Import traceback for detailed error logging
import sys # Import sys for flushing output
from flask import Flask, request, jsonify, Response, send_from_directory
from urllib.parse import urljoin, quote, unquote, urlparse
import yt_dlp

from clipping_logic import run_clipping_process

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
        return result[0] # Just return the path

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

def _parse_curl_command(curl_string):
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
def serve_index():
    return send_from_directory(UI_FOLDER, 'index.html')

@app.route('/api/video_info')
def get_video_info():
    curl_command = request.args.get('curl')
    if not curl_command: return jsonify({'error': 'Missing cURL parameter'}), 400
    try:
        url, headers = _parse_curl_command(curl_command)
        ydl_opts = {'quiet': True, 'http_headers': headers, 'skip_download': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            duration = info_dict.get('duration')
        if duration is None: raise ValueError("Failed to determine video duration.")
        return jsonify({'duration': duration})
    except Exception as e:
        print(f"Info Error: {e}")
        return jsonify({'error': str(e)}), 500

def generate_manifest(target_url, headers, curl_command):
    """A generator function to stream the rewritten manifest."""
    try:
        with requests.get(target_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            base_url_for_resolving = target_url.rsplit('/', 1)[0] + '/'
            for line_bytes in r.iter_lines():
                if line_bytes:
                    line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                    if line_str and not line_str.startswith('#'):
                        absolute_line_url = urljoin(base_url_for_resolving, line_str)
                        encoded_line_url = quote(absolute_line_url)
                        encoded_curl = quote(curl_command)
                        yield f"/api/hls_proxy?target={encoded_line_url}&curl={encoded_curl}\n"
                    else:
                        yield f"{line_str}\n"
    except Exception as e:
        print(f"--- Manifest Generation Error ---")
        traceback.print_exc()
        sys.stdout.flush()
        yield f"# FFMPEG_PROXY_ERROR: {e}\n"


@app.route('/api/hls_proxy')
def hls_proxy():
    curl_command = request.args.get('curl')
    target_url_encoded = request.args.get('target')
    print(f"--- HLS PROXY REQUEST ---")
    print(f"Target URL (encoded): {target_url_encoded}")
    sys.stdout.flush()

    if not curl_command or not target_url_encoded: 
        return "Missing cURL or target URL parameters", 400
    
    try:
        _, headers = _parse_curl_command(curl_command)
        target_url = unquote(target_url_encoded)
        print(f"Target URL (decoded): {target_url}")
        sys.stdout.flush()

        if target_url.strip().lower().startswith('curl'):
            target_url, _ = _parse_curl_command(target_url)
        
        parsed_url = urlparse(target_url)
        is_manifest = parsed_url.path.endswith('.m3u8')
        
        if is_manifest:
            print("Type: Manifest. Rewriting and streaming.")
            sys.stdout.flush()
            return Response(generate_manifest(target_url, headers, curl_command), mimetype='application/vnd.apple.mpegurl')
        else:
            print("Type: Media Segment. Streaming directly WITHOUT original auth headers.")
            sys.stdout.flush()
            # For segments, the auth tokens are in the URL. DO NOT send original headers.
            r = requests.get(target_url, stream=True)
            r.raise_for_status() 
            return Response(r.iter_content(chunk_size=8192), content_type=r.headers.get('Content-Type', ''))

    except Exception as e:
        print("--- HLS PROXY ERROR ---")
        traceback.print_exc()
        print("-----------------------")
        sys.stdout.flush()
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
        print(f"Preview Error: {e}")
        return "Error serving file.", 500

# Endpoint to repopulate job data after loading a session
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

@app.route('/api/csv_data')
def get_csv_data():
    path = request.args.get('path')
    file_type = request.args.get('type')
    if not path: return jsonify({'error': 'Missing path parameter'}), 400
    try:
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
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    api_instance = Api()
    window = webview.create_window('Clip It!', app, js_api=api_instance, width=1600, height=900, min_size=(1200, 700))
    api_instance.set_window(window)
    webview.start(debug=True)

