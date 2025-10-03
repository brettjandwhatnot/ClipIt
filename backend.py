import webview
import os
from dotenv import load_dotenv
import shlex
import re
import threading
import uuid
import requests
import csv
import json
import logging
import subprocess
import base64
import random
import google.generativeai as genai
from openai import OpenAI
from flask import Flask, request, jsonify, Response, send_from_directory
from urllib.parse import urljoin, quote, unquote
import yt_dlp
from datetime import timedelta

from clipping_logic import run_clipping_process

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- GLOBALS ---
app = Flask(__name__)
UI_FOLDER = os.path.dirname(os.path.abspath(__file__))
jobs = {}
chat_histories = {} 

# --- API CLIENT INITIALIZATION ---
# Securely load API keys from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logging.info("OpenAI client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
else:
    logging.warning("OPENAI_API_KEY environment variable not set. OpenAI features will be disabled.")

gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # FIXED: Correct model name is essential for the API call to work.
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        logging.info("Gemini client configured successfully.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini client: {e}")
else:
    logging.warning("GEMINI_API_KEY environment variable not set. Gemini features will be disabled.")

# --- CONSTANTS & GLOBALS ---
default_ai_prompt = """
Search for the most engaging clips to be used as an ad based on the following priorities. Was it a great deal. Was the item and price point social worthy, think if the item and price would catch someones attention. Is there hype around the item? Will it make for a good social clip? Remember these will be ads and we want to showcase product moments, so be sure to prioritize clips that have products sold as the forefront.
""".strip()

FRAME_ANALYSIS_PROMPT = """
You are a social media expert for Whatnot. Your task is to analyze a screenshot from a live stream and provide a concise, compelling summary for a potential social media ad. Follow this exact format:

**Item Name:** [The name of the item]
**Ad Potential Score:** [A score from 1-100 representing its viral potential]
**Estimated Market Value:** [A realistic price range for the item]

[A brief, engaging overview of 3-4 sentences explaining the item's popularity, history, and why it's a desirable find.]

Use the provided product name and transcript context to inform your analysis. You must search the web to verify the item and its market value. Provide reference links if available. Avoid the words "Auction," "Addiction," and "Gamble."
""".strip()

# --- HELPER FUNCTIONS ---
def to_seconds(time_str):
    if not time_str: return None
    try:
        s = str(time_str).strip().replace(',', '.')
        parts = s.split(':')
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except (ValueError, IndexError, TypeError): return None

def parse_livestream_time(time_str):
    if not time_str or not isinstance(time_str, str):
        return None
    
    parts = {'hr': 0, 'min': 0, 's': 0}
    is_negative = time_str.startswith('-')
    if is_negative:
        time_str = time_str[1:]

    matches = re.findall(r'(\d+)\s*(hr|min|s)', time_str)
    if not matches: return None

    for value, unit in matches:
        parts[unit] = int(value)
        
    total_seconds = parts['hr'] * 3600 + parts['min'] * 60 + parts['s']
    return -total_seconds if is_negative else total_seconds

# --- REINTRODUCED FROM backend.py ---
def format_time_for_srt(seconds):
    """Converts seconds to HH:MM:SS,ms format."""
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
# --- END REINTRODUCED SECTION ---

# --- WEBVIEW API CLASS ---
class Api:
    def __init__(self):
        self.window = None

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
            if not result: return {'success': False, 'error': 'Load cancelled by user.'}
            file_path = result[0]
            with open(file_path, 'r') as f:
                content = f.read()
            return {'success': True, 'data': content, 'path': file_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}

# --- FLASK ROUTES ---
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

def _get_url_and_headers(source_input: str):
    source_input = source_input.strip()
    default_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    if source_input.startswith('https://whatnot.com/player/'):
        manifest_url_encoded = source_input.split('/player/')[-1]
        manifest_url = unquote(manifest_url_encoded)
        while '%' in manifest_url:
            manifest_url = unquote(manifest_url)

        headers = {
            'User-Agent': default_user_agent,
            'Referer': 'https://www.whatnot.com/'
        }
        return manifest_url, headers
    else:
        url, headers = _parse_curl_command(source_input)
        if not any(key.lower() == 'user-agent' for key in headers):
            headers['User-Agent'] = default_user_agent
        if not any(key.lower() == 'referer' for key in headers):
             headers['Referer'] = 'https://www.whatnot.com/'
        return url, headers

@app.route('/')
def serve_index():
    return send_from_directory(UI_FOLDER, 'index.html')

@app.route('/api/get_default_prompt', methods=['GET'])
def get_default_prompt():
    return jsonify({'prompt': default_ai_prompt})

@app.route('/api/set_default_prompt', methods=['POST'])
def set_default_prompt():
    global default_ai_prompt
    data = request.json
    new_prompt = data.get('prompt')
    if new_prompt and isinstance(new_prompt, str):
        default_ai_prompt = new_prompt
        logging.info("Default AI prompt updated.")
        return jsonify({'success': True})
    return jsonify({'error': 'Invalid prompt data'}), 400

@app.route('/api/video_info')
def get_video_info():
    source_input = request.args.get('curl')
    if not source_input: return jsonify({'error': 'Missing source URL or cURL parameter'}), 400
    try:
        url, headers = _get_url_and_headers(source_input)
        ydl_opts = {'quiet': True, 'http_headers': headers, 'skip_download': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            duration = info_dict.get('duration')
        if duration is None: raise ValueError("Failed to determine video duration.")
        return jsonify({'duration': duration})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hls_proxy')
def hls_proxy():
    source_input = request.args.get('curl')
    target_url_encoded = request.args.get('target')
    if not source_input or not target_url_encoded: return "Missing source or target parameters", 400
    
    try:
        _, headers = _get_url_and_headers(source_input)
        
        target_url_to_fetch = unquote(target_url_encoded)
        if target_url_to_fetch.startswith('https://whatnot.com/player/'):
             actual_url_to_fetch, _ = _get_url_and_headers(target_url_to_fetch)
        else:
             actual_url_to_fetch = target_url_to_fetch

        r = requests.get(actual_url_to_fetch, headers=headers, stream=True)
        r.raise_for_status()
        
        content_type = r.headers.get('Content-Type', '')
        if 'mpegurl' in content_type or actual_url_to_fetch.endswith('.m3u8'):
            base_url_for_resolving = actual_url_to_fetch.rsplit('/', 1)[0] + '/'
            encoded_source = quote(source_input)
            def generate_rewritten_manifest():
                for line_bytes in r.iter_lines():
                    line = line_bytes.decode('utf-8', errors='ignore')
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        absolute_line_url = urljoin(base_url_for_resolving, stripped_line)
                        encoded_line_url = quote(absolute_line_url)
                        yield f"/api/hls_proxy?target={encoded_line_url}&curl={encoded_source}\n"
                    else:
                        yield f"{line}\n"
            return Response(generate_rewritten_manifest(), mimetype='application/vnd.apple.mpegurl')
        else:
            return Response(r.iter_content(chunk_size=8192), content_type=content_type)
    except Exception as e:
        logging.error(f"HLS Proxy Error: {e}")
        return str(e), 500

@app.route('/api/clip', methods=['POST'])
def start_clip_job():
    params = request.json
    job_id = str(uuid.uuid4())
    try:
        source_input = params.get('source_url', '')
        url, headers = _get_url_and_headers(source_input)
        params['resolved_url'] = url
        params['resolved_headers'] = headers
    except Exception as e:
        jobs[job_id] = {'status': 'failed', 'error': str(e)}
        return jsonify({'error': str(e)}), 400

    jobs[job_id] = {'status': 'pending', 'params': params, 'progress': 'Job queued'}
    thread = threading.Thread(target=run_clipping_process, args=(job_id, params, jobs))
    thread.daemon = True
    thread.start()
    return jsonify({'message': 'Job started.', 'job_id': job_id}), 202

@app.route('/api/cancel_job/<job_id>', methods=['GET'])
def cancel_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job.get('status') == 'running' or job.get('status') == 'pending':
        job['status'] = 'cancelling'
        process = job.get('process')
        if process:
            try:
                process.kill()
                logging.info(f"Sent kill signal to process for job {job_id}.")
                return jsonify({'success': True, 'message': 'Cancellation signal sent.'})
            except Exception as e:
                logging.error(f"Error terminating process for job {job_id}: {e}")
                return jsonify({'error': str(e)}), 500
        else:
            jobs[job_id]['status'] = 'cancelled'
            jobs[job_id]['result'] = 'Job cancelled by user before processing started.'
            return jsonify({'success': True, 'message': 'Job flagged for cancellation before next task.'})
    else:
        return jsonify({'error': 'Job is not in a cancellable state.'}), 400

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'error': 'Job not found'}), 404
    
    job_for_frontend = {k: v for k, v in job.items() if k != 'process'}
    
    return jsonify(job_for_frontend)

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
    if not path or not os.path.exists(path):
        return []

    with open(path, mode='r', encoding='utf-8-sig') as csvfile:
        if file_type == 'transcript':
            reader = csv.reader(csvfile)
            try: 
                next(reader) 
            except StopIteration: 
                csvfile.seek(0)
            
            for row in reader:
                if not row: continue
                line = row[0]
                match = re.match(r'"?(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\s*-->\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?"?\s*(.*)', line)
                if match:
                    start_time, text = match.groups()
                    cleaned_text = text.strip().strip('"')
                    data.append({'start': start_time.strip(), 'text': cleaned_text})
        elif file_type == 'chat':
             reader = csv.DictReader(csvfile)
             for row in reader:
                 data.append(row)
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
    except FileNotFoundError:
        return jsonify({'error': f'File not found at path: {path}'}), 404
    except Exception as e:
        logging.error(f"Error reading CSV {path}: {e}")
        return jsonify({'error': str(e)}), 500

# --- REINTRODUCED FROM backend.py ---
@app.route('/api/generate_transcript', methods=['POST'])
def generate_transcript():
    if not openai_client:
        return jsonify({'error': "OpenAI client not initialized. Please check API key."}), 500
    
    data = request.json
    source_input = data.get('source_url')
    if not source_input:
        return jsonify({'error': 'Source URL is required.'}), 400

    temp_dir = os.path.join(UI_FOLDER, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.m4a")
    transcript_filename = f"generated_transcript_{uuid.uuid4()}.csv"
    transcript_path = os.path.join(temp_dir, transcript_filename)

    try:
        url, headers = _get_url_and_headers(source_input)
        
        logging.info("Downloading audio for transcription...")
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': temp_audio_path,
            'http_headers': headers,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logging.info(f"Audio downloaded to {temp_audio_path}")

        if not os.path.exists(temp_audio_path):
            raise FileNotFoundError("Failed to download audio file.")

        logging.info("Transcribing audio with Whisper API...")
        with open(temp_audio_path, "rb") as audio_file:
            transcription_response = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="verbose_json"
            )
        
        logging.info("Transcription complete. Formatting and saving to CSV.")
        
        with open(transcript_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['WEBVTT']) 
            for segment in transcription_response.segments:
                start = format_time_for_srt(segment['start'])
                end = format_time_for_srt(segment['end'])
                text = segment['text'].strip()
                formatted_line = f'{start} --> {end} {text}'
                writer.writerow([formatted_line])
        
        logging.info(f"Transcript saved to {transcript_path}")
        return jsonify({'filePath': transcript_path})

    except Exception as e:
        logging.error(f"Transcript generation failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.route('/api/generate_partial_transcript', methods=['POST'])
def generate_partial_transcript():
    if not openai_client:
        return jsonify({'error': "OpenAI client not initialized. Please check API key."}), 500

    data = request.json
    source_input = data.get('source_url')
    sales_path = data.get('sales_csv_path')

    if not source_input or not sales_path:
        return jsonify({'error': 'Source URL and Sales CSV path are required.'}), 400

    temp_dir = os.path.join(UI_FOLDER, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    transcript_filename = f"generated_partial_transcript_{uuid.uuid4()}.csv"
    transcript_path = os.path.join(temp_dir, transcript_filename)

    all_transcript_lines = []

    try:
        url, headers = _get_url_and_headers(source_input)
        sales_data = _read_csv_data(sales_path, 'sales')
        sold_items = [item for item in sales_data if item.get('status', '').upper() == 'SOLD']

        logging.info(f"Found {len(sold_items)} sold items. Generating transcript segments...")

        for i, item in enumerate(sold_items):
            item_time_sec = to_seconds(item.get('time'))
            if item_time_sec is None:
                continue

            start_sec = max(0, item_time_sec - 45)
            duration = 60
            temp_audio_path = os.path.join(temp_dir, f"segment_{uuid.uuid4()}.m4a")
            
            logging.info(f"Processing segment {i+1}/{len(sold_items)} for item '{item.get('product_name')}' at {format_time_for_srt(start_sec)}")

            ffmpeg_command = [
                'ffmpeg', '-y', '-ss', str(start_sec), '-t', str(duration),
                '-i', url, '-vn', '-c:a', 'aac', '-b:a', '128k', temp_audio_path
            ]
            
            header_str = "".join([f"{key}: {value}\r\n" for key, value in headers.items()])
            ffmpeg_command.insert(1, '-headers')
            ffmpeg_command.insert(2, header_str)

            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

            if not os.path.exists(temp_audio_path):
                 logging.warning(f"Failed to extract audio for item at {item.get('time')}")
                 continue

            with open(temp_audio_path, "rb") as audio_file:
                transcription_response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            for segment in transcription_response.segments:
                absolute_start = start_sec + segment['start']
                absolute_end = start_sec + segment['end']
                start_formatted = format_time_for_srt(absolute_start)
                end_formatted = format_time_for_srt(absolute_end)
                text = segment['text'].strip()
                all_transcript_lines.append(f'{start_formatted} --> {end_formatted} {text}')

            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
        all_transcript_lines.sort()

        with open(transcript_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['WEBVTT'])
            for line in all_transcript_lines:
                writer.writerow([line])

        logging.info(f"Partial transcript saved to {transcript_path}")
        return jsonify({'filePath': transcript_path})

    except Exception as e:
        logging.error(f"Partial transcript generation failed: {e}")
        if isinstance(e, subprocess.CalledProcessError):
             logging.error(f"FFMPEG Error: {e.stderr}")
             return jsonify({'error': f"FFMPEG Error: {e.stderr}"}), 500
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    if not openai_client:
        return jsonify({'analysis': "OpenAI client not initialized. Please check your API key."})
    
    data = request.json
    image_data_url = data.get('image')
    session_id = data.get('session_id') or str(uuid.uuid4())
    current_time = data.get('currentTime')
    sales_path = data.get('sales_csv_path')
    transcript_path = data.get('transcript_csv_path')

    if not all([image_data_url, current_time is not None, sales_path, transcript_path]):
        return jsonify({'error': 'Missing image, currentTime, or CSV path data.'}), 400
    
    try:
        sales_data = _read_csv_data(sales_path, 'sales')
        transcript_data = _read_csv_data(transcript_path, 'transcript')

        closest_item = None
        min_time_diff = float('inf')
        time_window = 60
        sold_items = [item for item in sales_data if item.get('status', '').upper() == 'SOLD']
        for item in sold_items:
            item_time_sec = to_seconds(item.get('time'))
            if item_time_sec is not None:
                diff = abs(current_time - item_time_sec)
                if diff < min_time_diff and diff <= time_window:
                    min_time_diff = diff
                    closest_item = item

        context_start = max(0, current_time - 15)
        context_end = current_time + 15
        relevant_transcript = " ".join([
            line['text'] for line in transcript_data
            if line.get('start') and context_start <= to_seconds(line.get('start')) <= context_end
        ])
        
        context_text = "Analyze this screenshot from the live stream based on the system prompt."
        if closest_item:
            context_text += f"\nContext: The product being discussed is likely: '{closest_item.get('product_name', 'N/A')}'."
        if relevant_transcript:
            context_text += f"\nThe surrounding conversation was: '{relevant_transcript}'."

        if session_id not in chat_histories:
            chat_histories[session_id] = [{"role": "system", "content": FRAME_ANALYSIS_PROMPT}]

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                *chat_histories[session_id],
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            max_tokens=500,
        )
        analysis_text = response.choices[0].message.content
        chat_histories[session_id].append({"role": "user", "content": context_text})
        chat_histories[session_id].append({"role": "assistant", "content": analysis_text})
        return jsonify({'analysis': analysis_text, 'session_id': session_id})
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def ai_chat():
    if not openai_client:
        return jsonify({'response': "OpenAI client not initialized. Please check your API key."})
        
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    if not session_id or session_id not in chat_histories:
         return jsonify({'error': 'No active analysis session found. Please analyze a frame first.'}), 400

    chat_histories[session_id].append({"role": "user", "content": user_message})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=chat_histories[session_id],
            max_tokens=300
        )
        ai_response = response.choices[0].message.content
        chat_histories[session_id].append({"role": "assistant", "content": ai_response})
        return jsonify({'response': ai_response})
    except Exception as e:
        logging.error(f"OpenAI API error during chat: {e}")
        return jsonify({'error': str(e)}), 500
# --- END REINTRODUCED SECTION ---


def _find_moments_logic(final_user_prompt, sales_path, transcript_path, chat_path, num_clips_to_find):
    global gemini_model
    if not gemini_model:
        raise Exception("Gemini client not initialized. Please set API key.")

    transcript_data = _read_csv_data(transcript_path, 'transcript')
    sales_data = _read_csv_data(sales_path, 'sales')
    chat_data_raw = _read_csv_data(chat_path, 'chat')

    context_data = {
        "transcript": transcript_data,
        "sales": [item for item in sales_data if item.get('status', '').upper() == 'SOLD'],
        "chat": []
    }

    for row in chat_data_raw:
        time_sec = parse_livestream_time(row.get('Minutes into livestream'))
        if time_sec is not None and time_sec >= 0:
            context_data["chat"].append({
                'time_sec': time_sec,
                'username': row.get('Username'),
                'message': row.get('Message')
            })

    system_prompt = f"""
    You are a viral clip finder AI for a live stream clipping tool. Your task is to analyze the provided stream data (transcript, sales, and chat logs) to find up to {num_clips_to_find} engaging moments based on the user's request.

    USER REQUEST: "{final_user_prompt}"

    Analyze the user's request to determine their intent. Are they looking for exciting product sales, funny chat interactions, specific quotes, or general hype moments?

    Based on the intent, search through the provided JSON data which contains three keys: 'transcript', 'sales', and 'chat'.

    1.  If the user is asking for product/sales moments (e.g., "top sales", "best deals", "rare item"):
        -   Focus on the `sales` data.
        -   Cross-reference the sale `time` with the `transcript` and `chat` to find moments of hype.
        -   For each moment found, return an object representing the sold item from the `sales` list. You MUST add a new key, `clipName`, with a new, descriptive name for the clip based on the transcript and product details.

    2.  If the user is asking for non-product moments (e.g., "funniest moment", "chat interaction", "shout out", "specific quote"):
        -   Focus on the `transcript` and `chat` data. Ignore the `sales` data.
        -   For each moment, return a JSON object with three keys: `startTime` (in seconds), `endTime` (in seconds), and a short, catchy `clipName` (4-5 words max). The clip duration (endTime - startTime) should be between 15 and 45 seconds.

    Your final output must be a single JSON object with a key "results" which contains a list of your findings. It is okay to return fewer than {num_clips_to_find} if you can't find enough high-quality moments.
    """
    
    response = gemini_model.generate_content(
        [system_prompt, json.dumps(context_data, indent=0)],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
    )
    
    result = json.loads(response.text)
    return result.get('results', [])

@app.route('/api/ai_find_moments', methods=['POST'])
def ai_find_moments():
    global default_ai_prompt
    try:
        data = request.json
        custom_prompt = data.get('prompt')
        sales_path = data.get('sales_csv_path')
        transcript_path = data.get('transcript_csv_path')
        chat_path = data.get('chat_csv_path')

        final_user_prompt = custom_prompt or default_ai_prompt

        if not transcript_path:
            return jsonify({'error': 'A transcript CSV is required for all AI searches.'}), 400

        if custom_prompt:
            num_clips_to_find = 5 
            match = re.search(r'\b(\d+)\b', final_user_prompt)
            if match:
                requested_clips = int(match.group(1))
                num_clips_to_find = min(requested_clips, 10) 
        else:
            num_clips_to_find = 7
        
        found_items = _find_moments_logic(final_user_prompt, sales_path, transcript_path, chat_path, num_clips_to_find)
        return jsonify(found_items)

    except Exception as e:
        logging.error(f"AI Find Moments Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# --- REINTRODUCED FROM backend.py ---
@app.route('/api/find_and_clip', methods=['POST'])
def find_and_clip():
    global default_ai_prompt
    try:
        params = request.json
        sales_path = params.get('sales_csv_path')
        transcript_path = params.get('transcript_csv_path')
        chat_path = params.get('chat_csv_path')

        if not all([sales_path, transcript_path]):
            return jsonify({'error': 'Sales and transcript CSV paths are required for auto-clipping.'}), 400

        logging.info("Auto-finding top 15 clips for empty queue.")
        found_products = _find_moments_logic(default_ai_prompt, sales_path, transcript_path, chat_path, 15)

        if not found_products:
            return jsonify({'error': 'AI could not find any relevant clips to create.'}), 404

        selected_products = []
        for product in found_products:
            if 'product_id' in product:
                sold_time_sec = to_seconds(product.get('time'))
                if sold_time_sec is not None:
                    selected_products.append({
                        'name': product.get('product_name'),
                        'start': max(0, sold_time_sec - 45),
                        'end': sold_time_sec + 15,
                        'sold_time': product.get('time'),
                        'sold_price': product.get('sold_price_amount')
                    })
        
        if not selected_products:
             return jsonify({'error': 'AI found moments, but they were not product-based for auto-clipping.'}), 404

        params['selected_products'] = selected_products
        
        job_id = str(uuid.uuid4())
        source_input = params.get('source_url', '')
        url, headers = _get_url_and_headers(source_input)
        params['resolved_url'] = url
        params['resolved_headers'] = headers

        jobs[job_id] = {'status': 'pending', 'params': params, 'progress': 'Job queued'}
        thread = threading.Thread(target=run_clipping_process, args=(job_id, params, jobs))
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Auto-clip job started.', 'job_id': job_id}), 202

    except Exception as e:
        logging.error(f"Find and Clip Error: {e}")
        return jsonify({'error': str(e)}), 500
# --- END REINTRODUCED SECTION ---