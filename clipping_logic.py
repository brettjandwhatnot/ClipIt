import sys
import os
import subprocess
import re
import json
import csv
import shlex
import threading

def get_ffmpeg_path():
    if getattr(sys, 'frozen', False):
        if sys.platform == 'win32':
            ffmpeg_exe = 'ffmpeg.exe'
        else:
            ffmpeg_exe = 'ffmpeg'
        return os.path.join(sys._MEIPASS, ffmpeg_exe)
    else:
        return 'ffmpeg'

def to_seconds(time_str):
    if not time_str: return None
    try:
        s = str(time_str).strip().replace(',', '.')
        parts = s.split(':')
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except (ValueError, IndexError, TypeError): return None

def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', str(name))

def parse_curl_command(source_input):
    cleaned_string = re.sub(r'\\\s*\n?', ' ', source_input.strip())
    parts = shlex.split(cleaned_string)
    url = next((arg for arg in parts[1:] if arg.startswith('http')), None)
    if not url:
        raise ValueError("Could not extract URL from source cURL command.")
    
    headers = {}
    i = 0
    while i < len(parts):
        if parts[i] in ('-H', '--header'):
            key, value = parts[i + 1].split(':', 1)
            headers[key.strip()] = value.strip()
            i += 2
        else:
            i += 1
    return url, headers

def run_single_ffmpeg_clip(clip_request, url, headers, output_folder):
    start_sec = clip_request['start']
    duration = clip_request['end'] - start_sec
    if duration <= 0:
        return (False, f"SKIPPED: {clip_request['name']} has invalid duration.", None, None)

    safe_name = sanitize_filename(clip_request['name'])
    
    base_filename = safe_name
    output_filename = os.path.join(output_folder, f"{base_filename}.mp4")
    counter = 1
    while os.path.exists(output_filename):
        new_name = f"{base_filename} ({counter})"
        output_filename = os.path.join(output_folder, f"{new_name}.mp4")
        counter += 1
    
    thumbnail_filename = output_filename.replace('.mp4', '.jpg')
    thumbnail_time = start_sec + (duration / 2)

    ffmpeg_path = get_ffmpeg_path()

    command = [ffmpeg_path, '-y']
    if headers:
        header_str = "".join([f"{key}: {value}\r\n" for key, value in headers.items()])
        command.extend(['-headers', header_str])
    
    command.extend(['-ss', str(start_sec), '-i', url, '-t', str(duration), '-c', 'copy', '-movflags', '+faststart', output_filename])
    
    thumb_command = [ffmpeg_path, '-y']
    if headers:
        thumb_command.extend(['-headers', header_str])
    thumb_command.extend(['-ss', str(thumbnail_time), '-i', url, '-vframes', '1', '-q:v', '2', thumbnail_filename])

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
        final_filename_for_log = os.path.basename(output_filename)
        if result.returncode != 0:
            return (False, f"FAILED: {final_filename_for_log}. Error: {result.stderr[-500:]}", None, None)
        
        thumb_result = subprocess.run(thumb_command, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
        if thumb_result.returncode != 0:
            print(f"Thumbnail generation failed for {final_filename_for_log}: {thumb_result.stderr[-500:]}")

        return (True, f"SUCCESS: {final_filename_for_log}", output_filename, thumbnail_filename)
    except Exception as e:
        final_filename_for_log = os.path.basename(output_filename)
        return (False, f"CRITICAL FAIL: {final_filename_for_log}. Error: {e}", None, None)

def build_clip_jobs(params):
    all_jobs = []
    selected_clips = params.get('selected_products', [])
    if not selected_clips: 
        raise ValueError("No clips were provided for processing.")

    for i, clip_data in enumerate(selected_clips):
        start_sec = clip_data.get('start')
        end_sec = clip_data.get('end')
        
        if start_sec is not None and end_sec is not None and end_sec > start_sec:
            base_name = sanitize_filename(clip_data.get('name', f'clip_{i}'))
            sold_time = clip_data.get('sold_time')
            sold_price_cents = clip_data.get('sold_price')

            if sold_time and sold_price_cents:
                sold_time_formatted = sanitize_filename(sold_time).split('.')[0].replace(':', '-')
                try:
                    sold_price_dollars = int(sold_price_cents) / 100
                    sold_price_formatted = f"${sold_price_dollars:.0f}"
                except (ValueError, TypeError):
                    sold_price_formatted = "$0"
                final_name = f"{base_name}_{sold_time_formatted}_{sold_price_formatted}"
            else:
                final_name = base_name
            
            all_jobs.append({'start': start_sec, 'end': end_sec, 'name': final_name})
            
    return all_jobs

def run_clipping_process(job_id, params, jobs):
    try:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['progress'] = 'Building clip list...'
        jobs[job_id]['job_id'] = job_id

        clip_requests = build_clip_jobs(params)
        if not clip_requests:
            raise ValueError("No valid clipping tasks were generated from the input.")

        source_url, headers = parse_curl_command(params.get('source_url', ''))
        output_folder = params.get('output_folder')
        if not output_folder or not os.path.isdir(output_folder):
            raise ValueError("A valid output folder must be selected.")
        
        jobs[job_id]['output_folder'] = output_folder

        completed_count = 0
        total_clips = len(clip_requests)
        jobs[job_id]['progress'] = f'Starting to clip {total_clips} files...'
        
        completed_clip_paths = []

        for i, request in enumerate(clip_requests):
            jobs[job_id]['progress'] = f"Clipping {i+1}/{total_clips}: {request['name']}"
            was_successful, result_message, clip_path, thumb_path = run_single_ffmpeg_clip(request, source_url, headers, output_folder)
            
            if was_successful:
                completed_count += 1
                completed_clip_paths.append({
                    'video': clip_path, 
                    'thumbnail': thumb_path,
                    'start': request['start'],
                    'end': request['end']
                })
            
            print(f"Job {job_id}: {result_message}")
            
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = f"Clipping complete. Successfully created {completed_count}/{total_clips} clips."
        jobs[job_id]['completed_clips'] = completed_clip_paths

    except Exception as e:
        error_message = f"Failed: {e}"
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = error_message
        print(f"Job {job_id} FAILED: {error_message}")
