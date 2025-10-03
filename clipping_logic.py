import sys
import os
import subprocess
import re
import shlex
import logging

# --- Helper Functions ---

def get_ffmpeg_path():
    """Determines the correct path for the ffmpeg executable."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        return os.path.join(base_path, 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg')
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

# --- FFMPEG Execution ---

def run_single_ffmpeg_clip(clip_request, url, headers, output_folder, job_id, jobs):
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

    command = [ffmpeg_path, '-y', '-progress', 'pipe:1']
    if headers:
        header_str = "".join([f"{key}: {value}\r\n" for key, value in headers.items()])
        command.extend(['-headers', header_str])
    
    command.extend(['-protocol_whitelist', 'file,http,https,tcp,tls,crypto', '-ss', str(start_sec), '-i', url, '-t', str(duration), '-c', 'copy', '-movflags', '+faststart', output_filename])
    
    thumb_command = [ffmpeg_path, '-y']
    if headers:
        thumb_command.extend(['-headers', header_str])
    thumb_command.extend(['-protocol_whitelist', 'file,http,https,tcp,tls,crypto', '-ss', str(thumbnail_time), '-i', url, '-vframes', '1', '-q:v', '2', thumbnail_filename])

    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8', errors='ignore')
        jobs[job_id]['process'] = process

        # This loop now only reads progress and will be interrupted by process.kill()
        for line in process.stdout:
            if 'time=' in line:
                match = re.search(r'time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})', line)
                if match:
                    hours, mins, secs, _ = map(int, match.groups())
                    elapsed_secs = hours * 3600 + mins * 60 + secs
                    percent = min(100, (elapsed_secs / duration) * 100) if duration > 0 else 100
                    jobs[job_id]['progress'] = f"Processing {safe_name}: {percent:.1f}%"
        process.wait()

        final_filename_for_log = os.path.basename(output_filename)
        
        # Check status after the process has finished or been killed
        if jobs[job_id].get('status') == 'cancelling':
            # Clean up the partial file
            if os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                    logging.info(f"Removed partial file on cancel: {output_filename}")
                except OSError as e:
                    logging.error(f"Error removing partial file: {e}")
            return(False, f"CANCELLED: {final_filename_for_log}", None, None)

        if process.returncode != 0:
            return (False, f"FAILED: {final_filename_for_log}. FFMPEG exited with code {process.returncode}. Check logs.", None, None)
        
        thumb_result = subprocess.run(thumb_command, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
        if thumb_result.returncode != 0:
            print(f"Thumbnail generation failed for {final_filename_for_log}: {thumb_result.stderr[-500:]}")

        return (True, f"SUCCESS: {final_filename_for_log}", output_filename, thumbnail_filename)
    except Exception as e:
        final_filename_for_log = os.path.basename(output_filename)
        # Check if the job was cancelled, which can cause the exception
        if jobs[job_id].get('status') == 'cancelling':
            return(False, f"CANCELLED: {final_filename_for_log}", None, None)
        return (False, f"CRITICAL FAIL: {final_filename_for_log}. Error: {e}", None, None)


# --- Job Definition Logic ---

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

# --- Main Process Function (Called by backend.py) ---

def run_clipping_process(job_id, params, jobs):
    try:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['progress'] = 'Building clip list...'
        jobs[job_id]['job_id'] = job_id 

        clip_requests = build_clip_jobs(params)
        if not clip_requests:
            raise ValueError("No valid clipping tasks were generated from the input.")

        source_url = params['resolved_url']
        headers = params['resolved_headers']
        output_folder = params.get('output_folder')
        if not output_folder or not os.path.isdir(output_folder):
            raise ValueError("A valid output folder must be selected.")
        
        jobs[job_id]['output_folder'] = output_folder

        completed_count = 0
        total_clips = len(clip_requests)
        jobs[job_id]['progress'] = f'Starting to clip {total_clips} files...'
        
        completed_clip_paths = []

        for i, request in enumerate(clip_requests):
            if jobs[job_id].get('status') == 'cancelling':
                jobs[job_id]['status'] = 'cancelled'
                jobs[job_id]['result'] = f"Job cancelled by user. {completed_count}/{total_clips} clips created."
                break

            jobs[job_id]['progress'] = f"Queueing {i+1}/{total_clips}: {request['name']}"
            was_successful, result_message, clip_path, thumb_path = run_single_ffmpeg_clip(request, source_url, headers, output_folder, job_id, jobs)
            
            if was_successful:
                completed_count += 1
                completed_clip_paths.append({
                    'video': clip_path, 
                    'thumbnail': thumb_path,
                    'start_time_sec': request['start']  # Add the start time to the completed data
                })
            
            print(f"Job {job_id}: {result_message}")

        if jobs[job_id].get('status') != 'cancelled':
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = f"Clipping complete. Successfully created {completed_count}/{total_clips} clips."
            jobs[job_id]['completed_clips'] = completed_clip_paths

    except Exception as e:
        error_message = f"Failed: {e}"
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = error_message
        print(f"Job {job_id} FAILED: {error_message}")
    finally:
        # This cleanup now runs once at the very end of the job.
        if 'process' in jobs[job_id]:
            del jobs[job_id]['process']
        logging.info(f"Job {job_id} finished with status: {jobs[job_id].get('status')}")

