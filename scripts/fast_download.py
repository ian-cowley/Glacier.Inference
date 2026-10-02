import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_part(url, start_byte, end_byte, part_filename, part_idx, progress_tracker):
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Glacier/1.2.11',
        'Range': f'bytes={start_byte}-{end_byte}'
    })
    
    # Check if part already completed
    expected_len = end_byte - start_byte + 1
    if os.path.exists(part_filename):
        existing_size = os.path.getsize(part_filename)
        if existing_size == expected_len:
            progress_tracker[part_idx] = expected_len
            return part_filename

    with urllib.request.urlopen(req, timeout=30) as resp, open(part_filename, 'wb') as f:
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB buffer
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            progress_tracker[part_idx] = downloaded
            
    return part_filename

def main():
    if len(sys.argv) < 3:
        print("Usage: python fast_download.py <url> <output_file> [num_threads]")
        sys.exit(1)

    url = sys.argv[1]
    output_file = sys.argv[2]
    num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    print(f"Resolving redirect and headers for: {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Glacier/1.2.11'})
    with urllib.request.urlopen(req) as resp:
        final_url = resp.geturl()
        total_size = int(resp.headers.get('Content-Length', 0))

    total_gb = total_size / (1024 ** 3)
    print(f"Direct CDN URL resolved. Total size: {total_size:,} bytes ({total_gb:.2f} GB)")
    print(f"Downloading with {num_threads} parallel threads...")

    part_size = total_size // num_threads
    parts = []
    for i in range(num_threads):
        start = i * part_size
        end = (start + part_size - 1) if i < num_threads - 1 else (total_size - 1)
        part_file = f"{output_file}.part{i}"
        parts.append((i, start, end, part_file))

    progress = [0] * num_threads
    start_time = time.time()
    last_print = start_time

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(download_part, final_url, start, end, part_file, i, progress): i
            for i, start, end, part_file in parts
        }

        while not all(f.done() for f in futures):
            time.sleep(1.0)
            now = time.time()
            if now - last_print >= 2.0:
                downloaded = sum(progress)
                elapsed = now - start_time
                speed = downloaded / elapsed / (1024 * 1024) if elapsed > 0 else 0
                pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                eta = (total_size - downloaded) / (downloaded / elapsed) if downloaded > 0 and elapsed > 0 else 0
                print(f"\rProgress: {pct:5.1f}% | {mb_down:6.1f} / {mb_total:6.1f} MB | Speed: {speed:5.1f} MB/s | ETA: {eta:3.0f}s", end="", flush=True)
                last_print = now

        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - start_time
    avg_speed = total_size / elapsed / (1024 * 1024)
    print(f"\nAll {num_threads} parts downloaded in {elapsed:.1f}s ({avg_speed:.1f} MB/s). Assembling file...")

    # Concatenate parts
    with open(output_file, 'wb') as outfile:
        for _, _, _, part_file in parts:
            with open(part_file, 'rb') as pf:
                while True:
                    buf = pf.read(16 * 1024 * 1024)  # 16MB copy buffer
                    if not buf:
                        break
                    outfile.write(buf)
            try:
                os.remove(part_file)
            except Exception:
                pass

    final_size = os.path.getsize(output_file)
    print(f"Assembly complete! File saved: {output_file} ({final_size:,} bytes)")
    if final_size == total_size:
        print("Verification: SUCCESS (Exact byte count match).")
    else:
        print(f"Verification: WARNING (Size mismatch: {final_size} vs {total_size})")

if __name__ == '__main__':
    main()
