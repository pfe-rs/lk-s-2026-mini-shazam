import subprocess, time, os, glob, re, argparse, sys, json, hashlib, urllib.request, zipfile
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

from mutagen.id3 import ID3, TIT2, TPE1, TCON, TALB, TXXX, TCOP

DATASET_DIR = "dataset"
SELECTED_CSV = "selected_tracks.csv"
DOWNLOADED_LOG = "downloaded.txt"
FAILED_LOG = "failed.txt"
MANIFEST_FILE = "manifest.json"
FMA_TRACK_FILES_JSON = "fma_track_files.json"
FMA_ZIP_BASE = "https://os.unil.cloud.switch.ch/fma"
FMA_SUBSETS = {"small": "fma_small.zip", "medium": "fma_medium.zip", "full": "fma_full.zip"}

class RangeIO:
    def __init__(self, url):
        self.url = url
        self._pos = 0
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            self._size = int(resp.headers['Content-Length'])
    def seekable(self):
        return True
    def read(self, n=-1):
        if n == -1:
            n = self._size - self._pos
        if n <= 0:
            return b''
        end = self._pos + n - 1
        req = urllib.request.Request(self.url, method='GET')
        req.add_header('Range', f'bytes={self._pos}-{end}')
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self._pos += len(data)
        return data
    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        return self._pos
    def tell(self):
        return self._pos

log_lock = Lock()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.split(",")[0].strip() for line in f if line.strip()}

def append_id(path, track_id, title):
    with log_lock, open(path, "a") as f:
        f.write(f"{track_id},{title}\n")

def remove_id(path, track_id):
    with log_lock:
        if not os.path.exists(path):
            return
        with open(path) as f:
            lines = f.readlines()
        with open(path, "w") as f:
            for line in lines:
                if not line.startswith(track_id + ","):
                    f.write(line)

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def find_mp3(track_id, genre_dir):
    for f in glob.glob(os.path.join(genre_dir, f"{track_id}_*")):
        if f.endswith('.mp3'):
            return f
    return None

def embed_tags(mp3_path, row):
    try:
        audio = ID3(mp3_path)
    except Exception:
        from mutagen.id3 import ID3NoHeaderError
        audio = ID3()
    audio.add(TIT2(encoding=3, text=str(row['title'])))
    audio.add(TPE1(encoding=3, text=str(row['artist'])))
    audio.add(TCON(encoding=3, text=str(row['genre'])))
    audio.add(TALB(encoding=3, text=str(row['album'])))
    audio.add(TXXX(encoding=3, desc="fma_track_id", text=str(row['track_id'])))
    audio.add(TCOP(encoding=3, text=str(row['license'])))
    audio.save(mp3_path)

def sanitize(s):
    return re.sub(r'[/\\:;"<>|&]', ' ', s).strip()

def convert_to_mp3(src, track_id, genre_dir):
    dst = os.path.splitext(src)[0] + ".mp3"
    if src == dst:
        return src
    result = subprocess.run(["ffmpeg", "-y", "-i", src, "-q:a", "0", dst],
                            capture_output=True, text=True, timeout=120)
    if result.returncode == 0 and os.path.exists(dst):
        os.remove(src)
        return dst
    return None

def find_or_convert_audio(prefix):
    for ext in ["*.mp3", "*.m4a", "*.webm", "*.opus"]:
        for f in glob.glob(prefix + ext):
            if f.endswith(".mp3"):
                return f
            converted = convert_to_mp3(f, os.path.basename(prefix).rstrip("_"), os.path.dirname(prefix))
            if converted:
                return converted
    return None

SEARCH_FORMATS = [
    lambda a, t: f"{a} {t} full song",
    lambda a, t: f"{a} - {t}",
    lambda a, t: f"{t} {a}",
    lambda a, t: f'"{t}" {a} audio',
    lambda a, t: f"{a} {t} music",
    lambda a, t: f"{t} {a} full",
]

def record_fma_metadata(manifest, track_id, fma_files):
    if track_id in fma_files and track_id not in manifest:
        manifest[track_id] = {
            "source_url": fma_files[track_id]["track_url"],
            "fma_file": fma_files[track_id]["track_file"],
            "status": "not_downloaded",
        }

def download_one(row, manifest, fma_files, subset_map):
    track_id = str(row['track_id'])
    title = sanitize(str(row['title']))
    artist = sanitize(str(row['artist']))
    genre = str(row['genre']).replace('/', '_')
    genre_dir = os.path.join(DATASET_DIR, genre)
    ensure_dir(genre_dir)

    output_template = os.path.join(genre_dir, track_id + "_%(uploader)s_%(title)s_%(id)s.%(ext)s")
    prefix = os.path.join(genre_dir, f"{track_id}_")

    for f in glob.glob(prefix + "*"):
        os.remove(f)

    expected_duration = float(row['duration'])
    max_duration = expected_duration * 2.5
    min_duration = expected_duration * 0.3

    existing = manifest.get(track_id)
    if existing and existing.get("source_url") and "youtube.com" in existing["source_url"]:
        sources = [existing["source_url"]]
    else:
        sources = [f"ytsearch5:{fmt(artist, title)}" for fmt in SEARCH_FORMATS]

    for source in sources:
        for attempt in range(2):
            duration_filter = f"duration > {min_duration:.0f} & duration < {max_duration:.0f}"
            cmd = [
                "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "0",
                "--add-metadata", "--embed-thumbnail",
                "--match-filter", duration_filter,
                "--retries", "5", "--extractor-retries", "5", "--fragment-retries", "5",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "--output", output_template,
                "--print", "after_download:webpage_url",
                source
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"  [{track_id}] attempt {attempt+1} timed out")
                continue

            if result.returncode == 0:
                mp3 = find_or_convert_audio(prefix)
                if mp3:
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", mp3],
                        capture_output=True, text=True, timeout=15
                    )
                    actual_dur = float(json.loads(probe.stdout)['format']['duration'])
                    if actual_dur > max_duration or actual_dur < min_duration:
                        print(f"  [{track_id}] bad duration {actual_dur:.0f}s (expected {expected_duration:.0f}s), skipping")
                        os.remove(mp3)
                        continue

                    embed_tags(mp3, row)
                    append_id(DOWNLOADED_LOG, track_id, title)

                    url = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
                    entry = {
                        "filename": os.path.basename(mp3),
                        "md5": file_md5(mp3),
                        "title": title,
                        "artist": artist
                    }
                    if url and not url.startswith("ytsearch"):
                        entry["source_url"] = url
                    if track_id in fma_files:
                        entry["fma_file"] = fma_files[track_id]["track_file"]
                        entry.setdefault("source_url", fma_files[track_id]["track_url"])
                    manifest[track_id] = entry
                    return True
                else:
                    leftovers = [f for f in os.listdir(genre_dir) if f.startswith(track_id + "_")] if os.path.exists(genre_dir) else []
                    if leftovers:
                        print(f"  [{track_id}] got {leftovers} but couldn't convert to mp3")
                    else:
                        print(f"  [{track_id}] no file downloaded from: {source[:120]}")
            else:
                err = result.stderr[:300].replace('\n', ' ')
                if "Sign in" in err or "age" in err.lower() or "private" in err.lower() or "removed" in err.lower():
                    print(f"  [{track_id}] unavailable: {err[:100]}")
                    break
                print(f"  [{track_id}] {source[:80]} attempt {attempt+1}: {err[:120]}")

    print(f"  [{track_id}] trying FMA official dataset (subset={subset_map.get(track_id, '?')})...")
    dst = download_from_fma_zip(track_id, genre_dir, subset_map, fma_files)
    if dst:
        embed_tags(dst, row)
        append_id(DOWNLOADED_LOG, track_id, title)
        entry = {
            "filename": os.path.basename(dst),
            "md5": file_md5(dst),
            "title": title,
            "artist": artist,
            "fma_file": fma_files.get(track_id, {}).get("track_file", ""),
            "source_url": fma_files.get(track_id, {}).get("track_url", ""),
        }
        manifest[track_id] = entry
        return True
    else:
        print(f"  [{track_id}] FMA dataset also failed")
        record_fma_metadata(manifest, track_id, fma_files)

    append_id(FAILED_LOG, track_id, title)
    return False

def load_fma_track_files():
    mapping = {}
    if os.path.exists(FMA_TRACK_FILES_JSON):
        with open(FMA_TRACK_FILES_JSON) as f:
            for line in f:
                entry = json.loads(line)
                mapping[str(entry["track_id"])] = {
                    "track_file": str(entry["track_file"]),
                    "track_url": str(entry["track_url"]),
                }
    return mapping

def load_fma_subset_map():
    mapping = {}
    tracks = pd.read_csv("fma/fma_metadata/tracks.csv", header=[0, 1], low_memory=False)
    tracks = tracks.iloc[2:].copy()
    tracks.columns = tracks.columns.map("_".join).str.strip("_")
    tracks = tracks.rename(columns={tracks.columns[0]: "track_id"})
    for _, row in tracks.iterrows():
        tid = str(int(row["track_id"]))
        subset = str(row["set_subset"]).strip()
        if subset in ("small", "medium", "full", "large"):
            mapping[tid] = "full" if subset == "large" else subset
    return mapping

def download_from_fma_zip(track_id, genre_dir, subset_map, fma_files):
    subset = subset_map.get(track_id)
    if not subset:
        return None
    zip_url = f"{FMA_ZIP_BASE}/{FMA_SUBSETS[subset]}"
    fma = fma_files.get(track_id)
    if not fma:
        return None

    tid_int = int(track_id)
    arcname = f"fma_{subset}/{tid_int // 1000:03d}/{tid_int:06d}.mp3"
    dst = os.path.join(genre_dir, f"{track_id}_{arcname.replace('/', '_')}")
    if os.path.exists(dst):
        return dst

    try:
        rio = RangeIO(zip_url)
        z = zipfile.ZipFile(rio)
        z.extract(arcname, genre_dir)
        extracted = os.path.join(genre_dir, arcname)
        if os.path.exists(extracted):
            os.renames(extracted, dst)
        if os.path.exists(dst):
            return dst
    except Exception as e:
        print(f"  [{track_id}] FMA zip download failed: {e}")
    return None

def get_dataset_size():
    total = 0
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.endswith('.mp3'):
                total += os.path.getsize(os.path.join(root, f))
    return total

def main():
    parser = argparse.ArgumentParser(description="Download FMA tracks from YouTube")
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Number of parallel download workers (default: 1)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry tracks previously marked as failed")
    args = parser.parse_args()

    df = pd.read_csv(SELECTED_CSV)
    downloaded_ids = load_ids(DOWNLOADED_LOG)
    failed_ids = load_ids(FAILED_LOG)
    manifest = load_manifest()
    fma_files = load_fma_track_files()
    subset_map = load_fma_subset_map()

    for tid in list(downloaded_ids):
        row = df[df['track_id'].astype(str) == tid]
        if len(row) == 0:
            continue
        genre = str(row.iloc[0]['genre']).replace('/', '_')
        genre_dir = os.path.join(DATASET_DIR, genre)
        mp3 = find_mp3(tid, genre_dir)
        if not mp3:
            print(f"  [{tid}] in log but file missing, will re-download")
            downloaded_ids.discard(tid)
        elif tid not in manifest:
            manifest[tid] = {
                "filename": os.path.basename(mp3),
                "md5": file_md5(mp3),
                "title": sanitize(str(row.iloc[0]['title'])),
                "artist": sanitize(str(row.iloc[0]['artist']))
            }
            if tid in fma_files:
                manifest[tid]["fma_file"] = fma_files[tid]["track_file"]
                manifest[tid]["source_url"] = fma_files[tid]["track_url"]

    for tid in failed_ids:
        record_fma_metadata(manifest, tid, fma_files)

    if args.retry_failed:
        remaining = df[~df['track_id'].astype(str).isin(downloaded_ids)]
        print(f"Retrying {len(remaining)} tracks not yet downloaded (includes previously failed)")
    else:
        remaining = df[~df['track_id'].astype(str).isin(downloaded_ids | failed_ids)]
        print(f"Skipping {len(failed_ids)} previously failed tracks (use --retry-failed to retry)")

    print(f"Total: {len(df)} | Already done: {len(downloaded_ids)} | Failed: {len(failed_ids)} | Remaining: {len(remaining)} | Workers: {args.workers}")

    if remaining.empty:
        save_manifest(manifest)
        print("Nothing to do.")
        return
    rows = [row for _, row in remaining.iterrows()]
    done_count = len(downloaded_ids)
    fail_count = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, row, manifest, fma_files, subset_map): row for row in rows}
        for fut in as_completed(futures):
            completed += 1
            row = futures[fut]
            ok = fut.result()
            if ok:
                done_count += 1
                remove_id(FAILED_LOG, str(row['track_id']))
            else:
                fail_count += 1
            print(f"  [{completed}/{len(rows)}] {row['track_id']}: {'OK' if ok else 'FAIL'}")
            time.sleep(2)

    save_manifest(manifest)
    fail_count = len(load_ids(FAILED_LOG))

    print(f"\nDone: {done_count} succeeded, {fail_count} failed")
    print(f"Dataset size: {get_dataset_size() / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
