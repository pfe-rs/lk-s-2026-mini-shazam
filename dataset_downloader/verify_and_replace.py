import json, hashlib, os, sys, time, shutil, subprocess, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
MANIFEST = ROOT / "manifest.json"
PROGRESS = ROOT / "progress"
PROGRESS.mkdir(exist_ok=True)

LOG = PROGRESS / "log.txt"
STATS = PROGRESS / "stats.txt"
CURRENT = PROGRESS / "current.txt"
FAILED = PROGRESS / "failed.txt"

log_fh = open(LOG, "w")
start_time = time.time()
stats = {"ok": 0, "replaced": 0, "failed": 0, "skipped": 0, "processed": 0}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=log_fh, flush=True)


def write_stats():
    elapsed = time.time() - start_time
    rate = stats["processed"] / elapsed * 60 if elapsed > 0 else 0
    with open(STATS, "w") as f:
        f.write(f"Processed: {stats['processed']}\n")
        f.write(f"OK:        {stats['ok']}\n")
        f.write(f"Replaced:  {stats['replaced']}\n")
        f.write(f"Failed:    {stats['failed']}\n")
        f.write(f"Skipped:   {stats['skipped']}\n")
        f.write(f"Rate:      {rate:.1f}/min\n")
        f.write(f"Elapsed:   {elapsed:.0f}s\n")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def download_url(url, dest, timeout=30):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        resp = urllib.request.urlopen(req, timeout=timeout)
        with open(dest, "wb") as f:
            while chunk := resp.read(8192):
                f.write(chunk)
        return True
    except Exception:
        return False


def download_youtube(url, dest):
    try:
        cmd = [
            "yt-dlp", "-x", "--audio-format", "mp3",
            "--audio-quality", "0", "--no-playlist",
            "-o", dest, url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if os.path.exists(dest):
            return True
        for ext in [".mp3", ".webm", ".m4a"]:
            if os.path.exists(dest + ext):
                os.rename(dest + ext, dest)
                return True
        return False
    except Exception:
        return False


def main():
    manifest = json.load(open(MANIFEST))
    total = len(manifest)
    failed_ids = []

    log(f"Starting: {total} tracks")

    for i, (tid, entry) in enumerate(manifest.items(), 1):
        title = entry.get("title", "?")
        url = entry.get("source_url", "")
        expected = entry.get("md5", "")
        genre = entry.get("genre", "Unknown").replace("/", "_")
        filename = entry.get("filename", "")

        with open(CURRENT, "w") as f:
            f.write(f"{i}/{total} — {tid}: {title}\n")

        if not url:
            log(f"SKIP {tid}: no URL")
            stats["processed"] += 1
            write_stats()
            continue

        existing_path = DATASET / filename if filename else None
        existing_size = 0
        if existing_path and existing_path.exists():
            existing_size = existing_path.stat().st_size

        tmp = str(PROGRESS / f"dl_{tid}.mp3")

        is_youtube = "youtube.com" in url or "youtu.be" in url
        if is_youtube:
            ok = download_youtube(url, tmp)
        else:
            ok = download_url(url, tmp)

        if not ok or not os.path.exists(tmp) or os.path.getsize(tmp) < 1000:
            log(f"FAIL {tid}: {title}")
            failed_ids.append(tid)
            stats["failed"] += 1
            if os.path.exists(tmp):
                os.remove(tmp)
            stats["processed"] += 1
            write_stats()
            continue

        dl_size = os.path.getsize(tmp)
        got = md5(tmp)

        if got == expected:
            log(f"OK {tid}: {title}")
            os.remove(tmp)
            stats["ok"] += 1
        elif existing_size > 0 and dl_size < existing_size * 0.5:
            log(f"SKIP {tid}: {title} dl={dl_size} existing={existing_size} (download too small)")
            os.remove(tmp)
            stats["skipped"] += 1
        else:
            if existing_path and existing_path.exists():
                os.remove(str(existing_path))
            dest_dir = DATASET / genre
            dest_dir.mkdir(exist_ok=True)
            new_name = f"{tid}_{genre}.mp3"
            new_path = dest_dir / new_name
            shutil.move(tmp, str(new_path))
            entry["filename"] = os.path.relpath(str(new_path), str(DATASET))
            entry["md5"] = got
            log(f"REPLACE {tid}: {title} manifest={expected[:12]} got={got[:12]}")
            stats["replaced"] += 1

        stats["processed"] += 1
        write_stats()

        if stats["processed"] % 25 == 0:
            with open(MANIFEST, "w") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(FAILED, "w") as f:
        f.write("\n".join(failed_ids) + "\n" if failed_ids else "")

    with open(CURRENT, "w") as f:
        f.write("DONE\n")

    log(f"Done: {stats['ok']} ok, {stats['replaced']} replaced, {stats['failed']} failed, {stats['skipped']} skipped")
    write_stats()
    log_fh.close()


if __name__ == "__main__":
    main()
