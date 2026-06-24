import json, hashlib, os, sys, time, shutil, subprocess, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
MANIFEST = ROOT / "manifest.json"
PROGRESS = ROOT / "progress"
PROGRESS.mkdir(exist_ok=True)

RCURRENT = PROGRESS / "rcurrent.txt"
RSTATS = PROGRESS / "rstats.txt"
RLOG = PROGRESS / "rlog.txt"
RFAILED = PROGRESS / "rfailed.txt"

log_fh = open(RLOG, "w")
start = time.time()

stats = {"ok": 0, "redownloaded": 0, "failed": 0, "skipped": 0, "total": 0}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=log_fh, flush=True)


def write_stats():
    elapsed = time.time() - start
    with open(RSTATS, "w") as f:
        f.write(f"Total:      {stats['total']}\n")
        f.write(f"Already OK: {stats['ok']}\n")
        f.write(f"Re-DL'd:    {stats['redownloaded']}\n")
        f.write(f"Failed:     {stats['failed']}\n")
        f.write(f"Skipped:    {stats['skipped']}\n")
        f.write(f"Rate:       {stats['total']/(elapsed/60):.0f}/min\n")
        f.write(f"Elapsed:    {elapsed//60}m{elapsed%60}s\n")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def dl_yt(url, dest):
    cmd = ["yt-dlp", "-x", "--audio-format", "mp3",
           "--audio-quality", "0", "--no-playlist",
           "-o", dest, url]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    for ext in [".mp3", ".webm", ".m4a"]:
        p = dest + ext
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            os.rename(p, dest)
            return True
    return False


def dl_fma(url, dest, timeout=60):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        resp = urllib.request.urlopen(req, timeout=timeout)
        with open(dest, "wb") as f:
            while chunk := resp.read(8192):
                f.write(chunk)
        return os.path.getsize(dest) > 1000
    except Exception:
        return False


def main():
    manifest = json.load(open(MANIFEST))
    total = len(manifest)
    stats["total"] = total

    log(f"Starting recovery: {total} tracks")
    failed_ids = []
    corrupted_count = 0

    for i, (tid, entry) in enumerate(manifest.items(), 1):
        title = entry.get("title", "?")
        url = entry.get("source_url", "")
        expected = entry.get("md5", "")
        filename = entry.get("filename", "")

        with open(RCURRENT, "w") as f:
            f.write(f"{i}/{total} — {tid}: {title}\n")

        if not url:
            log(f"SKIP {tid}: no URL")
            stats["skipped"] += 1
            write_stats()
            continue

        existing = DATASET / filename if filename else None

        if existing and existing.exists() and existing.stat().st_size > 1000:
            if md5(str(existing)) == expected:
                stats["ok"] += 1
                write_stats()
                continue

        corrupted_count += 1
        tmp = str(PROGRESS / f"rec_{tid}.mp3")

        is_yt = "youtube.com" in url or "youtu.be" in url
        ok = dl_yt(url, tmp) if is_yt else dl_fma(url, tmp)

        if not ok:
            log(f"FAIL {tid}: {title}")
            failed_ids.append(tid)
            stats["failed"] += 1
            if os.path.exists(tmp):
                os.remove(tmp)
            write_stats()
            continue

        size = os.path.getsize(tmp)
        got = md5(tmp)

        if existing:
            existing.parent.mkdir(parents=True, exist_ok=True)
            if existing.exists():
                os.remove(str(existing))
            shutil.move(tmp, str(existing))
        else:
            genre = entry.get("genre", "Unknown").replace("/", "_")
            dest_dir = DATASET / genre
            dest_dir.mkdir(parents=True, exist_ok=True)
            new_name = f"{tid}_{genre}.mp3"
            new_path = dest_dir / new_name
            shutil.move(tmp, str(new_path))
            entry["filename"] = os.path.relpath(str(new_path), str(DATASET))

        entry["md5"] = got

        if size < 50000:
            log(f"WARN {tid}: {title} only {size} bytes (might be preview)")
        else:
            log(f"OK {tid}: {title} ({size//1000}KB)")
        stats["redownloaded"] += 1
        write_stats()

        if stats["redownloaded"] % 10 == 0:
            with open(MANIFEST, "w") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(RFAILED, "w") as f:
        f.write("\n".join(failed_ids) + "\n")

    with open(RCURRENT, "w") as f:
        f.write("DONE\n")

    log(f"=== Done: {stats['ok']} ok, {stats['redownloaded']} re-downloaded, {stats['failed']} failed, {stats['skipped']} skipped ===")
    log(f"Corrupted/missing found: {corrupted_count}")
    write_stats()
    log_fh.close()


if __name__ == "__main__":
    main()
