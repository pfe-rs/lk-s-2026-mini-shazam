import json, os, sys
from pathlib import Path

ROOT = Path(__file__).parent
MANIFEST = ROOT / "manifest.json"
BACKUP = ROOT / "manifest_backup.json"
DATASET = ROOT / "dataset"
DRY_RUN = "--dry-run" in sys.argv

manifest = json.load(open(MANIFEST))
fma_json = {}
for line in open(ROOT / "fma_track_files.json"):
    e = json.loads(line)
    fma_json[str(e["track_id"])] = e

yt_id = 0
fma_ok = 0
fma_fixed = 0
unknown = 0
changes = []

for tid, entry in manifest.items():
    filename = entry.get("filename", "")
    current_url = entry.get("source_url", "")
    base = os.path.basename(filename).replace(".mp3", "")
    parts = base.split("_")

    if len(parts) >= 2:
        last = parts[-1]
        if len(last) == 11 and all(c.isalnum() or c in "-_" for c in last):
            correct_url = f"https://www.youtube.com/watch?v={last}"
            if current_url != correct_url:
                changes.append((tid, entry.get("title", "?"), current_url, correct_url))
                entry["source_url"] = correct_url
            yt_id += 1
            continue

    if "fma_" in base.lower():
        if tid in fma_json and fma_json[tid].get("track_url"):
            fma_url = fma_json[tid]["track_url"]
            if current_url != fma_url:
                changes.append((tid, entry.get("title", "?"), current_url, fma_url))
                entry["source_url"] = fma_url
                fma_fixed += 1
            else:
                fma_ok += 1
            continue
        elif tid in fma_json and fma_json[tid].get("track_file"):
            fma_file = fma_json[tid]["track_file"].replace("\\/", "/")
            fp = fma_file.split("/")
            if len(fp) >= 4:
                curator, artist, album = fp[1], fp[2], fp[3]
                track_name = fp[-1].replace(".mp3", "")
                if track_name.startswith(artist + "_ - _"):
                    track_name = track_name[len(artist + "_ - _"):]
                fma_url = f"http://freemusicarchive.org/music/{curator}/{artist}/{album}/{track_name}"
                if current_url != fma_url:
                    changes.append((tid, entry.get("title", "?"), current_url, fma_url))
                    entry["source_url"] = fma_url
                    fma_fixed += 1
                else:
                    fma_ok += 1
                continue

    if "freemusicarchive.org" in current_url:
        fma_ok += 1
        continue

    unknown += 1
    print(f"  UNKNOWN {tid}: {entry.get('title','?')} fn={base}")

print(f"\nYouTube IDs found:   {yt_id}")
print(f"FMA URLs correct:    {fma_ok}")
print(f"FMA URLs fixed:      {fma_fixed}")
print(f"Unknown:             {unknown}")
print(f"URLs to change:      {len(changes)}")

if changes:
    print(f"\n{'='*60}")
    print("SAMPLE CHANGES (first 10):")
    for tid, title, old, new in changes[:10]:
        print(f"  {tid}: {title}")
        print(f"    old: {old[:80]}")
        print(f"    new: {new[:80]}")

if DRY_RUN:
    print("\n[DRY RUN] No changes written.")
else:
    if "--yes" in sys.argv:
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\nManifest updated. {len(changes)} URLs changed.")
    else:
        print(f"\nRun with --yes to apply, or --dry-run to preview only.")
