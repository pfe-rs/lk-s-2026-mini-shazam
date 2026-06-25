import json, sys, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
MANIFEST = ROOT / "manifest.json"
BLACKLIST = ROOT / "corrupted_blacklist.json"


def check_file(filename: str) -> tuple[str, str | None]:
    import librosa
    path = DATASET / filename
    if not path.exists():
        return (filename, "file_not_found")
    try:
        data, _ = librosa.load(str(path), sr=None, mono=True)
        if len(data) == 0:
            return (filename, "empty_audio")
        return (filename, None)
    except Exception as e:
        err = str(e)[:200] or type(e).__name__
        return (filename, err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    filenames = [e.get("filename", "") for e in manifest.values() if e.get("filename")]
    total = len(filenames)

    print(f"Proveravam {total} fajlova sa {args.workers} worker-a...")

    corrupted = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_file, fn): fn for fn in filenames}
        for future in as_completed(futures):
            fn, err = future.result()
            if err:
                corrupted[fn] = err
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{total} — {len(corrupted)} corrupted")

    print(f"\nUkupno: {total} fajlova, {len(corrupted)} corrupted/neispravnih")

    BLACKLIST.write_text(json.dumps(corrupted, indent=2, ensure_ascii=False))
    print(f"Blacklist sacuvan u: {BLACKLIST}")

    if corrupted:
        print("\nPrvih 10:")
        for fn, err in list(corrupted.items())[:10]:
            print(f"  {fn}: {err}")


if __name__ == "__main__":
    main()
