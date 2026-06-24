import json, hashlib, os, sys, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
MANIFEST = ROOT / "manifest.json"


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_url(url: str, timeout: int = 15) -> tuple[str, int, str]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        resp = urllib.request.urlopen(req, timeout=timeout)
        return (url, resp.status, "")
    except urllib.error.HTTPError as e:
        return (url, e.code, str(e.reason))
    except Exception as e:
        return (url, 0, str(e)[:120])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-urls", action="store_true", help="HEAD-request every URL")
    parser.add_argument("-j", "--workers", type=int, default=1, help="Parallel URL checks")
    args = parser.parse_args()

    manifest = json.load(open(MANIFEST))

    print("=== MD5 Verification ===")
    md5_ok = 0
    md5_fail = 0
    file_missing = 0
    no_entry = 0

    for tid, entry in manifest.items():
        filename = entry.get("filename")
        expected_md5 = entry.get("md5")

        if not filename or not expected_md5:
            no_entry += 1
            continue

        full_path = DATASET / filename
        if not full_path.exists():
            file_missing += 1
            print(f"  MISSING FILE: {tid} -> {filename}")
            continue

        actual_md5 = file_md5(str(full_path))
        if actual_md5 == expected_md5:
            md5_ok += 1
        else:
            md5_fail += 1
            print(f"  MD5 MISMATCH: {tid} {entry.get('title','?')}")
            print(f"    manifest: {expected_md5}")
            print(f"    actual:   {actual_md5}")

    print(f"\nMD5 OK:       {md5_ok}")
    print(f"MD5 MISMATCH: {md5_fail}")
    print(f"File missing: {file_missing}")
    print(f"No entry:     {no_entry}")

    if args.check_urls:
        print(f"\n=== URL Verification ({args.workers} workers) ===")
        urls = [(tid, e["source_url"]) for tid, e in manifest.items() if e.get("source_url")]

        ok = 0
        fail = 0
        errors = {}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(check_url, url): (tid, url) for tid, url in urls}
            for i, future in enumerate(as_completed(futures), 1):
                tid, url = futures[future]
                _, status, err = future.result()
                if 200 <= status < 400:
                    ok += 1
                else:
                    fail += 1
                    errors[tid] = (url, status, err)
                    print(f"  [{i}/{len(urls)}] FAIL {tid}: HTTP {status} - {err[:80]}")
                if i % 100 == 0:
                    print(f"  ... {i}/{len(urls)} checked")

        print(f"\nURL OK:   {ok}")
        print(f"URL FAIL: {fail}")
        if errors:
            print("\nFailed URLs:")
            for tid, (url, status, err) in sorted(errors.items()):
                print(f"  {tid}: {url} -> HTTP {status} ({err[:60]})")

    total = len(manifest)
    print(f"\n{'='*40}")
    print(f"Total entries: {total}")
    print(f"All MD5 match: {'YES' if md5_fail == 0 and file_missing == 0 else 'NO'}")


if __name__ == "__main__":
    main()
