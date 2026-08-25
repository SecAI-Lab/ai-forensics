"""Vendor the published SST-2 splits and pre-generated trigger tables.

Three problems this solves, all of which would otherwise corrupt the results:

1. **The cached ``stanfordnlp/sst2`` is the wrong split.** It is GLUE's
   *phrase-level* SST-2: 67,349 training rows that are sentence *fragments*
   ("hide new secretions from the parental units"), and a test split whose
   labels are all ``-1``. Every backdoor paper this project is measured against
   — Hidden Killer (ACL 2021), StyleBkd (EMNLP 2021), OpenBackdoor (NeurIPS 2022
   D&B) — uses the *sentence-level* split of 6,920 / 872 / 1,821. Training on
   fragments and comparing ASR to their tables would be comparing two different
   tasks.

2. **The stealthy triggers cannot be generated locally.** A syntactic trigger
   needs SCPN and a style trigger needs a STRAP-style transfer model. Both
   authors released the *exact* paraphrases used in their papers, so vendoring
   them makes our poison identical to the published poison instead of a
   re-implementation that would differ in unmeasurable ways.

3. **Provenance.** Every file is recorded with its source URL and SHA-256, so a
   detection result can name the bytes it was scored against.

Sources (official author releases):
  * ``thunlp/HiddenKiller``  — clean sentence-level SST-2, and the SCPN
    syntactic paraphrases at a 20% poison rate.
  * ``thunlp/StyleAttack``   — the Bible-style transfers.

Usage::

    python scripts/fetch_backdoor_data.py
    python scripts/fetch_backdoor_data.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from aiforensics import PROCESSED_DIR

HK = "https://raw.githubusercontent.com/thunlp/HiddenKiller/main/data"
SA = "https://raw.githubusercontent.com/thunlp/StyleAttack/main/data"

FILES = {
    "clean/train.tsv": f"{HK}/clean/sst-2/train.tsv",
    "clean/dev.tsv": f"{HK}/clean/sst-2/dev.tsv",
    "clean/test.tsv": f"{HK}/clean/sst-2/test.tsv",
    "scpn/train.tsv": f"{HK}/scpn/20/sst-2/train.tsv",
    "scpn/dev.tsv": f"{HK}/scpn/20/sst-2/dev.tsv",
    "scpn/test.tsv": f"{HK}/scpn/20/sst-2/test.tsv",
    "bible/train.tsv": f"{SA}/transfer/bible/sst-2/train.tsv",
    "bible/dev.tsv": f"{SA}/transfer/bible/sst-2/dev.tsv",
    "bible/test.tsv": f"{SA}/transfer/bible/sst-2/test.tsv",
}

ROOT = PROCESSED_DIR / "sst2-sentence"


def read_tsv(path: Path) -> list[tuple[str, int]]:
    """Read a ``text\\tlabel`` TSV, skipping a header row if present."""
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        text, _, label = line.rpartition("\t")
        if i == 0 and label.strip().lower() in {"label", "labels"}:
            continue
        rows.append((text.strip(), int(label)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    digests = {}
    for rel, url in FILES.items():
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() and not args.verify_only:
            print(f"  fetching {rel} ...", flush=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
        if dest.exists():
            digests[rel] = {
                "url": url,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "bytes": dest.stat().st_size,
            }

    print("\n" + "=" * 78)
    print("SPLIT SIZES  (published: 6,920 / 872 / 1,821)")
    print("=" * 78)
    clean = {s: read_tsv(ROOT / f"clean/{s}.tsv") for s in ("train", "dev", "test")}
    for s, rows in clean.items():
        print(f"  clean/{s:5s} {len(rows):6,d} rows   "
              f"labels={sorted({y for _, y in rows})}")

    print("\n" + "=" * 78)
    print("TRIGGER TABLES — row alignment against the clean split")
    print("=" * 78)
    checks = {}
    for kind in ("scpn", "bible"):
        for split in ("train", "test"):
            p = ROOT / f"{kind}/{split}.tsv"
            if not p.exists():
                continue
            trig = read_tsv(p)
            base = clean[split]
            aligned = len(trig) == len(base)
            if aligned:
                changed = sum(1 for a, b in zip(trig, base, strict=True) if a[0] != b[0])
                flipped = sum(1 for a, b in zip(trig, base, strict=True) if a[1] != b[1])
                rate = changed / max(len(base), 1)
                print(f"  {kind}/{split:5s} {len(trig):6,d} rows  aligned=YES  "
                      f"changed={changed:5,d} ({rate:.2%})  label-flipped={flipped:,d}")
            else:
                changed = flipped = -1
                rate = float("nan")
                print(f"  {kind}/{split:5s} {len(trig):6,d} rows  aligned=NO "
                      f"(clean {split} has {len(base):,d}) — this table is a SUBSET, "
                      f"not a row-aligned overlay; join it by text, not by index")
            checks[f"{kind}/{split}"] = {
                "rows": len(trig), "row_aligned": aligned,
                "changed": changed, "label_flipped": flipped, "changed_rate": rate,
            }

    (ROOT / "provenance.json").write_text(json.dumps(
        {"files": digests, "alignment": checks}, indent=2))
    print(f"\nwritten: {ROOT}/provenance.json")
    print(f"         {len(digests)} files, SHA-256 recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
