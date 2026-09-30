"""Draw road-sign photos from Wikimedia Commons at random, with a license check and nothing else.

    python3 sample_signs.py OUT.jsonl SET GROUP=CATEGORY:N [GROUP=CATEGORY:N ...]

For each group, Commons search draws files at random from the whole tree under CATEGORY
(`deepcat:`, JPEG only, random order), and the first N with a CC0, public domain or CC BY
license are kept in the order drawn. No image is looked at. Each row gets the group as its
label and the fields of images_signs.jsonl, with `removed` and `note` empty.

images_signs.jsonl came from two draws on 2026-09-30:

    python3 sample_signs.py drawn.jsonl random "stop=Photographs of stop signs:20" \\
        "yield=Photographs of yield signs:20" "one_way=One-way traffic road signs:20" \\
        "speed_limit=Speed limit road signs:20" "no_entry=No entry road signs:20"
    python3 sample_signs.py hard.jsonl damaged "damaged=Damaged road signs:40" \\
        "obscured=Obscured road signs:40"

Before any run, each image of the first draw was checked by eye for two things only: that it
is a photograph, and that the sign its category names is in it; 2 of 100 failed and have
`removed` set. The second draw was labeled by hand (stop, yield, one_way, speed_limit,
no_entry, other, or unclear where no one answer is right), and 4 scans of drawings were
removed. Commons search sorts at random with no seed, so a new draw gives other files.
"""

import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request

USER_AGENT = "lichen-image-bench/0.1 (https://github.com/Mushroom-Systems/lichen)"
API = "https://commons.wikimedia.org/w/api.php"


def get(**params) -> dict:
    params.update(action="query", format="json")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def permissive(name: str) -> bool:
    """CC0, public domain, or CC BY without SA, NC or ND."""
    l = name.lower()
    if l.startswith(("public domain", "pd", "cc0")):
        return True
    return l.startswith("cc by ") and not any(x in l for x in ("sa", "nc", "nd"))


def plain(s: str | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def draw(category: str, n: int, seen: set) -> list[dict]:
    kept = []
    for _ in range(10):
        if len(kept) >= n:
            break
        hits = get(list="search", srnamespace=6, srsearch=f'deepcat:"{category}" filemime:image/jpeg',
                   srlimit=50, srsort="random")["query"]["search"]
        titles = [h["title"] for h in hits if h["title"] not in seen]
        if not titles:
            continue
        pages = get(titles="|".join(titles), prop="imageinfo", iiprop="url|extmetadata|mime",
                    iiurlwidth=960)["query"]["pages"].values()
        info = {p["title"]: p["imageinfo"][0] for p in pages if "imageinfo" in p}
        for title in titles:  # in the order the search drew them
            ii = info.get(title)
            meta = (ii or {}).get("extmetadata", {})
            license_name = plain(meta.get("LicenseShortName", {}).get("value"))
            if len(kept) >= n or not ii or ii.get("mime") != "image/jpeg" or not permissive(license_name):
                continue
            req = urllib.request.Request(ii["thumburl"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            seen.add(title)
            kept.append({"title": title, "page": ii["descriptionurl"], "url": ii["thumburl"],
                         "sha256": hashlib.sha256(data).hexdigest(), "license": license_name,
                         "license_url": plain(meta.get("LicenseUrl", {}).get("value")),
                         "author": plain(meta.get("Artist", {}).get("value"))[:200]})
            time.sleep(0.2)
    return kept


def main() -> None:
    out, set_name, groups = sys.argv[1], sys.argv[2], sys.argv[3:]
    seen: set = set()
    with open(out, "a", encoding="utf-8") as f:
        for spec in groups:
            group, rest = spec.split("=", 1)
            category, n = rest.rsplit(":", 1)
            rows = draw(category, int(n), seen)
            for row in rows:
                f.write(json.dumps({"set": set_name, "category": group, "label": group, **row,
                                    "removed": None, "note": None}, ensure_ascii=False) + "\n")
            print(f"{group}: {len(rows)} from {category}", file=sys.stderr)


if __name__ == "__main__":
    main()
