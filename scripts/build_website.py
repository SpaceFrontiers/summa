#!/usr/bin/env python3
"""Build the Pages source tree from website navigation and maintained guides."""

import argparse
import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
GITHUB = "https://github.com/SpaceFrontiers/summa/blob/main/"
RAW = "https://raw.githubusercontent.com/SpaceFrontiers/summa/main/"
LINK = re.compile(
    r"(?P<image>!?)\[(?P<label>[^\]\n]*)\]\((?P<target>[^\s)]+)(?P<title>\s+\"[^\"]*\")?\)"
)


def build(output: Path) -> None:
    output = output.resolve()
    if output in (WEBSITE, ROOT) or WEBSITE in output.parents or output in ROOT.parents:
        raise ValueError("Output must be a separate build directory")
    pages = {}
    for page in WEBSITE.rglob("*.md"):
        content = page.read_text()
        match = re.search(r"^source_doc: (.+)$", content, re.MULTILINE)
        if match:
            source = (ROOT / match[1]).resolve()
            if not source.is_relative_to(ROOT) or not source.is_file():
                raise ValueError(f"Invalid source in {page}: {match[1]}")
            pages[source] = page.relative_to(WEBSITE)
    shutil.copytree(WEBSITE, output, dirs_exist_ok=True)
    for source, page in pages.items():
        template = (WEBSITE / page).read_text()
        frontmatter = template[: template.index("\n---", 4) + 4]
        frontmatter = re.sub(r"^source_doc: .+\n", "", frontmatter, flags=re.MULTILINE)

        def rewrite(match: re.Match, source=source, page=page) -> str:
            target = match["target"]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith(("#", "/")):
                return match[0]
            local = (source.parent / unquote(parsed.path)).resolve()
            if not local.is_relative_to(ROOT) or not local.exists():
                raise ValueError(
                    f"Broken source link in {source.relative_to(ROOT)}: {target}"
                )
            if local in pages and not match["image"]:
                destination = os.path.relpath(
                    output / pages[local], (output / page).parent
                )
            else:
                base = RAW if match["image"] else GITHUB
                destination = base + quote(local.relative_to(ROOT).as_posix())
            if parsed.query:
                destination += "?" + parsed.query
            if parsed.fragment:
                destination += "#" + parsed.fragment
            return f"{match['image']}[{match['label']}]({destination}{match['title'] or ''})"

        body = (
            LINK.sub(rewrite, source.read_text())
            if source.suffix == ".md"
            else "```protobuf\n" + source.read_text() + "```\n"
        )
        # Keep source examples containing Liquid-looking syntax literal.
        (output / page).write_text(
            frontmatter + "\n\n{% raw %}\n" + body + "{% endraw %}\n"
        )
    print(f"Prepared {len(pages)} current guides in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".context" / "website")
    build(parser.parse_args().output)
