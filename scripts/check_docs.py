#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["markdown-it-py==4.0.0"]
# ///
"""Check repository Markdown links, guide navigation, and Cargo benchmark inventory.

Run from any directory with `uv run /path/to/repository/scripts/check_docs.py`.
External URLs are deliberately excluded from this deterministic CI check.
"""

import re
import subprocess
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import tomllib
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = MarkdownIt("commonmark")


class HtmlLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.anchors.add(attrs["id"])
        if tag == "a" and "name" in attrs:
            self.anchors.add(attrs["name"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])


def parse_document(path):
    tokens = MARKDOWN.parse(path.read_text())
    anchors = set()
    links = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            children = tokens[i + 1].children or []
            text = "".join(
                t.content
                for t in children
                if t.type in {"text", "code_inline", "image"}
            )
            slug = "".join(
                ch
                for ch in text.lower()
                if ch in "-_ " or unicodedata.category(ch)[0] in {"L", "N", "M"}
            ).replace(" ", "-")
            candidate = slug
            suffix = 0
            while candidate in anchors:
                suffix += 1
                candidate = f"{slug}-{suffix}"
            anchors.add(candidate)
        for child in [token, *(token.children or [])]:
            line = token.map[0] + 1 if token.map else 1
            if child.type == "link_open":
                links.append((child.attrGet("href"), line))
            elif child.type == "image":
                links.append((child.attrGet("src"), line))
            elif child.type in {"html_inline", "html_block"}:
                html = HtmlLinks()
                html.feed(child.content)
                anchors.update(html.anchors)
                links.extend((url, line) for url in html.links)
    return anchors, links


def check(root, paths):
    documents = {path.resolve(): parse_document(path) for path in paths}
    errors = []
    linked_from_index = set()
    link_count = 0
    for path, (_, links) in documents.items():
        for url, line in links:
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                continue
            target = (
                (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            )
            location = f"{path.relative_to(root)}:{line}"
            link_count += 1
            if not target.is_relative_to(root):
                errors.append(f"{location}: link leaves repository: {url}")
            elif not target.exists():
                errors.append(f"{location}: missing file: {url}")
            elif (
                parsed.fragment
                and target in documents
                and unquote(parsed.fragment) not in documents[target][0]
            ):
                errors.append(f"{location}: missing heading: {url}")
            if path == root / "docs/README.md":
                linked_from_index.add(target)

    for path in documents:
        if (
            path.parent == root / "docs"
            and path.name != "README.md"
            and path not in linked_from_index
        ):
            errors.append(f"docs/README.md: guide is not linked: {path.name}")

    guide = root / "docs/benchmarks.md"
    listed = set(
        re.findall(r"^\| `(summa-[\w-]+)`\s*\| `([\w-]+)`", guide.read_text(), re.M)
    )
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    actual = set()
    for member in workspace["workspace"]["members"]:
        manifest = tomllib.loads((root / member / "Cargo.toml").read_text())
        for bench in manifest.get("bench", []):
            actual.add((manifest["package"]["name"], bench["name"]))
    for package, target in sorted(actual - listed):
        errors.append(f"docs/benchmarks.md: missing target: {package}/{target}")
    for package, target in sorted(listed - actual):
        errors.append(f"docs/benchmarks.md: unknown target: {package}/{target}")
    return errors, link_count, len(actual)


def main():
    # Include newly added files so the command works before git add.
    files = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "*.md",
            ],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    paths = sorted({ROOT / name for name in files if name and (ROOT / name).is_file()})
    errors, links, benchmarks = check(ROOT, paths)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"Checked {len(paths)} Markdown files, {links} local links, and {benchmarks} benchmark targets."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
