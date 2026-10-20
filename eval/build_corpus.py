#!/usr/bin/env python3
"""Build the evaluation corpus: five PDFs rendered from chapters of the
PostgreSQL 16 documentation (PostgreSQL License, see eval/corpus/LICENSE).

Needs the Debian/Ubuntu package `postgresql-doc-16` for the HTML sources.
The generated PDFs are committed, so this only needs re-running to change the corpus.
"""
import os
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

# Override with PG_DOC_HTML if dpkg was configured to skip /usr/share/doc.
HTML = Path(os.environ.get("PG_DOC_HTML", "/usr/share/doc/postgresql-doc-16/html"))
OUT = Path(__file__).parent / "corpus"

BOOKS = {
    "postgres-mvcc.pdf": ["mvcc-intro", "transaction-iso", "explicit-locking"],
    "postgres-indexes.pdf": ["indexes-intro", "indexes-types", "indexes-multicolumn", "indexes-unique",
                             "indexes-expressional", "indexes-partial", "indexes-index-only-scans"],
    "postgres-wal.pdf": ["wal-reliability", "wal-intro", "wal-async-commit", "wal-configuration"],
    "postgres-vacuum.pdf": ["routine-vacuuming"],
    "postgres-fulltext.pdf": ["textsearch-intro", "textsearch-controls", "textsearch-indexes"],
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def flowables(page: str, styles):
    soup = BeautifulSoup((HTML / f"{page}.html").read_text(), "html.parser")
    body = soup.select_one("div.sect1, div.chapter, div.sect2") or soup.body
    for nav in body.select(".navheader, .navfooter, .toc"):
        nav.decompose()
    out = []
    for node in body.find_all(["h2", "h3", "h4", "p", "pre", "li", "dt"]):
        if node.find_parent(["li", "p", "pre"]) is not None and node.name != "pre":
            continue  # nested; parent already rendered
        text = " ".join(node.get_text(" ").split())
        if not text:
            continue
        if node.name in ("h2", "h3", "h4"):
            out.append(Paragraph(esc(text), styles["Heading2" if node.name == "h2" else "Heading3"]))
        elif node.name == "pre":
            out.append(Preformatted(node.get_text(), styles["Code"], maxLineLength=90))
        elif node.name in ("li", "dt"):
            out.append(Paragraph("• " + esc(text), styles["BodyText"]))
        else:
            out.append(Paragraph(esc(text), styles["BodyText"]))
        out.append(Spacer(1, 4))
    return out


def main() -> int:
    if not HTML.exists():
        print("install postgresql-doc-16 first", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    for name, pages in BOOKS.items():
        story = []
        for p in pages:
            story += flowables(p, styles)
        doc = SimpleDocTemplate(str(OUT / name), pagesize=letter, title=name, author="PostgreSQL Global Development Group",
                                invariant=1)
        doc.build(story)
        print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
