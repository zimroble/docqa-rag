import pytest

from docqa.chunking import Page, chunk_pages, clean, load, read_text
from tests.conftest import make_pdf


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunks_overlap_and_cover_every_word():
    chunks = chunk_pages([Page(1, words(100))], size=40, overlap=10)
    assert [len(c.text.split()) for c in chunks] == [40, 40, 40]
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(words(100).split())


def test_chunks_never_cross_pages():
    chunks = chunk_pages([Page(1, words(50, "a")), Page(2, words(5, "b")), Page(3, "")], size=40, overlap=0)
    assert [(c.page, c.index) for c in chunks] == [(1, 0), (1, 1), (2, 2)]
    assert all(w.startswith("b") for w in chunks[2].text.split())


def test_short_page_is_one_chunk():
    assert len(chunk_pages([Page(1, "just a few words")], size=40, overlap=10)) == 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "x")], size, overlap)


def test_clean_joins_hyphenated_line_breaks_and_whitespace():
    assert clean("concur-\nrency   control\n\n") == "concurrency control"


def test_text_files_use_form_feed_as_page_break():
    pages = read_text(b"first page\fsecond page")
    assert [(p.number, p.text) for p in pages] == [(1, "first page"), (2, "second page")]


def test_pdf_pages_are_numbered_from_one():
    pages = load("doc.pdf", make_pdf(["alpha page", "beta page"]))
    assert [p.number for p in pages] == [1, 2]
    assert "beta" in pages[1].text


def test_unsupported_type():
    with pytest.raises(ValueError):
        load("image.png", b"\x89PNG")
