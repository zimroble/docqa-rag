import io
import os

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docqa.config import Settings
from docqa.embeddings import get_embedder
from docqa.main import create_app

TEST_DB = os.environ.get("DOCQA_TEST_DATABASE_URL", "postgresql://docqa:docqa@localhost:5432/docqa_test")


def make_pdf(pages: list[str]) -> bytes:
    """Render each string onto its own PDF page (wrapping long lines)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        y = 740
        for para in text.split("\n"):
            line = ""
            for word in para.split():
                if len(line) + len(word) > 90:
                    c.drawString(50, y, line)
                    y -= 14
                    line = ""
                line += word + " "
            if line:
                c.drawString(50, y, line)
                y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


class FakeLLM:
    """Records prompts and answers by citing the first source."""

    def __init__(self, reply="The answer is in the first source [1]."):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


@pytest.fixture(scope="session")
def embedder():
    return get_embedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def client(embedder, fake_llm):
    app = create_app(Settings(database_url=TEST_DB, chunk_words=60, chunk_overlap=10), embedder, fake_llm)
    with TestClient(app) as c:
        c.app.state.db.reset()
        yield c
