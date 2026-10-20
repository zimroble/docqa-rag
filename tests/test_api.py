from tests.conftest import make_pdf

FILLER = ("The quarterly report covers office logistics, meeting schedules, and "
          "general administrative updates for all staff members. ") * 3

PAGES = [
    FILLER + "Our cafeteria now serves vegetarian lunch options every Tuesday.",
    FILLER + "The parking garage on level B is closed for resurfacing until March.",
    FILLER + "Employees receive twenty days of paid vacation per calendar year.",
    FILLER + "The VPN must be used when connecting to internal systems from home.",
]


def upload(client, name="handbook.pdf", data=None):
    data = data if data is not None else make_pdf(PAGES)
    return client.post("/api/documents", files={"file": (name, data, "application/pdf")})


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "llm": True}


def test_upload_and_list(client):
    r = upload(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["pages"] == 4 and body["chunks"] >= 4 and not body["duplicate"]
    docs = client.get("/api/documents").json()
    assert [d["filename"] for d in docs] == ["handbook.pdf"]


def test_duplicate_upload_is_detected(client):
    data = make_pdf(PAGES)  # same bytes twice (reportlab stamps a creation time)
    first = upload(client, data=data).json()
    second = upload(client, name="copy.pdf", data=data).json()
    assert second["duplicate"] and second["document_id"] == first["document_id"]
    assert len(client.get("/api/documents").json()) == 1


def test_search_finds_the_right_page_in_every_mode(client):
    upload(client)
    for mode in ("keyword", "vector", "hybrid"):
        r = client.post("/api/search", json={"question": "How many vacation days do employees get?", "mode": mode, "k": 3})
        assert r.status_code == 200
        assert r.json()[0]["page"] == 3, mode


def test_ask_returns_cited_answer(client, fake_llm):
    upload(client)
    r = client.post("/api/ask", json={"question": "When is the parking garage closed?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "generative"
    assert body["citations"][0]["page"] == 2
    assert body["citations"][0]["filename"] == "handbook.pdf"
    assert "resurfacing" in fake_llm.calls[0][1]
    assert set(body["timing_ms"]) == {"retrieval", "generation"}


def test_document_filter(client):
    a = upload(client).json()["document_id"]
    b = upload(client, "other.pdf", make_pdf(["Vacation policy: contractors get no paid vacation."])).json()["document_id"]
    r = client.post("/api/search", json={"question": "paid vacation", "document_ids": [b]})
    assert {h["document_id"] for h in r.json()} == {b}
    r = client.post("/api/search", json={"question": "paid vacation", "document_ids": [a]})
    assert {h["document_id"] for h in r.json()} == {a}


def test_delete_cascades_to_chunks(client):
    doc = upload(client).json()["document_id"]
    assert client.delete(f"/api/documents/{doc}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.post("/api/search", json={"question": "vacation"}).json() == []
    assert client.delete(f"/api/documents/{doc}").status_code == 404


def test_text_upload(client):
    r = client.post("/api/documents", files={"file": ("notes.md", b"# Notes\nThe launch date is June 4.\fPage two.", "text/markdown")})
    assert r.status_code == 201
    assert r.json()["pages"] == 2


def test_rejects_bad_uploads(client):
    assert client.post("/api/documents", files={"file": ("x.png", b"\x89PNG....", "image/png")}).status_code == 400
    assert client.post("/api/documents", files={"file": ("x.pdf", b"", "application/pdf")}).status_code == 400
    assert client.post("/api/documents", files={"file": ("x.pdf", b"%PDF-1.4 garbage", "application/pdf")}).status_code == 400
    assert upload(client, "blank.pdf", make_pdf([""])).status_code == 422


def test_validation_errors(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/search", json={"question": "x", "k": 0}).status_code == 422
    assert client.post("/api/search", json={"question": "x", "mode": "magic"}).status_code == 422


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "DocQA" in r.text
