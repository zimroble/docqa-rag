from docqa.answer import NOT_FOUND, answer_question, build_prompt, extract_citations, strip_invalid_markers
from docqa.retrieval import Hit, rrf_fuse, to_or_query
from tests.conftest import FakeLLM


def hits(n):
    return [Hit(i, 1, "doc.pdf", i + 1, f"passage {i}", 1.0) for i in range(n)]


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf_fuse([[1, 2, 3], [3, 2, 4]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == 2 or ids[0] == 3
    assert set(ids) == {1, 2, 3, 4}
    # 2 and 3 appear in both lists, so they beat 1 and 4
    assert set(ids[:2]) == {2, 3}


def test_rrf_scores():
    fused = dict(rrf_fuse([[7]], k=60))
    assert abs(fused[7] - 1 / 61) < 1e-12


def test_or_query_dedupes_and_strips_punctuation():
    assert to_or_query("What's VACUUM? vacuum!") == "what | s | vacuum"
    assert to_or_query("?!") == ""


def test_prompt_numbers_sources_with_page():
    p = build_prompt("q?", hits(2))
    assert "[1] (doc.pdf, page 1)" in p and "[2] (doc.pdf, page 2)" in p
    assert p.endswith("Question: q?")


def test_citations_are_mapped_in_order_of_first_use():
    cites = extract_citations("B [2]. A [1][2]. Bogus [9].", hits(3))
    assert [c.n for c in cites] == [2, 1]
    assert cites[0].page == 2


def test_invalid_markers_are_removed():
    assert strip_invalid_markers("x [1] y [0] z [4]", 3) == "x [1] y  z "


def test_generative_answer_uses_llm_and_cites():
    llm = FakeLLM("Use VACUUM [2].")
    ans = answer_question("q", hits(3), llm)
    assert ans.mode == "generative"
    assert ans.answer == "Use VACUUM [2]."
    assert [c.page for c in ans.citations] == [2]
    assert "Sources:" in llm.calls[0][1]


def test_extractive_mode_without_llm():
    ans = answer_question("q", hits(2), None)
    assert ans.mode == "extractive"
    assert len(ans.citations) == 2


def test_no_hits_means_not_found_without_calling_llm():
    llm = FakeLLM()
    ans = answer_question("q", [], llm)
    assert ans.answer == NOT_FOUND
    assert llm.calls == []
