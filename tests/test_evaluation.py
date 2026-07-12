"""
Tests for the evaluation/ package:
    models.py       -> EvaluationResult
    metrics.py       -> precision_at_k, recall_at_k, keyword_coverage_score, cosine_similarity_score
    llm_judge.py      -> LLMJudge (faithfulness, ONE llm call per score_faithfulness())
    evaluators.py     -> RoutingEvaluator, RetrievalEvaluator, AnswerEvaluator
    run_evaluation.py -> load_ground_truth, run_single, run_evaluation, write_results_csv, summarize

The pipeline tests build a real RouterAgent, but wire it up with
MockLLMClient/MockVectorStore (see conftest.py) so they never call the
real Gemini API or a real ChromaDB instance.
"""

import csv
import json

import pytest

from evaluation.models import EvaluationResult
from evaluation import metrics
from evaluation.llm_judge import LLMJudge
from evaluation.evaluators import RoutingEvaluator, RetrievalEvaluator, AnswerEvaluator
from evaluation.run_evaluation import (
    load_ground_truth,
    run_single,
    run_evaluation,
    write_results_csv,
    summarize,
    CSV_FIELDNAMES,
    DEFAULT_GROUND_TRUTH_PATH,
)
from src.agents.router_agent.router_main import RouterAgent
from src.framework.core.data_models import RAGResponse

from conftest import MockLLMClient, make_chunk


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

def test_precision_at_k_counts_chunks_containing_a_keyword():
    chunks = [
        make_chunk("You need a TIN from the IRD.", chunk_id="1"),
        make_chunk("Completely unrelated text.", chunk_id="2"),
    ]
    assert metrics.precision_at_k(chunks, ["TIN"], k=2) == pytest.approx(0.5)


def test_precision_at_k_empty_chunks_or_keywords_is_zero():
    assert metrics.precision_at_k([], ["TIN"], k=3) == 0.0
    assert metrics.precision_at_k([make_chunk("text")], [], k=3) == 0.0


def test_recall_at_k_fraction_of_keywords_found_across_top_k():
    chunks = [
        make_chunk("You need a TIN.", chunk_id="1"),
        make_chunk("Apply via the IRD portal.", chunk_id="2"),
    ]
    assert metrics.recall_at_k(chunks, ["TIN", "IRD", "NIC"], k=2) == pytest.approx(2 / 3)


def test_recall_at_k_empty_keywords_is_zero():
    assert metrics.recall_at_k([make_chunk("text")], [], k=3) == 0.0


def test_keyword_coverage_score_all_matched():
    assert metrics.keyword_coverage_score("You need a TIN from the IRD.", ["TIN", "IRD"]) == 1.0


def test_keyword_coverage_score_partial_match_is_case_insensitive():
    score = metrics.keyword_coverage_score("You need a tin from the ird.", ["TIN", "IRD", "NIC"])
    assert score == pytest.approx(2 / 3)


def test_keyword_coverage_score_empty_keyword_list_is_zero():
    assert metrics.keyword_coverage_score("any answer", []) == 0.0


def test_keyword_coverage_score_handles_empty_answer():
    assert metrics.keyword_coverage_score("", ["TIN"]) == 0.0


def test_matched_keywords_returns_only_found_terms():
    found = metrics.matched_keywords("You need a TIN from the IRD.", ["TIN", "IRD", "NIC"])
    assert found == ["TIN", "IRD"]


def test_cosine_similarity_score_identical_text_is_close_to_one():
    score = metrics.cosine_similarity_score(
        "You need a TIN from the IRD e-Services portal.",
        "You need a TIN from the IRD e-Services portal.",
    )
    assert score == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_score_empty_text_is_zero():
    assert metrics.cosine_similarity_score("", "something") == 0.0
    assert metrics.cosine_similarity_score("something", "") == 0.0


def test_cosine_similarity_score_never_calls_an_llm(monkeypatch):
    # Force the sentence-transformers path to fail so we exercise the
    # dependency-free fallback -- either way, no network/API call happens.
    monkeypatch.setattr(metrics, "_embedding_model", lambda: (_ for _ in ()).throw(ImportError()))
    score = metrics.cosine_similarity_score("tin registration", "tin registration process")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------

def test_evaluation_result_to_row_matches_csv_fieldnames():
    result = EvaluationResult(
        question="q",
        expected_agent="agent1_tin_registration",
        predicted_agent="agent1_tin_registration",
        precision_at_1=1.0,
        precision_at_3=1.0,
        recall_at_1=1.0,
        recall_at_3=1.0,
        cosine_accuracy=0.9,
        keyword_score=1.0,
        faithfulness=0.8,
        latency_seconds=0.1,
        generated_answer="a",
    )
    row = result.to_row()
    assert set(row.keys()) == set(EvaluationResult.CSV_FIELDNAMES)
    assert row["error"] == ""


# ---------------------------------------------------------------------------
# llm_judge.py
# ---------------------------------------------------------------------------

def test_llm_judge_makes_exactly_one_llm_call_per_score():
    llm = MockLLMClient(default_response='{"faithfulness": 0.75}')
    judge = LLMJudge(llm)

    score = judge.score_faithfulness(
        question="How do I register for a TIN?",
        answer="Apply via the IRD e-Services portal.",
        context_chunks=[make_chunk("Apply via the IRD e-Services portal.")],
    )

    assert score == pytest.approx(0.75)
    assert len(llm.calls) == 1


def test_llm_judge_returns_zero_on_unparseable_response():
    llm = MockLLMClient(default_response="not json at all")
    judge = LLMJudge(llm)

    score = judge.score_faithfulness("q", "a", [])

    assert score == 0.0
    assert len(llm.calls) == 1


def test_llm_judge_clamps_out_of_range_scores():
    llm = MockLLMClient(default_response='{"faithfulness": 5.0}')
    judge = LLMJudge(llm)

    score = judge.score_faithfulness("q", "a", [])

    assert score == 1.0


# ---------------------------------------------------------------------------
# evaluators.py
# ---------------------------------------------------------------------------

def test_routing_evaluator_true_and_false():
    assert RoutingEvaluator.is_correct("agent1_tin_registration", "agent1_tin_registration") is True
    assert RoutingEvaluator.is_correct("agent1_tin_registration", "agent2_individual_income_tax") is False


def test_retrieval_evaluator_returns_all_four_scores():
    response = RAGResponse(
        question="q",
        retrieved_chunks=[make_chunk("You need a TIN from the IRD.")],
        answer="a",
    )
    scores = RetrievalEvaluator().evaluate(response, ["TIN", "IRD"])
    assert set(scores.keys()) == {"precision_at_1", "precision_at_3", "recall_at_1", "recall_at_3"}


def test_answer_evaluator_calls_judge_exactly_once():
    judge_llm = MockLLMClient(default_response='{"faithfulness": 0.6}')
    judge = LLMJudge(judge_llm)
    evaluator = AnswerEvaluator(judge)

    response = RAGResponse(
        question="How do I register for a TIN?",
        retrieved_chunks=[make_chunk("Apply via the IRD e-Services portal.")],
        answer="Apply via the IRD e-Services portal.",
    )

    scores = evaluator.evaluate(response, ["TIN", "IRD"], "Apply via the IRD e-Services portal.")

    assert set(scores.keys()) == {"keyword_score", "cosine_accuracy", "faithfulness"}
    assert scores["faithfulness"] == pytest.approx(0.6)
    assert len(judge_llm.calls) == 1  # exactly one Gemini/LLM call for this question


# ---------------------------------------------------------------------------
# ground truth loading
# ---------------------------------------------------------------------------

def test_load_ground_truth_returns_ten_questions():
    data = load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)

    assert len(data) == 10
    for item in data:
        assert {"id", "question", "expected_agent", "keywords"} <= item.keys()


def test_load_ground_truth_from_custom_path(tmp_path):
    custom_data = [
        {
            "id": "T1",
            "question": "test question",
            "expected_agent": "agent1_tin_registration",
            "reference_answer": "ref",
            "keywords": ["TIN"],
        }
    ]
    path = tmp_path / "custom_ground_truth.json"
    path.write_text(json.dumps(custom_data), encoding="utf-8")

    assert load_ground_truth(path) == custom_data


# ---------------------------------------------------------------------------
# run_single / run_evaluation (mocked router + mocked judge, no real API calls)
# ---------------------------------------------------------------------------

@pytest.fixture
def router_with_mocks(populated_vector_store):
    llm = MockLLMClient(default_response="You need a TIN via the IRD e-Services portal.")
    return RouterAgent(llm=llm, vector_store=populated_vector_store)


@pytest.fixture
def answer_evaluator():
    judge_llm = MockLLMClient(default_response='{"faithfulness": 0.7}')
    return AnswerEvaluator(LLMJudge(judge_llm))


def test_run_single_scores_a_correctly_routed_question(router_with_mocks, answer_evaluator):
    item = {
        "id": "Q1",
        "question": "How do I register for a TIN?",
        "expected_agent": "agent1_tin_registration",
        "keywords": ["TIN", "IRD"],
    }

    row = run_single(router_with_mocks, item, answer_evaluator, RetrievalEvaluator())

    assert row["error"] == ""
    assert row["predicted_agent"] == "agent1_tin_registration"
    assert row["predicted_agent"] == row["expected_agent"]
    assert row["faithfulness"] == pytest.approx(0.7)
    assert set(row.keys()) == set(CSV_FIELDNAMES)


def test_run_single_captures_exceptions_without_raising(populated_vector_store, answer_evaluator):
    class ExplodingLLM(MockLLMClient):
        def generate(self, prompt, system_instruction=""):
            raise RuntimeError("simulated API failure")

    router = RouterAgent(llm=ExplodingLLM(), vector_store=populated_vector_store)

    item = {
        "id": "Q_ERR",
        "question": "some totally unrelated question that needs LLM routing",
        "expected_agent": "agent2_individual_income_tax",
        "keywords": ["APIT"],
    }

    row = run_single(router, item, answer_evaluator, RetrievalEvaluator())

    assert row["error"] != ""
    assert "simulated API failure" in row["error"]


def test_run_evaluation_writes_csv_and_returns_summary(tmp_path, router_with_mocks):
    ground_truth_path = tmp_path / "gt.json"
    ground_truth_path.write_text(
        json.dumps(
            [
                {
                    "id": "Q1",
                    "question": "How do I register for a TIN?",
                    "expected_agent": "agent1_tin_registration",
                    "keywords": ["TIN"],
                },
                {
                    "id": "Q2",
                    "question": "What is my APIT deduction?",
                    "expected_agent": "agent2_individual_income_tax",
                    "keywords": ["APIT"],
                },
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "results.csv"
    judge_llm = MockLLMClient(default_response='{"faithfulness": 0.9}')

    summary = run_evaluation(
        ground_truth_path=ground_truth_path,
        output_path=output_path,
        router=router_with_mocks,
        judge_llm=judge_llm,
    )

    assert output_path.exists()

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert summary["total_questions"] == 2
    assert 0.0 <= summary["routing_accuracy"] <= 1.0
    # ONE extra LLM call per question was made for the faithfulness judge
    assert len(judge_llm.calls) == 2


def test_run_evaluation_respects_limit(router_with_mocks, answer_evaluator, tmp_path):
    output_path = tmp_path / "results_limited.csv"
    judge_llm = MockLLMClient(default_response='{"faithfulness": 0.5}')

    summary = run_evaluation(
        ground_truth_path=DEFAULT_GROUND_TRUTH_PATH,
        output_path=output_path,
        limit=3,
        router=router_with_mocks,
        judge_llm=judge_llm,
    )

    assert summary["total_questions"] == 3
    assert len(judge_llm.calls) == 3


def test_summarize_computes_accuracy_and_averages():
    rows = [
        {
            "expected_agent": "agent1_tin_registration",
            "predicted_agent": "agent1_tin_registration",
            "precision_at_1": 1.0, "precision_at_3": 1.0,
            "recall_at_1": 1.0, "recall_at_3": 1.0,
            "cosine_accuracy": 0.8, "keyword_score": 1.0,
            "faithfulness": 1.0, "latency_seconds": 0.5, "error": "",
        },
        {
            "expected_agent": "agent1_tin_registration",
            "predicted_agent": "agent2_individual_income_tax",
            "precision_at_1": 0.0, "precision_at_3": 0.0,
            "recall_at_1": 0.0, "recall_at_3": 0.0,
            "cosine_accuracy": 0.4, "keyword_score": 0.5,
            "faithfulness": 0.5, "latency_seconds": 1.5, "error": "",
        },
    ]

    summary = summarize(rows)

    assert summary["total_questions"] == 2
    assert summary["successful_runs"] == 2
    assert summary["routing_accuracy"] == 0.5
    assert summary["avg_keyword_score"] == 0.75
    assert summary["avg_latency_seconds"] == 1.0
    assert summary["avg_cosine_accuracy"] == pytest.approx(0.6)
    assert summary["avg_faithfulness"] == 0.75


def test_summarize_excludes_errored_rows_from_averages():
    rows = [
        {
            "expected_agent": "agent1_tin_registration",
            "predicted_agent": "agent1_tin_registration",
            "precision_at_1": 1.0, "precision_at_3": 1.0,
            "recall_at_1": 1.0, "recall_at_3": 1.0,
            "cosine_accuracy": 0.8, "keyword_score": 1.0,
            "faithfulness": 1.0, "latency_seconds": 0.5, "error": "",
        },
        {"error": "RuntimeError: boom"},
    ]

    summary = summarize(rows)

    assert summary["total_questions"] == 2
    assert summary["successful_runs"] == 1
    assert summary["errors"] == 1
    assert summary["routing_accuracy"] == 1.0


def test_write_results_csv_uses_evaluation_result_fieldnames(tmp_path):
    rows = [
        EvaluationResult(
            question="q", expected_agent="a", predicted_agent="a",
            precision_at_1=1.0, precision_at_3=1.0, recall_at_1=1.0, recall_at_3=1.0,
            cosine_accuracy=1.0, keyword_score=1.0, faithfulness=1.0,
            latency_seconds=0.1, generated_answer="ans",
        ).to_row()
    ]
    output_path = tmp_path / "out.csv"

    write_results_csv(rows, output_path)

    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDNAMES
