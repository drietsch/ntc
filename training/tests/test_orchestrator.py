"""Orchestrator tests with canned claude-style outputs (no real invocation)."""

import asyncio
import json

import pytest

from synthetic.orchestrator import (
    TeacherError,
    build_request_gen_prompt,
    build_tool_gen_prompt,
    build_verify_vote_prompt,
    extract_json_payload,
    generate_examples,
    parse_claude_envelope,
    run_claude_batch,
    write_shard,
)

CALENDAR_TOOL = {
    "name": "calendar.create",
    "description": "Create a calendar event",
    "parameters": {"title": {"type": "string", "required": True}},
}

VALID_EXAMPLE = {
    "id": "ex-1",
    "lang": "de",
    "utterance": "mach einen Zahnarzttermin",
    "candidates": [CALENDAR_TOOL],
    "gold": {
        "action": "CALL",
        "tool": "calendar.create",
        "arguments": [
            {
                "parameter": "title",
                "semantic_type": "STRING",
                "value": "Zahnarzttermin",
                "char_span": {"start": 11, "end": 25},
                "surface": "Zahnarzttermin",
            }
        ],
    },
}

# The gold has ASK-without-unresolved: fails DatasetExample validation.
INVALID_EXAMPLE = {**VALID_EXAMPLE, "id": "ex-2", "gold": {"action": "ASK", "unresolved": []}}


def envelope(result) -> str:
    """A claude -p --output-format json style reply."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 1200,
            "result": result if isinstance(result, str) else json.dumps(result),
            "session_id": "s-123",
        }
    )


class TestEnvelopeParsing:
    def test_parses_result_field(self):
        assert parse_claude_envelope(envelope("hello")) == "hello"

    def test_rejects_non_json_stdout(self):
        with pytest.raises(TeacherError, match="not JSON"):
            parse_claude_envelope("plain text crash")

    def test_rejects_missing_result(self):
        with pytest.raises(TeacherError, match="no `result`"):
            parse_claude_envelope(json.dumps({"type": "result"}))

    def test_rejects_error_envelope(self):
        raw = json.dumps({"is_error": True, "result": "rate limited"})
        with pytest.raises(TeacherError, match="error result"):
            parse_claude_envelope(raw)

    def test_extracts_fenced_json(self):
        payload = extract_json_payload('Here you go:\n```json\n[{"a": 1}]\n```')
        assert payload == [{"a": 1}]

    def test_extracts_bare_json(self):
        assert extract_json_payload('[{"a": 1}]') == [{"a": 1}]


class TestRunClaudeBatch:
    def test_batch_returns_results_in_order(self):
        async def runner(prompt: str) -> str:
            await asyncio.sleep(0)
            return envelope(f"answer to: {prompt}")

        results = asyncio.run(run_claude_batch(["p1", "p2", "p3"], runner=runner))
        assert results == ["answer to: p1", "answer to: p2", "answer to: p3"]

    def test_batch_captures_errors_per_prompt(self):
        async def runner(prompt: str) -> str:
            if prompt == "boom":
                return "not json"
            return envelope("ok")

        results = asyncio.run(run_claude_batch(["fine", "boom"], runner=runner))
        assert results[0] == "ok"
        assert isinstance(results[1], TeacherError)

    def test_concurrency_is_bounded(self):
        active = 0
        peak = 0

        async def runner(prompt: str) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return envelope("ok")

        asyncio.run(run_claude_batch([f"p{i}" for i in range(8)], concurrency=2, runner=runner))
        assert peak <= 2


class TestGenerateValidateRepair:
    def test_valid_output_accepted_without_retry(self):
        calls = []

        async def runner(prompt: str) -> str:
            calls.append(prompt)
            return envelope([VALID_EXAMPLE])

        report = asyncio.run(generate_examples(["gen-1"], runner=runner))
        assert len(calls) == 1
        assert report.retried == 0
        assert not report.rejected
        assert [ex.id for ex in report.examples] == ["ex-1"]

    def test_invalid_output_retried_once_with_feedback(self):
        calls = []

        async def runner(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return envelope([INVALID_EXAMPLE])
            return envelope([VALID_EXAMPLE])

        report = asyncio.run(generate_examples(["gen-1"], runner=runner))
        assert len(calls) == 2
        # The repair prompt carries the original prompt, the bad reply, and
        # the validation feedback.
        assert calls[1].startswith("gen-1")
        assert "failed validation" in calls[1]
        assert "unresolved" in calls[1]  # pydantic error mentions the ASK rule
        assert report.retried == 1
        assert [ex.id for ex in report.examples] == ["ex-1"]
        assert not report.rejected

    def test_still_invalid_after_retry_is_rejected(self):
        async def runner(prompt: str) -> str:
            return envelope([INVALID_EXAMPLE])

        report = asyncio.run(generate_examples(["gen-1"], runner=runner))
        assert report.retried == 1
        assert not report.examples
        assert len(report.rejected) == 1
        assert report.rejected[0]["stage"] == "retry-validate"

    def test_teacher_failure_is_rejected_without_retry(self):
        async def runner(prompt: str) -> str:
            return "segfault, no json"

        report = asyncio.run(generate_examples(["gen-1"], runner=runner))
        assert report.retried == 0
        assert report.rejected[0]["stage"] == "teacher"

    def test_non_array_payload_is_retried(self):
        calls = []

        async def runner(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return envelope(VALID_EXAMPLE)  # object, not array
            return envelope([VALID_EXAMPLE])

        report = asyncio.run(generate_examples(["gen-1"], runner=runner))
        assert len(calls) == 2
        assert [ex.id for ex in report.examples] == ["ex-1"]


class TestShardOutput:
    def test_write_shard_and_manifest(self, tmp_path):
        async def runner(prompt: str) -> str:
            return envelope([VALID_EXAMPLE])

        prompts = ["gen-1"]
        report = asyncio.run(generate_examples(prompts, runner=runner))
        shard = write_shard(tmp_path, "shard-000", report.examples, prompts)

        lines = shard.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "ex-1"

        manifest = json.loads((tmp_path / "shard-000.manifest.json").read_text())
        assert manifest["count"] == 1
        assert len(manifest["prompt_sha256"]) == 64
        assert "created_at" in manifest


class TestPromptBuilders:
    def test_prompts_mention_their_inputs(self):
        assert "calendar" in build_tool_gen_prompt("calendar", 4)
        p = build_request_gen_prompt([CALENDAR_TOOL], "de", 8)
        assert "calendar.create" in p and "de" in p
        assert "Zahnarzttermin" in build_verify_vote_prompt(VALID_EXAMPLE)
