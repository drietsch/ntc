"""Batch teacher driver for the synthetic data engine (skeleton).

The teacher is headless Claude Code (`claude -p --output-format json`), NOT
the raw Anthropic API. Every teacher call goes through `run_claude_batch`,
which shells out once per prompt (prompt on stdin), bounded by a concurrency
semaphore, and parses the CLI's JSON envelope (`{"result": "..."}`).

Pipeline stages (spec: data engine):
1. tool-gen     — generate themed tool families (raw schemas);
2. request-gen  — generate utterances + gold labels against candidate tools;
3. verify-vote  — k-vote verification of generated examples.

Generated examples are validated against `datasets.schema.DatasetExample`;
invalid outputs get exactly one repair retry with the validation feedback
appended to the prompt. Accepted examples are written as JSONL shards with a
`{prompt_sha256, created_at, count}` manifest.

Tests inject a fake `runner`; nothing here invokes `claude` during tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from datasets.schema import DatasetExample

CLAUDE_CMD: tuple[str, ...] = ("claude", "-p", "--output-format", "json")

#: A runner takes a prompt and returns the raw stdout of the teacher process.
Runner = Callable[[str], Awaitable[str]]


class TeacherError(RuntimeError):
    """The teacher process failed or returned an unusable envelope."""


# --- prompt builders --------------------------------------------------------


def build_tool_gen_prompt(domain: str, n_tools: int, lang: str = "en") -> str:
    return (
        "You generate tool schemas for a function-calling dataset.\n"
        f"Domain: {domain}. Language for descriptions: {lang}.\n"
        f"Produce a JSON array of {n_tools} OpenAI-style tool schemas "
        '({"name", "description", "parameters": {...}}). '
        "Parameter fields: type, description, required, enum, format, semantic.\n"
        "Reply with ONLY the JSON array."
    )


def build_request_gen_prompt(
    candidates: Sequence[dict[str, Any]], lang: str, n_examples: int, split: str = "train"
) -> str:
    return (
        "You generate training examples for a neural tool-call compiler.\n"
        f"Candidate tools (raw schemas):\n{json.dumps(list(candidates), ensure_ascii=False)}\n"
        f"Language: {lang}. Produce a JSON array of {n_examples} dataset examples, "
        "each shaped as {id, lang, utterance, candidates, gold: {action: CALL|ASK|NO_CALL, "
        "tool, arguments: [{parameter, semantic_type, value, char_span: [start, end), "
        f'surface}}], unresolved}}, split: "{split}", tags}}.\n'
        "char_span offsets are CHARACTER offsets into the utterance; the span text must "
        "equal `surface`. ASK examples need at least one unresolved entry; CALL examples "
        "need a tool from the candidate list. Mix CALL/ASK/NO_CALL.\n"
        "Reply with ONLY the JSON array."
    )


def build_verify_vote_prompt(example: dict[str, Any]) -> str:
    return (
        "You verify a generated dataset example for a neural tool-call compiler.\n"
        f"Example:\n{json.dumps(example, ensure_ascii=False)}\n"
        "Check: does the gold action/tool/argument set faithfully represent the "
        "utterance given the candidate tools? Are all spans exact?\n"
        'Reply with ONLY a JSON object {"verdict": "ACCEPT"|"REJECT", "reasons": [...]}.'
    )


# --- teacher invocation -----------------------------------------------------


# Transient process-level failures (empty stderr, exit 1) happen under
# concurrency; retry with exponential backoff before giving up on a prompt.
PROCESS_RETRIES = 3
BACKOFF_BASE_S = 4.0


async def _run_claude_once(prompt: str, cmd: Sequence[str] = CLAUDE_CMD) -> str:
    """Default runner: one headless-Claude call, prompt on stdin, retried on
    transient process failures."""
    last: str = ""
    for attempt in range(PROCESS_RETRIES):
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(prompt.encode())
        if proc.returncode == 0:
            return stdout.decode()
        last = f"claude exited {proc.returncode}: {stderr.decode()[:300]}"
        if attempt < PROCESS_RETRIES - 1:
            await asyncio.sleep(BACKOFF_BASE_S * (2**attempt))
    raise TeacherError(last)


def parse_claude_envelope(raw_stdout: str) -> str:
    """Extract the `result` field from a `claude -p --output-format json` reply."""
    try:
        envelope = json.loads(raw_stdout)
    except json.JSONDecodeError as e:
        raise TeacherError(f"teacher stdout is not JSON: {e}") from e
    if not isinstance(envelope, dict) or "result" not in envelope:
        raise TeacherError("teacher envelope has no `result` field")
    if envelope.get("is_error"):
        raise TeacherError(f"teacher reported an error result: {envelope['result']!r:.500}")
    result = envelope["result"]
    if not isinstance(result, str):
        raise TeacherError(f"teacher `result` is not a string: {type(result).__name__}")
    return result


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_payload(result_text: str) -> Any:
    """Parse the teacher's answer text as JSON, tolerating a ```json fence."""
    text = result_text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise TeacherError(f"teacher result is not valid JSON: {e}") from e


async def run_claude_batch(
    prompts: Sequence[str],
    concurrency: int = 4,
    runner: Runner | None = None,
) -> list[str | TeacherError]:
    """Run all prompts through the teacher, at most `concurrency` at a time.

    Returns, per prompt (in order), the extracted `result` text or the
    `TeacherError` that call raised.
    """
    runner = runner or _run_claude_once
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> str | TeacherError:
        async with sem:
            try:
                return parse_claude_envelope(await runner(prompt))
            except TeacherError as e:
                return e

    return list(await asyncio.gather(*(one(p) for p in prompts)))


# --- generate -> validate -> repair ----------------------------------------


@dataclass
class BatchReport:
    examples: list[DatasetExample] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    retried: int = 0


def _validate_examples(payload: Any) -> list[DatasetExample]:
    if not isinstance(payload, list):
        raise ValidationError.from_exception_data(
            "DatasetExample", [{"type": "list_type", "loc": (), "input": payload}]
        )
    return [DatasetExample.model_validate(item) for item in payload]


def repair_prompt(original_prompt: str, bad_output: str, error: Exception) -> str:
    return (
        f"{original_prompt}\n\n"
        "Your previous reply failed validation and must be corrected.\n"
        f"Previous reply:\n{bad_output}\n"
        f"Validation errors:\n{error}\n"
        "Reply again with ONLY the corrected JSON array."
    )


async def generate_examples(
    prompts: Sequence[str],
    concurrency: int = 4,
    runner: Runner | None = None,
) -> BatchReport:
    """request-gen stage: prompts -> validated examples, one repair retry each."""
    report = BatchReport()
    results = await run_claude_batch(prompts, concurrency=concurrency, runner=runner)

    retry_prompts: list[str] = []
    retry_meta: list[dict[str, Any]] = []
    for prompt, result in zip(prompts, results, strict=True):
        if isinstance(result, TeacherError):
            report.rejected.append({"prompt": prompt, "error": str(result), "stage": "teacher"})
            continue
        try:
            report.examples.extend(_validate_examples(extract_json_payload(result)))
        except (TeacherError, ValidationError) as e:
            retry_prompts.append(repair_prompt(prompt, result, e))
            retry_meta.append({"prompt": prompt, "first_error": str(e)})

    if retry_prompts:
        report.retried = len(retry_prompts)
        retry_results = await run_claude_batch(retry_prompts, concurrency=concurrency, runner=runner)
        for meta, result in zip(retry_meta, retry_results, strict=True):
            if isinstance(result, TeacherError):
                report.rejected.append({**meta, "error": str(result), "stage": "retry-teacher"})
                continue
            try:
                report.examples.extend(_validate_examples(extract_json_payload(result)))
            except (TeacherError, ValidationError) as e:
                report.rejected.append({**meta, "error": str(e), "stage": "retry-validate"})

    return report


# --- shard + manifest output ------------------------------------------------


def write_shard(
    out_dir: str | Path,
    shard_name: str,
    examples: Sequence[DatasetExample],
    prompts: Sequence[str],
) -> Path:
    """Write `<shard_name>.jsonl` + `<shard_name>.manifest.json`.

    The manifest pins provenance: sha256 over the prompts that produced the
    shard, creation timestamp, and example count.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_path = out_dir / f"{shard_name}.jsonl"
    with shard_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
            f.write("\n")

    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode())
        digest.update(b"\x00")
    manifest = {
        "prompt_sha256": digest.hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "count": len(examples),
    }
    (out_dir / f"{shard_name}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return shard_path
