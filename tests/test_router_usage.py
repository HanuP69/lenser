"""The account of the calls to the models, checked line by line.

A call goes into the sink of the session as one line of JSON. The tests
here open the file again and read every line, so the numbers that the
improvement plan is judged against do not drift quietly.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lecturelens.router import usage as router_usage
from lecturelens.router.llm_router import (
    ProviderUnavailableError,
    wrap_model_with_circuit_breaker
)


class FakeQueue:
    """The gate before the provider, told how it answers."""

    def __init__(self, allowed: bool = True, is_open: bool = False) -> None:
        self.allowed = allowed
        self.is_open_flag = is_open
        self.failures = 0
        self.successes = 0

    async def is_allowed(self) -> bool:
        return self.allowed

    async def is_open(self) -> bool:
        return self.is_open_flag

    async def record_failure(self) -> None:
        self.failures += 1

    async def record_success(self) -> None:
        self.successes += 1


class FakeModel:
    """A provider that answers or fails in the shape the test wants."""

    def __init__(self, response: Any = None, error: Any = None) -> None:
        self.response = response
        self.error = error

    async def ainvoke(self, messages: Any) -> Any:
        if self.error is not None:
            raise self.error
        return self.response


class UsageAccountTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.sink = Path(folder.name) / "llm_usage.jsonl"
        router_usage.reset_for_tests()
        router_usage.set_sink(self.sink)
        self.addCleanup(router_usage.reset_for_tests)

    def read_rows(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.sink.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_the_token_counts_are_read_from_the_known_shapes(self) -> None:
        first = SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 20})
        second = SimpleNamespace(response_metadata={"tokenUsage": {"prompt_tokens": 30, "completion_tokens": 4}})
        third = SimpleNamespace(additional_kwargs={"usage": {"prompt_token_count": 12, "candidates_token_count": 3}})
        fourth = SimpleNamespace(additional_kwargs={"other": 1})
        self.assertEqual(router_usage.tokens_from_response(first), (100, 20))
        self.assertEqual(router_usage.tokens_from_response(second), (30, 4))
        self.assertEqual(router_usage.tokens_from_response(third), (12, 3))
        self.assertEqual(router_usage.tokens_from_response(fourth), (None, None))

    def test_every_call_writes_one_line_with_its_name(self) -> None:
        with router_usage.stage("summarize"):
            router_usage.record(
                "structuring", "unorouter_gemma", "ok",
                input_tokens=1000, output_tokens=250,
                request_seconds=4.0, total_seconds=4.5, model="gemma-4-26b:free"
            )
            router_usage.record("structuring", "unorouter_structuring", "limited")
        rows = self.read_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["stage"], "summarize")
        self.assertEqual(rows[0]["outcome"], "ok")
        self.assertEqual(rows[0]["input_tokens"], 1000)
        self.assertEqual(rows[0]["model"], "gemma-4-26b:free")
        self.assertEqual(rows[1]["stage"], "summarize")
        self.assertEqual(rows[1]["outcome"], "limited")
        self.assertEqual(rows[1]["provider"], "unorouter_structuring")

    def test_the_stage_is_only_named_inside_its_own_window(self) -> None:
        with router_usage.stage("latex"):
            pass
        router_usage.record("structuring", "p", "ok")
        rows = self.read_rows()
        self.assertEqual(rows[0]["stage"], "")

    def test_the_totals_say_how_the_provider_did(self) -> None:
        router_usage.record("structuring", "p1", "ok", input_tokens=10, output_tokens=5, request_seconds=1.0)
        router_usage.record("structuring", "p1", "limited")
        router_usage.record("screen_ocr", "p2", "failed", error="boom")
        totals = router_usage.snapshot()
        structuring = totals["tasks"]["structuring"]
        self.assertEqual(structuring["attempts"], 2)
        self.assertEqual(structuring["ok"], 1)
        self.assertEqual(structuring["limited"], 1)
        self.assertEqual(structuring["input_tokens"], 10)
        self.assertEqual(structuring["providers"]["p1"]["ok"], 1)
        self.assertEqual(totals["tasks"]["screen_ocr"]["failed"], 1)

    def test_a_refused_provider_is_written_down_before_the_chain_moves_on(self) -> None:
        wrapped = wrap_model_with_circuit_breaker(
            FakeModel(),
            "fake_provider",
            FakeQueue(allowed=False),
            FakeQueue(is_open=False),
            None,
            "structuring",
            "fake-model"
        )
        with self.assertRaises(ProviderUnavailableError):
            asyncio.run(wrapped.ainvoke(["hello"]))
        rows = self.read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "limited")
        self.assertEqual(rows[0]["task"], "structuring")

    def test_a_failed_call_carries_its_error(self) -> None:
        wrapped = wrap_model_with_circuit_breaker(
            FakeModel(error=ConnectionError("connection reset")),
            "fake_provider",
            FakeQueue(allowed=True),
            FakeQueue(is_open=False),
            None,
            "screen_ocr",
            "fake-vision"
        )
        with self.assertRaises(ProviderUnavailableError):
            asyncio.run(wrapped.ainvoke(["hello"]))
        rows = self.read_rows()
        self.assertEqual(rows[0]["outcome"], "failed")
        self.assertIn("connection", rows[0]["error"])

    def test_an_answer_counts_its_tokens_and_names_the_provider(self) -> None:
        answer = SimpleNamespace(
            usage_metadata={"input_tokens": 900, "output_tokens": 120},
            additional_kwargs={}
        )
        wrapped = wrap_model_with_circuit_breaker(
            FakeModel(response=answer),
            "fake_provider",
            FakeQueue(),
            FakeQueue(is_open=False),
            None,
            "structuring",
            "fake-model"
        )
        with router_usage.stage("definitions"):
            response = asyncio.run(wrapped.ainvoke(["hello"]))
        self.assertEqual(response.additional_kwargs["lecturelens_provider"], "fake_provider")
        rows = self.read_rows()
        self.assertEqual(rows[0]["outcome"], "ok")
        self.assertEqual(rows[0]["input_tokens"], 900)
        self.assertEqual(rows[0]["output_tokens"], 120)
        self.assertEqual(rows[0]["stage"], "definitions")
        self.assertEqual(rows[0]["task"], "structuring")

    def test_an_open_breaker_is_not_counted_as_a_failure_of_the_model(self) -> None:
        breaker = FakeQueue(allowed=True, is_open=True)
        wrapped = wrap_model_with_circuit_breaker(
            FakeModel(),
            "fake_provider",
            FakeQueue(allowed=True),
            breaker,
            None,
            "structuring",
            "fake-model"
        )
        with self.assertRaises(ProviderUnavailableError):
            asyncio.run(wrapped.ainvoke(["hello"]))
        rows = self.read_rows()
        self.assertEqual(rows[0]["outcome"], "breaker_open")
        self.assertEqual(breaker.failures, 0)


if __name__ == "__main__":
    unittest.main()
