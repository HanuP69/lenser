import asyncio
import unittest
from typing import Any, Optional
from unittest import mock

import redis.asyncio as redis_module

from lecturelens.config.settings import Settings

from lecturelens.core.circuit_breaker import RedisCircuitBreaker, RedisRateLimiter
from lecturelens.router import llm_router
from lecturelens.router.llm_router import (
    ProviderUnavailableError,
    build_router_for_task,
    get_router_status,
    is_provider_failure,
    wrap_model_with_circuit_breaker
)
from lecturelens.router.messages import (
    build_image_message,
    extract_provider_name,
    parse_confidence_footer,
    response_to_text
)
from langchain_core.messages import AIMessage, HumanMessage
from tests.support import index_by


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeModel:
    def __init__(self, name: str, error: Optional[BaseException] = None, delay: float = 0.0) -> None:
        self.name = name
        self.error = error
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, messages: Any) -> Any:
        self.calls += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return AIMessage(content=f"answer from {self.name}")


CONFIG = {
    "providers": {
        "local": {"type": "ollama", "model": "m", "circuit_breaker": {"failure_threshold": 2, "cooldown_seconds": 30}},
        "cloud_a": {"type": "openai_compatible", "model": "x", "api_key_env": "TEST_KEY_A"},
        "cloud_b": {"type": "google_genai", "model": "y", "api_key_env": "TEST_KEY_B"}
    },
    "task_routing": {"structuring": ["local", "cloud_a", "cloud_b"]}
}


class RouterTestBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = redis_module.from_url(Settings().redis_url, decode_responses=True)
        try:
            await self.client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False
        if self.redis_available:
            await self.clear_keys()

    async def clear_keys(self) -> None:
        for provider in ("local", "cloud_a", "cloud_b", "t_provider"):
            for key in await self.client.keys(f"lecturelens:*{provider}*"):
                await self.client.delete(key)

    async def asyncTearDown(self) -> None:
        if self.redis_available:
            await self.clear_keys()
            await self.client.aclose()

    def install_models(self, models: dict[str, FakeModel]) -> Any:
        def fake_create(provider_name: str, provider_config: dict, ollama_url: str = "") -> Any:
            return models.get(provider_name)

        return mock.patch.object(llm_router, "create_base_chat_model", side_effect=fake_create)


class TestFailureClassification(unittest.TestCase):
    def test_classification(self) -> None:
        self.assertTrue(is_provider_failure(ConnectionError("x")))
        self.assertTrue(is_provider_failure(asyncio.TimeoutError()))
        self.assertTrue(is_provider_failure(StatusError(503)))
        self.assertTrue(is_provider_failure(StatusError(429)))
        self.assertTrue(is_provider_failure(StatusError(401)))
        self.assertFalse(is_provider_failure(StatusError(400)))
        self.assertFalse(is_provider_failure(ValueError("bad prompt")))
        self.assertFalse(is_provider_failure(KeyError("k")))


class TestRouter(RouterTestBase):
    async def test_local_mode_never_builds_cloud_providers(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available")
        models = {"local": FakeModel("local"), "cloud_a": FakeModel("cloud_a"), "cloud_b": FakeModel("cloud_b")}
        with self.install_models(models):
            router = build_router_for_task("structuring", self.client, CONFIG, llm_mode="local")
            models["local"].error = ConnectionError("ollama down")
            with self.assertRaises(ProviderUnavailableError):
                await router.ainvoke([HumanMessage(content="hi")])
        self.assertEqual(models["cloud_a"].calls, 0)
        self.assertEqual(models["cloud_b"].calls, 0)

    async def test_hybrid_falls_back_to_cloud_and_tags_provider(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available")
        models = {
            "local": FakeModel("local", error=ConnectionError("down")),
            "cloud_a": FakeModel("cloud_a"),
            "cloud_b": FakeModel("cloud_b")
        }
        with self.install_models(models):
            router = build_router_for_task("structuring", self.client, CONFIG, llm_mode="hybrid")
            response = await router.ainvoke([HumanMessage(content="hi")])
        self.assertEqual(response_to_text(response), "answer from cloud_a")
        self.assertEqual(extract_provider_name(response), "cloud_a")

    async def test_prompt_errors_do_not_consume_cloud_quota(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available")
        models = {
            "local": FakeModel("local", error=ValueError("bad prompt")),
            "cloud_a": FakeModel("cloud_a"),
            "cloud_b": FakeModel("cloud_b")
        }
        with self.install_models(models):
            router = build_router_for_task("structuring", self.client, CONFIG, llm_mode="hybrid")
            with self.assertRaises(ValueError):
                await router.ainvoke([HumanMessage(content="hi")])
        self.assertEqual(models["cloud_a"].calls, 0)

    async def test_breaker_trips_and_skips_failed_provider(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available")
        models = {
            "local": FakeModel("local", error=ConnectionError("down")),
            "cloud_a": FakeModel("cloud_a"),
            "cloud_b": FakeModel("cloud_b")
        }
        with self.install_models(models):
            router = build_router_for_task("structuring", self.client, CONFIG, llm_mode="hybrid")
            for _ in range(4):
                await router.ainvoke([HumanMessage(content="hi")])
        self.assertEqual(models["local"].calls, 2)
        self.assertEqual(models["cloud_a"].calls, 4)

    async def test_rate_limit_denial_does_not_count_as_failure(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available")
        model = FakeModel("t_provider")
        limiter = RedisRateLimiter(self.client, "t_provider", 1, 60)
        breaker = RedisCircuitBreaker(self.client, "t_provider", 2, 60)
        wrapped = wrap_model_with_circuit_breaker(model, "t_provider", limiter, breaker)
        await wrapped.ainvoke([HumanMessage(content="a")])
        for _ in range(3):
            with self.assertRaises(ProviderUnavailableError):
                await wrapped.ainvoke([HumanMessage(content="b")])
        self.assertFalse(await breaker.is_open())
        self.assertEqual(model.calls, 1)

    async def test_timeout_counts_as_provider_failure(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available")
        model = FakeModel("t_provider", delay=1.0)
        limiter = RedisRateLimiter(self.client, "t_provider", 10, 60)
        breaker = RedisCircuitBreaker(self.client, "t_provider", 1, 60)
        wrapped = wrap_model_with_circuit_breaker(model, "t_provider", limiter, breaker, timeout_seconds=0.05)
        with self.assertRaises(ProviderUnavailableError):
            await wrapped.ainvoke([HumanMessage(content="a")])
        self.assertTrue(await breaker.is_open())

    async def test_providers_without_api_keys_are_skipped(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available")
        with mock.patch.dict("os.environ", {}, clear=False):
            for name in ("TEST_KEY_A", "TEST_KEY_B"):
                import os
                os.environ.pop(name, None)
            config = {
                "providers": {"cloud_a": CONFIG["providers"]["cloud_a"]},
                "task_routing": {"structuring": ["cloud_a"]}
            }
            with self.assertRaises(ValueError):
                build_router_for_task("structuring", self.client, config, llm_mode="hybrid")

    async def test_router_status_reports_state(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available")
        breaker = RedisCircuitBreaker(self.client, "local", 1, 30)
        await breaker.record_failure()
        rows = await get_router_status(self.client, CONFIG, "structuring", "local")
        by_name = index_by(rows, "provider")
        self.assertEqual(by_name["local"]["state"], "breaker tripped")
        self.assertEqual(by_name["cloud_a"]["state"], "disabled")

    async def test_guard_store_outage_fails_open(self) -> None:
        model = FakeModel("t_provider")

        class BrokenLimiter:
            async def is_allowed(self) -> bool:
                raise redis_module.ConnectionError("down")

        class BrokenBreaker:
            async def is_open(self) -> bool:
                raise redis_module.ConnectionError("down")

            async def record_success(self) -> None:
                raise redis_module.ConnectionError("down")

            async def record_failure(self) -> None:
                raise redis_module.ConnectionError("down")

        wrapped = wrap_model_with_circuit_breaker(model, "t_provider", BrokenLimiter(), BrokenBreaker())
        response = await wrapped.ainvoke([HumanMessage(content="a")])
        self.assertEqual(response_to_text(response), "answer from t_provider")


class TestMessages(unittest.TestCase):
    def test_image_message_uses_dict_image_url(self) -> None:
        message = build_image_message("read this", b"\x89PNG", "image/png")
        blocks = message.content
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[1]["type"], "image_url")
        self.assertIsInstance(blocks[1]["image_url"], dict)
        self.assertTrue(blocks[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_confidence_footer(self) -> None:
        text, confidence = parse_confidence_footer("x = 2\nCONFIDENCE: 0.8")
        self.assertEqual(text, "x = 2")
        self.assertAlmostEqual(confidence, 0.8)
        text, confidence = parse_confidence_footer("no footer here")
        self.assertEqual(text, "no footer here")
        self.assertEqual(confidence, 0.5)
        _, percent = parse_confidence_footer("a\nCONFIDENCE: 85")
        self.assertAlmostEqual(percent, 0.85)

    def test_response_to_text_handles_blocks_and_strings(self) -> None:
        self.assertEqual(response_to_text("plain"), "plain")
        message = AIMessage(content=[{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])
        self.assertEqual(response_to_text(message), "one\ntwo")


    def test_router_yaml_loads_screen_ocr_task(self) -> None:
        from pathlib import Path
        import yaml
        config_path = Path("router.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file_handle:
                data = yaml.safe_load(file_handle)
            self.assertIn("screen_ocr", data.get("task_routing", {}))
            self.assertNotIn("board_ocr", data.get("task_routing", {}))


if __name__ == "__main__":
    unittest.main()
