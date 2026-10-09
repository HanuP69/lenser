import os
from pathlib import Path
from typing import Any, Optional
import yaml
import redis.asyncio as redis
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from lecturelens.core.circuit_breaker import RedisRateLimiter, RedisCircuitBreaker


def load_router_config(config_path: Path) -> dict[str, Any]:
    resolved_path = config_path
    if not resolved_path.exists():
        project_root_path = Path(__file__).resolve().parent.parent.parent / "router.yaml"
        if project_root_path.exists():
            resolved_path = project_root_path

    if not resolved_path.exists():
        return {
            "providers": {
                "gemma": {
                    "type": "ollama",
                    "model": "gemma4:e4b",
                    "rate_limit": {"max_rpm": 120, "window_seconds": 60},
                    "circuit_breaker": {"failure_threshold": 3, "cooldown_seconds": 30}
                }
            },
            "task_routing": {
                "board_ocr": ["gemma"],
                "structuring": ["gemma"],
                "qa": ["gemma"]
            }
        }

    with open(resolved_path, "r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)
    if data is None:
        return {}
    return data


def create_base_chat_model(
    provider_name: str,
    provider_config: dict[str, Any],
    ollama_url: str = "http://localhost:11434"
) -> Optional[BaseChatModel]:
    provider_type = provider_config.get("type", "")
    model_name = provider_config.get("model", "")

    if provider_type == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=ollama_url,
                model=model_name
            )
        except ImportError:
            return None

    if provider_type == "openai_compatible":
        try:
            from langchain_openai import ChatOpenAI
            base_url = provider_config.get("base_url", "https://openrouter.ai/api/v1")
            env_var_name = provider_config.get("api_key_env", "OPENROUTER_API_KEY")
            api_key = os.environ.get(env_var_name, "dummy_key")
            return ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name
            )
        except ImportError:
            return None

    if provider_type == "google_genai":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            env_var_name = provider_config.get("api_key_env", "GOOGLE_API_KEY")
            api_key = os.environ.get(env_var_name, "dummy_key")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key
            )
        except ImportError:
            return None

    return None


def wrap_model_with_circuit_breaker(
    model: BaseChatModel,
    provider_name: str,
    rate_limiter: RedisRateLimiter,
    circuit_breaker: RedisCircuitBreaker
) -> Any:
    async def invoke_guarded(messages: Any) -> Any:
        allowed = await rate_limiter.is_allowed()
        if not allowed:
            raise RuntimeError(f"Rate limit exceeded for provider: {provider_name}")

        open_status = await circuit_breaker.is_open()
        if open_status:
            raise RuntimeError(f"Circuit breaker open for provider: {provider_name}")

        try:
            result = await model.ainvoke(messages)
            await circuit_breaker.record_success()
            return result
        except Exception as invocation_error:
            await circuit_breaker.record_failure()
            raise invocation_error

    return RunnableLambda(invoke_guarded)


def build_router_for_task(
    task_type: str,
    redis_client: redis.Redis,
    router_config: dict[str, Any],
    ollama_url: str = "http://localhost:11434"
) -> Any:
    providers_config = router_config.get("providers", {})
    task_routing = router_config.get("task_routing", {})
    provider_chain_names = task_routing.get(task_type, [])

    if len(provider_chain_names) == 0:
        raise ValueError(f"No providers configured for task type: {task_type}")

    wrapped_models: list[Any] = []
    for provider_name in provider_chain_names:
        if provider_name not in providers_config:
            continue

        current_provider_config = providers_config[provider_name]
        base_model = create_base_chat_model(
            provider_name=provider_name,
            provider_config=current_provider_config,
            ollama_url=ollama_url
        )
        if base_model is None:
            continue

        rate_limit_info = current_provider_config.get("rate_limit", {})
        max_rpm = rate_limit_info.get("max_rpm", 60)
        window_seconds = rate_limit_info.get("window_seconds", 60)

        breaker_info = current_provider_config.get("circuit_breaker", {})
        failure_threshold = breaker_info.get("failure_threshold", 3)
        cooldown_seconds = breaker_info.get("cooldown_seconds", 30)

        rate_limiter = RedisRateLimiter(
            redis_client=redis_client,
            provider_name=provider_name,
            max_requests_per_window=max_rpm,
            window_seconds=window_seconds
        )
        circuit_breaker = RedisCircuitBreaker(
            redis_client=redis_client,
            provider_name=provider_name,
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds
        )

        wrapped = wrap_model_with_circuit_breaker(
            model=base_model,
            provider_name=provider_name,
            rate_limiter=rate_limiter,
            circuit_breaker=circuit_breaker
        )
        wrapped_models.append(wrapped)

    if len(wrapped_models) == 0:
        raise ValueError(f"Failed to initialize any providers for task: {task_type}")

    primary_model = wrapped_models[0]
    if len(wrapped_models) == 1:
        return primary_model

    fallback_models = []
    for index in range(1, len(wrapped_models)):
        fallback_models.append(wrapped_models[index])

    router_chain = primary_model.with_fallbacks(fallback_models)
    return router_chain
