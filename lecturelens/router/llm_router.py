import asyncio
import os
from pathlib import Path
from typing import Any, Optional

import yaml
import redis.asyncio as redis
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from lecturelens.core.circuit_breaker import RedisRateLimiter, RedisCircuitBreaker
from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.router")

NON_PROVIDER_ERROR_TYPES: tuple[type, ...] = (ValueError, TypeError, KeyError, AttributeError)
NON_PROVIDER_STATUS_CODES: tuple[int, ...] = (400, 404, 413, 422)


class ProviderUnavailableError(RuntimeError):
    pass


def is_provider_failure(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code in NON_PROVIDER_STATUS_CODES:
            return False
        return True
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    if isinstance(error, NON_PROVIDER_ERROR_TYPES):
        return False
    return True


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
                    "cloud": False,
                    "model": "gemma4:e4b",
                    "num_ctx": 8192,
                    "rate_limit": {"max_rpm": 120, "window_seconds": 60},
                    "circuit_breaker": {"failure_threshold": 3, "cooldown_seconds": 30}
                },
                "unorouter": {
                    "type": "openai_compatible",
                    "cloud": True,
                    "base_url": "https://api.unorouter.com/v1",
                    "model": "gemma-4-26b:free",
                    "api_key_env": "UNOROUTER_API_KEY",
                    "timeout_seconds": 120,
                    "rate_limit": {"max_rpm": 20, "window_seconds": 60},
                    "circuit_breaker": {"failure_threshold": 3, "cooldown_seconds": 60}
                }
            },
            "task_routing": {
                "board_ocr": ["unorouter", "gemma"],
                "structuring": ["unorouter", "gemma"],
                "qa": ["unorouter", "gemma"]
            }
        }

    with open(resolved_path, "r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)
    if data is None:
        return {}
    return data


def is_cloud_provider(provider_config: dict[str, Any]) -> bool:
    if "cloud" in provider_config:
        return bool(provider_config["cloud"])
    return provider_config.get("type", "") != "ollama"


def provider_allowed_for_mode(provider_config: dict[str, Any], llm_mode: str) -> bool:
    cloud_flag = is_cloud_provider(provider_config)
    if llm_mode == "local":
        return not cloud_flag
    if llm_mode == "cloud":
        return cloud_flag
    return True


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
        except ImportError:
            logger.warning("langchain_ollama missing, provider %s skipped", provider_name)
            return None
        model_name = os.environ.get("LECTURELENS_OLLAMA_MODEL") or provider_config.get("model", "")
        return ChatOllama(
            base_url=ollama_url,
            model=model_name,
            num_ctx=int(provider_config.get("num_ctx", 8192)),
            temperature=float(provider_config.get("temperature", 0.2))
        )

    if provider_type == "openai_compatible":
        env_var_name = provider_config.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.environ.get(env_var_name, "")
        if len(api_key) == 0:
            api_key = os.environ.get(f"LECTURELENS_{env_var_name}", "")
        if len(api_key) == 0:
            logger.info("provider %s skipped: %s not set", provider_name, env_var_name)
            return None
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            logger.warning("langchain_openai missing, provider %s skipped", provider_name)
            return None
        base_url = provider_config.get("base_url", "https://openrouter.ai/api/v1")
        model_name = os.environ.get(f"LECTURELENS_{provider_name.upper()}_MODEL") or provider_config.get("model", "")
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            max_retries=0
        )

    if provider_type == "google_genai":
        env_var_name = provider_config.get("api_key_env", "GOOGLE_API_KEY")
        api_key = os.environ.get(env_var_name, "")
        if len(api_key) == 0:
            api_key = os.environ.get(f"LECTURELENS_{env_var_name}", "")
        if len(api_key) == 0:
            logger.info("provider %s skipped: %s not set", provider_name, env_var_name)
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.warning("langchain_google_genai missing, provider %s skipped", provider_name)
            return None
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            max_retries=0
        )

    return None


def tag_response_with_provider(response: Any, provider_name: str) -> Any:
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict):
        additional["lecturelens_provider"] = provider_name
    return response


def wrap_model_with_circuit_breaker(
    model: BaseChatModel,
    provider_name: str,
    rate_limiter: RedisRateLimiter,
    circuit_breaker: RedisCircuitBreaker,
    timeout_seconds: Optional[float] = None
) -> Any:
    async def invoke_guarded(messages: Any) -> Any:
        try:
            breaker_open = await circuit_breaker.is_open()
            if breaker_open:
                raise ProviderUnavailableError(f"circuit breaker open for {provider_name}")
            allowed = await rate_limiter.is_allowed()
            if not allowed:
                raise ProviderUnavailableError(f"rate limit reached for {provider_name}")
        except ProviderUnavailableError:
            raise
        except Exception as guard_error:
            logger.warning("guard store unavailable for %s: %s", provider_name, guard_error)

        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                result = await asyncio.wait_for(model.ainvoke(messages), timeout_seconds)
            else:
                result = await model.ainvoke(messages)
        except asyncio.CancelledError:
            raise
        except Exception as invocation_error:
            if is_provider_failure(invocation_error):
                try:
                    await circuit_breaker.record_failure()
                except Exception as guard_error:
                    logger.warning("failure not recorded for %s: %s", provider_name, guard_error)
                raise ProviderUnavailableError(
                    f"{provider_name} failed: {invocation_error}"
                ) from invocation_error
            raise

        try:
            await circuit_breaker.record_success()
        except Exception as guard_error:
            logger.warning("success not recorded for %s: %s", provider_name, guard_error)
        return tag_response_with_provider(result, provider_name)

    return RunnableLambda(invoke_guarded)


def build_router_for_task(
    task_type: str,
    redis_client: redis.Redis,
    router_config: dict[str, Any],
    ollama_url: str = "http://localhost:11434",
    llm_mode: str = "local",
    timeout_seconds: Optional[float] = None
) -> Any:
    providers_config = router_config.get("providers", {})
    task_routing = router_config.get("task_routing", {})
    provider_chain_names = list(task_routing.get(task_type, []))

    if len(provider_chain_names) == 0:
        raise ValueError(f"No providers configured for task type: {task_type}")

    primary_override = os.environ.get("LECTURELENS_PRIMARY_PROVIDER")
    if primary_override is not None and primary_override in provider_chain_names:
        provider_chain_names = [primary_override] + [name for name in provider_chain_names if name != primary_override]

    wrapped_models: list[Any] = []
    for provider_name in provider_chain_names:
        if provider_name not in providers_config:
            continue

        current_provider_config = providers_config[provider_name]
        if not provider_allowed_for_mode(current_provider_config, llm_mode):
            continue

        base_model = create_base_chat_model(
            provider_name=provider_name,
            provider_config=current_provider_config,
            ollama_url=ollama_url
        )
        if base_model is None:
            continue

        rate_limit_info = current_provider_config.get("rate_limit", {})
        breaker_info = current_provider_config.get("circuit_breaker", {})

        rate_limiter = RedisRateLimiter(
            redis_client=redis_client,
            provider_name=provider_name,
            max_requests_per_window=rate_limit_info.get("max_rpm", 60),
            window_seconds=rate_limit_info.get("window_seconds", 60)
        )
        circuit_breaker = RedisCircuitBreaker(
            redis_client=redis_client,
            provider_name=provider_name,
            failure_threshold=breaker_info.get("failure_threshold", 3),
            cooldown_seconds=breaker_info.get("cooldown_seconds", 30)
        )

        provider_timeout = current_provider_config.get("timeout_seconds", timeout_seconds)
        timeout_val: Optional[float] = float(provider_timeout) if (provider_timeout is not None and float(provider_timeout) > 0) else None
        wrapped = wrap_model_with_circuit_breaker(
            model=base_model,
            provider_name=provider_name,
            rate_limiter=rate_limiter,
            circuit_breaker=circuit_breaker,
            timeout_seconds=timeout_val
        )
        wrapped_models.append(wrapped)

    if len(wrapped_models) == 0:
        raise ValueError(
            f"Failed to initialize any providers for task: {task_type} (llm_mode={llm_mode})"
        )

    primary_model = wrapped_models[0]
    if len(wrapped_models) == 1:
        return primary_model

    fallback_models: list[Any] = []
    for index in range(1, len(wrapped_models)):
        fallback_models.append(wrapped_models[index])

    return primary_model.with_fallbacks(
        fallback_models,
        exceptions_to_handle=(ProviderUnavailableError,)
    )


async def get_router_status(
    redis_client: redis.Redis,
    router_config: dict[str, Any],
    task_type: str,
    llm_mode: str = "local"
) -> list[dict[str, Any]]:
    providers_config = router_config.get("providers", {})
    chain_names = router_config.get("task_routing", {}).get(task_type, [])
    rows: list[dict[str, Any]] = []
    for provider_name in chain_names:
        if provider_name not in providers_config:
            continue
        provider_config = providers_config[provider_name]
        rate_limit_info = provider_config.get("rate_limit", {})
        breaker_info = provider_config.get("circuit_breaker", {})
        row: dict[str, Any] = {"provider": provider_name, "state": "OK", "calls": 0, "remaining": 0}

        if not provider_allowed_for_mode(provider_config, llm_mode):
            row["state"] = "disabled"
            rows.append(row)
            continue

        limiter = RedisRateLimiter(
            redis_client, provider_name,
            rate_limit_info.get("max_rpm", 60), rate_limit_info.get("window_seconds", 60)
        )
        breaker = RedisCircuitBreaker(
            redis_client, provider_name,
            breaker_info.get("failure_threshold", 3), breaker_info.get("cooldown_seconds", 30)
        )
        try:
            row["calls"] = await limiter.get_current_count()
            row["remaining"] = await limiter.get_remaining()
            if await breaker.is_open():
                row["state"] = "breaker tripped"
        except Exception:
            row["state"] = "unknown"
        rows.append(row)
    return rows
