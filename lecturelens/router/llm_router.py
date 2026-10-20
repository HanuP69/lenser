import asyncio
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import yaml
import redis.asyncio as redis
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from lecturelens.core.circuit_breaker import RedisRateLimiter, RedisCircuitBreaker
from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.router")

NON_PROVIDER_ERROR_TYPES: tuple[type, ...] = (ValueError, TypeError, KeyError, AttributeError)
NON_PROVIDER_STATUS_CODES: tuple[int, ...] = (400, 404, 413, 422)


class ProviderUnavailableError(RuntimeError):
    pass


def is_provider_failure(error: BaseException) -> bool:
    err_msg = str(error).lower()
    for signature in (
        "not found", "model", "unsupported", "image", "vision",
        "decommissioned", "deprecated", "retired", "does not exist",
        "rate limit", "quota", "overloaded", "capacity", "timeout",
        "not available", "no endpoints"
    ):
        if signature in err_msg:
            return True

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


def load_dotenv_into_environ() -> None:
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent.parent / ".env"):
        if candidate.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(candidate, override=False)
            except Exception:
                pass
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[len("export "):].strip()
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if v.startswith(("'", '"')) and len(v) >= 2 and v[0] == v[-1]:
                            v = v[1:-1]
                        else:
                            hash_pos = v.find("#")
                            if hash_pos != -1:
                                v = v[:hash_pos].strip()
                            v = v.strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            except Exception:
                pass


load_dotenv_into_environ()


def load_router_config(config_path: Path) -> dict[str, Any]:
    load_dotenv_into_environ()
    resolved_path = config_path
    if not resolved_path.exists():
        project_root_path = Path(__file__).resolve().parent.parent.parent / "router.yaml"
        if project_root_path.exists():
            resolved_path = project_root_path

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Router configuration file not found at {resolved_path}. "
            "Please ensure router.yaml exists in the project root."
        )

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


def extract_provider_api_keys(
    provider_name: str,
    provider_config: dict[str, Any]
) -> list[str]:
    env_var_name = provider_config.get("api_key_env", "OPENROUTER_API_KEY")
    keys: list[str] = []

    plural_name = f"{env_var_name}S"
    candidate_names = [
        plural_name,
        f"LECTURELENS_{plural_name}",
        env_var_name,
        f"LECTURELENS_{env_var_name}"
    ]

    if env_var_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        candidate_names.extend([
            "GEMINI_API_KEYS",
            "LECTURELENS_GEMINI_API_KEYS",
            "GEMINI_API_KEY",
            "LECTURELENS_GEMINI_API_KEY",
            "GOOGLE_API_KEYS",
            "LECTURELENS_GOOGLE_API_KEYS"
        ])

    for candidate_name in candidate_names:
        raw = os.environ.get(candidate_name, "").strip()
        if len(raw) > 0:
            tokens = re.split(r"[,;\n]+", raw)
            for token in tokens:
                cleaned = token.strip().strip("'\"")
                if len(cleaned) > 0 and cleaned not in keys:
                    keys.append(cleaned)
    return keys


class RotatingChatModel(Runnable):
    def __init__(self, models: list[Any], keys: list[str]) -> None:
        self.models: list[Any] = models
        self.keys: list[str] = keys
        self.current_index: int = 0
        self.key_count: int = len(models)
        self.lock: asyncio.Lock = asyncio.Lock()
        self.sync_lock: threading.Lock = threading.Lock()

    def invoke(self, input: Any, config: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        with self.sync_lock:
            idx = self.current_index % len(self.models)
            self.current_index = (self.current_index + 1) % len(self.models)

        last_error: Optional[Exception] = None
        for attempt in range(len(self.models)):
            active_idx = (idx + attempt) % len(self.models)
            active_model = self.models[active_idx]
            try:
                return active_model.invoke(input, config=config, **kwargs)
            except Exception as invocation_error:
                last_error = invocation_error
                error_str = str(invocation_error).lower()
                status_code = getattr(invocation_error, "status_code", None)
                if status_code in (429, 500, 502, 503, 504) or "rate limit" in error_str or "quota" in error_str:
                    logger.warning(
                        "key index %d hit rate limit/error (%s), rotating to next key (attempt %d/%d)",
                        active_idx, invocation_error, attempt + 1, len(self.models)
                    )
                    continue
                raise
        if last_error is not None:
            raise last_error

    async def ainvoke(self, input: Any, config: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        async with self.lock:
            idx = self.current_index % len(self.models)
            self.current_index = (self.current_index + 1) % len(self.models)

        last_error: Optional[Exception] = None
        for attempt in range(len(self.models)):
            active_idx = (idx + attempt) % len(self.models)
            active_model = self.models[active_idx]
            try:
                return await active_model.ainvoke(input, config=config, **kwargs)
            except Exception as invocation_error:
                last_error = invocation_error
                error_str = str(invocation_error).lower()
                status_code = getattr(invocation_error, "status_code", None)
                if status_code in (429, 500, 502, 503, 504) or "rate limit" in error_str or "quota" in error_str:
                    logger.warning(
                        "key index %d hit rate limit/error (%s), rotating to next key (attempt %d/%d)",
                        active_idx, invocation_error, attempt + 1, len(self.models)
                    )
                    continue
                raise
        if last_error is not None:
            raise last_error


RotatingChatOpenAI = RotatingChatModel


def create_base_chat_model(
    provider_name: str,
    provider_config: dict[str, Any],
    ollama_url: Optional[str] = None
) -> Optional[Any]:
    resolved_ollama_url = ollama_url or os.environ.get("LECTURELENS_OLLAMA_URL", "http://127.0.0.1:11434")
    provider_type = provider_config.get("type", "")
    model_name = provider_config.get("model", "")

    if provider_type == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            logger.warning("langchain_ollama missing, provider %s skipped", provider_name)
            return None
        configured_model = str(provider_config.get("model", "")).strip()
        # The environment wins over router.yaml, which is the same order the
        # openai_compatible and google_genai providers already use. The
        # router.yaml value used to win here, so LECTURELENS_LOCAL_MODEL and
        # OLLAMA_MODEL were silently ignored for local models.
        model_name = (
            os.environ.get("LECTURELENS_OLLAMA_MODEL")
            or os.environ.get("LECTURELENS_LOCAL_MODEL")
            or os.environ.get("OLLAMA_MODEL")
            or configured_model
        )
        return ChatOllama(
            base_url=resolved_ollama_url,
            model=model_name,
            num_ctx=int(provider_config.get("num_ctx", 8192)),
            temperature=float(provider_config.get("temperature", 0.2))
        )

    if provider_type == "openai_compatible":
        api_keys = extract_provider_api_keys(provider_name, provider_config)
        if len(api_keys) == 0:
            logger.info("provider %s skipped: %s not set", provider_name, provider_config.get("api_key_env", ""))
            return None
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            logger.warning("langchain_openai missing, provider %s skipped", provider_name)
            return None
        base_url = provider_config.get("base_url", "https://openrouter.ai/api/v1")
        model_name = (
            os.environ.get(f"LECTURELENS_{provider_name.upper()}_MODEL")
            or os.environ.get(f"{provider_name.upper()}_MODEL")
            or provider_config.get("model", "")
        )
        if len(api_keys) == 1:
            return ChatOpenAI(
                base_url=base_url,
                api_key=api_keys[0],
                model=model_name,
                max_retries=2
            )
        models = [
            ChatOpenAI(
                base_url=base_url,
                api_key=key,
                model=model_name,
                max_retries=1
            )
            for key in api_keys
        ]
        return RotatingChatModel(models, api_keys)

    if provider_type == "google_genai":
        api_keys = extract_provider_api_keys(provider_name, provider_config)
        if len(api_keys) == 0:
            logger.info("provider %s skipped: %s not set", provider_name, provider_config.get("api_key_env", ""))
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.warning("langchain_google_genai missing, provider %s skipped", provider_name)
            return None
        model_name = (
            os.environ.get("LECTURELENS_GOOGLE_MODEL")
            or os.environ.get("LECTURELENS_GEMINI_MODEL")
            or os.environ.get("GOOGLE_MODEL")
            or os.environ.get(f"LECTURELENS_{provider_name.upper()}_MODEL")
            or provider_config.get("model", "")
        )
        if len(api_keys) == 1:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_keys[0],
                max_retries=0
            )
        models = [
            ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                max_retries=0
            )
            for key in api_keys
        ]
        return RotatingChatModel(models, api_keys)

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
    ollama_url: Optional[str] = None,
    llm_mode: str = "local",
    timeout_seconds: Optional[float] = None
) -> Any:
    resolved_ollama_url = ollama_url or os.environ.get("LECTURELENS_OLLAMA_URL", "http://127.0.0.1:11434")
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
            ollama_url=resolved_ollama_url
        )
        if base_model is None:
            continue

        rate_limit_info = current_provider_config.get("rate_limit", {})
        breaker_info = current_provider_config.get("circuit_breaker", {})

        base_rpm = rate_limit_info.get("max_rpm", 60)
        key_count = getattr(base_model, "key_count", 1)
        effective_max_rpm = base_rpm * key_count

        rate_limiter = RedisRateLimiter(
            redis_client=redis_client,
            provider_name=provider_name,
            max_requests_per_window=effective_max_rpm,
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

        keys = extract_provider_api_keys(provider_name, provider_config)
        key_count = max(1, len(keys))
        effective_max_rpm = rate_limit_info.get("max_rpm", 60) * key_count
        limiter = RedisRateLimiter(
            redis_client, provider_name,
            effective_max_rpm, rate_limit_info.get("window_seconds", 60)
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
