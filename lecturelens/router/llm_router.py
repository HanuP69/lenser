import asyncio
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import yaml
import redis.asyncio as redis
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from lecturelens.core.circuit_breaker import RedisRateLimiter, RedisCircuitBreaker
from lecturelens.core.logging_setup import get_logger
from lecturelens.router import usage as router_usage

logger = get_logger("lecturelens.router")

NON_PROVIDER_ERROR_TYPES: tuple[type, ...] = (ValueError, TypeError, KeyError, AttributeError)
NON_PROVIDER_STATUS_CODES: tuple[int, ...] = (400, 404, 413, 422)

# The words a provider uses when the request budget is used up.
RATE_LIMIT_SIGNATURES: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota",
    "requests per minute",
    "requests per second",
    "resource_exhausted",
)


def is_key_rotation_error(error: BaseException) -> bool:
    """Whether the next key has to be tried for this failure.

    The rotation exists to spread a list of keys over the requests, so
    every failure the *next* key could survive makes the chain move on: a
    budget that is used up, an outage, and a key that was invalid or was
    taken away. An error of the request itself (a wrong model name, a bad
    message) would fail on every key the same way, so it does not rotate.
    """
    if is_rate_limit_error(error):
        return True
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code >= 500:
            return True
        # A key that was refused does not refuse the other keys.
        return status_code in (401, 403, 408)
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    error_str = str(error).lower()
    return any(
        signature in error_str
        for signature in (
            "rate limit", "quota", "overloaded", "capacity",
            "timeout", "timed out", "connection", "unauthorized",
            "forbidden", "invalid api key", "api key"
        )
    )


def is_rate_limit_error(error: BaseException) -> bool:
    """Whether the provider refused the call because the budget was used up.

    A rate limit is not an outage, so it must not trip the circuit breaker.
    The sliding window rate limiter is the tool for that, and the breaker
    would take the best model away for the whole cooldown after a single
    answer that the limit was reached.
    """
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    message = str(error).lower()
    return any(signature in message for signature in RATE_LIMIT_SIGNATURES)


class ProviderUnavailableError(RuntimeError):
    pass


def is_configuration_error(error: BaseException) -> bool:
    """Whether the failure is a configuration problem rather than an outage.

    A wrong model name, a bad request or a payload that is too large are
    configuration problems. They must not trip the circuit breaker, because
    the provider is not unhealthy, but the chain still has to move on to the
    next provider instead of failing the whole task.
    """
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in NON_PROVIDER_STATUS_CODES
    return False


def is_provider_failure(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        # An explicit HTTP status is authoritative. 400, 404, 413 and 422
        # are client side configuration errors and must not trip the circuit
        # breaker. The keyword scan used to run first, so a 404 with the body
        # "model 'x' not found" was classified as a provider failure and
        # disabled the provider.
        if status_code in NON_PROVIDER_STATUS_CODES:
            return False
        return True

    err_msg = str(error).lower()
    for signature in (
        "not found", "model", "unsupported", "image", "vision",
        "decommissioned", "deprecated", "retired", "does not exist",
        "rate limit", "quota", "overloaded", "capacity", "timeout",
        "not available", "no endpoints"
    ):
        if signature in err_msg:
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
        # One lock protects the rotation counter. It used to be guarded by a
        # threading.Lock on the sync path and an asyncio.Lock on the async
        # path, so a concurrent sync and async call could read the same index
        # and use one key twice while skipping another.
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
                if is_key_rotation_error(invocation_error):
                    logger.warning(
                        "key index %d hit a retryable error (%s), rotating to next key (attempt %d/%d)",
                        active_idx, invocation_error, attempt + 1, len(self.models)
                    )
                    continue
                raise
        if last_error is not None:
            raise last_error

    async def ainvoke(self, input: Any, config: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        # The threading lock is held for a very short, non blocking critical
        # section, so it is safe to use from the event loop.
        with self.sync_lock:
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
                if is_key_rotation_error(invocation_error):
                    logger.warning(
                        "key index %d hit a retryable error (%s), rotating to next key (attempt %d/%d)",
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


def model_display_name(model: Any) -> str:
    """The name of the model behind a wrapper, as far as it tells."""
    inner = model
    models = getattr(inner, "models", None)
    if isinstance(models, list) and models:
        inner = models[0]
    for attribute in ("model_name", "model"):
        value = getattr(inner, attribute, None)
        if isinstance(value, str) and value:
            return value
    return ""


def wrap_model_with_circuit_breaker(
    model: BaseChatModel,
    provider_name: str,
    rate_limiter: RedisRateLimiter,
    circuit_breaker: RedisCircuitBreaker,
    timeout_seconds: Optional[float] = None,
    task_type: str = "",
    display_model: str = ""
) -> Any:
    async def invoke_guarded(messages: Any) -> Any:
        entry_time = time.monotonic()

        def refused(outcome: str) -> ProviderUnavailableError:
            # The provider was not even asked. The try is still written
            # down, because the burn of the chain of fallbacks is part
            # of the cost that the measurement counts.
            router_usage.record(
                task_type, provider_name, outcome,
                request_seconds=0.0,
                total_seconds=time.monotonic() - entry_time,
                model=display_model,
            )
            return ProviderUnavailableError(f"{outcome} for {provider_name}")

        try:
            breaker_open = await circuit_breaker.is_open()
            if breaker_open:
                raise refused("breaker_open")
            allowed = await rate_limiter.is_allowed()
            if not allowed:
                raise refused("limited")
        except ProviderUnavailableError:
            raise
        except Exception as guard_error:
            # The guard store is down. The provider is still tried, and
            # the call itself is what gets recorded.
            logger.warning("guard store unavailable for %s: %s", provider_name, guard_error)

        call_time = time.monotonic()
        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                result = await asyncio.wait_for(model.ainvoke(messages), timeout_seconds)
            else:
                result = await model.ainvoke(messages)
        except asyncio.CancelledError:
            raise
        except Exception as invocation_error:
            total_seconds = time.monotonic() - entry_time
            request_seconds = time.monotonic() - call_time
            router_usage.record(
                task_type, provider_name,
                "limited" if is_rate_limit_error(invocation_error) else "failed",
                request_seconds=request_seconds,
                total_seconds=total_seconds,
                model=display_model,
                error=str(invocation_error),
            )
            if is_configuration_error(invocation_error):
                # A misconfigured model is not an outage, so the circuit
                # breaker is left alone, but the chain has to move on to the
                # next provider. Raising the original error here would skip
                # every fallback and fail the whole task.
                raise ProviderUnavailableError(
                    f"{provider_name} is misconfigured: {invocation_error}"
                ) from invocation_error
            if is_provider_failure(invocation_error):
                if not is_rate_limit_error(invocation_error):
                    try:
                        await circuit_breaker.record_failure()
                    except Exception as guard_error:
                        logger.warning("failure not recorded for %s: %s", provider_name, guard_error)
                else:
                    # The budget was used up, the provider is still healthy.
                    logger.info(
                        "%s reached its request limit, moving on without tripping the breaker",
                        provider_name
                    )
                raise ProviderUnavailableError(
                    f"{provider_name} failed: {invocation_error}"
                ) from invocation_error
            raise

        try:
            await circuit_breaker.record_success()
        except Exception as guard_error:
            logger.warning("success not recorded for %s: %s", provider_name, guard_error)
        input_tokens, output_tokens = router_usage.tokens_from_response(result)
        router_usage.record(
            task_type, provider_name, "ok",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_seconds=time.monotonic() - call_time,
            total_seconds=time.monotonic() - entry_time,
            model=display_model,
        )
        return tag_response_with_provider(result, provider_name)

    return RunnableLambda(invoke_guarded)


def build_router_for_task(
    task_type: str,
    redis_client: redis.Redis,
    router_config: dict[str, Any],
    ollama_url: Optional[str] = None,
    llm_mode: str = "local",
    timeout_seconds: Optional[float] = None,
    local_model: Optional[str] = None
) -> Any:
    resolved_ollama_url = ollama_url or os.environ.get("LECTURELENS_OLLAMA_URL", "http://127.0.0.1:11434")
    if local_model:
        # The name of the local model is a setting of the machine, and the
        # settings read the .env file. The environment of the process does
        # not hold it, so the name in router.yaml would win and a machine
        # with another model would ask for a model it does not have.
        os.environ["LECTURELENS_LOCAL_MODEL"] = str(local_model)
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
            # The breaker is scoped to the task as well as the provider. It
            # used to be keyed by the provider alone, so a vision failure on a
            # shared provider also disabled the structuring and QA routes.
            provider_name=f"{task_type}:{provider_name}",
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
            timeout_seconds=timeout_val,
            task_type=task_type,
            display_model=model_display_name(base_model)
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
            redis_client, f"{task_type}:{provider_name}",
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
