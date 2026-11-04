"""
LLM Utilities

Wrapper functions for LLM API calls using LiteLLM.
Follows tau2-bench pattern for model routing and provider handling.
"""

from typing import List, Optional, Any, Dict, Tuple
import os
import time
import threading
from loguru import logger

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("LiteLLM not installed. LLM features will be limited.")


# Configure litellm
if LITELLM_AVAILABLE:
    litellm.set_verbose = False

# Import routes
try:
    from benchmark.utils.llm_routes import get_litellm_kwargs
except ImportError:
    get_litellm_kwargs = None


# Rate limit retry settings (tuned for parallel execution)
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_WAIT = 5
RATE_LIMIT_MAX_WAIT = 60
EMPTY_CHOICES_MAX_RETRIES = 2

# Route metrics collection (opt-in, used by comparison scripts)
# Uses threading.Event for thread-safe flag operations
_ROUTE_METRICS_EVENT = threading.Event()


def enable_route_metrics():
    """Enable route metrics collection for relay vs direct comparison."""
    _ROUTE_METRICS_EVENT.set()


def disable_route_metrics():
    """Disable route metrics collection."""
    _ROUTE_METRICS_EVENT.clear()


# Concurrency control: shared semaphore to limit in-flight LLM calls
# Set via set_concurrency_limit() before starting parallel workers
_LLM_CALL_SEMAPHORE: Optional[threading.Semaphore] = None


def set_concurrency_limit(max_concurrent_calls: int) -> None:
    """Set the maximum number of concurrent LLM calls across all threads.

    Call this once before dispatching parallel workers.
    Typical value: max_workers * 2 (e.g., 3 workers -> 6 concurrent calls).

    Args:
        max_concurrent_calls: Maximum in-flight LLM calls. 0 disables the limit.
    """
    global _LLM_CALL_SEMAPHORE
    if max_concurrent_calls > 0:
        _LLM_CALL_SEMAPHORE = threading.Semaphore(max_concurrent_calls)
        logger.info(f"[Concurrency] LLM call semaphore set to {max_concurrent_calls}")
    else:
        _LLM_CALL_SEMAPHORE = None
        logger.info("[Concurrency] LLM call semaphore disabled")


def _is_quota_exhausted_error(error: Exception) -> bool:
    """Check if the error is an account-level quota exhaustion (insufficient funds).

    Distinct from rate-limit (429) — quota exhaustion is non-recoverable until the
    account is topped up. Hitting one mid-sim should fail-fast the entire run rather
    than producing garbage assistant replies like "[Error: ... 用户额度不足 ...]"
    that pollute downstream eval/judge.
    """
    error_str = str(error)
    quota_markers = [
        "insufficient_user_quota",
        "insufficient_quota",
        "用户额度不足",
        "exceeded your current quota",
        "balance is insufficient",
        "balance is not enough",
    ]
    return any(m in error_str for m in quota_markers)


def _fail_fast_on_quota_exhausted(error: Exception) -> None:
    """Hard-kill the entire process if `error` indicates account quota exhaustion.

    Uses os._exit(2) to bypass try/except handlers in callers. Logs the offending
    error first so the operator can see why everything stopped.
    """
    if _is_quota_exhausted_error(error):
        import os
        logger.error(
            "[QUOTA EXHAUSTED] Account balance insufficient — killing the entire "
            f"process to avoid producing garbage replies. Error: {error}"
        )
        os._exit(2)


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if the error is a rate limit error (429) or service unavailable (503)"""
    error_str = str(error).lower()
    rate_limit_indicators = [
        "rate limit", "rate_limit", "ratelimit", "429",
        "too many requests", "quota exceeded",
        "resource_exhausted", "resource exhausted",
    ]
    for indicator in rate_limit_indicators:
        if indicator in error_str:
            return True
    return False


def _is_retryable_error(error: Exception) -> bool:
    """
    Check if the error is retryable (rate limit, overloaded, or temporary server error).
    
    Handles:
    - 429: Rate limit / Too Many Requests
    - 503: Service Unavailable / Model Overloaded
    - 500: Internal Server Error (temporary)
    - 502: Bad Gateway
    - 504: Gateway Timeout
    """
    error_str = str(error).lower()
    retryable_indicators = [
        # Rate limit errors
        "rate limit", "rate_limit", "ratelimit", "429",
        "too many requests", "quota exceeded", 
        "resource_exhausted", "resource exhausted",
        # Service unavailable / overloaded
        "503", "unavailable", "overloaded", "model is overloaded",
        "service unavailable", "temporarily unavailable",
        # Other server errors
        "500", "internal server error", "internalservererror",
        "502", "bad gateway",
        "504", "gateway timeout",
        "timeout", "timed out",
        # Gemini specific
        "geminiexception",
        # Relay/litellm response parsing errors (transient)
        "model_dump",
    ]
    for indicator in retryable_indicators:
        if indicator in error_str:
            return True
    return False


def _is_thinking_model(model: str) -> bool:
    """
    Check if the model is a "thinking model" that has special parameter requirements.
    
    Thinking models (like GPT-5, o1, o3, Gemini-3) often:
    - Don't support temperature != 1
    - Don't support top_p, presence_penalty, frequency_penalty
    - Use internal reasoning tokens before text output
    """
    model_lower = model.lower()
    thinking_model_patterns = [
        "gpt-5",
        "o1-",
        "o1",
        "o3-",
        "o3",
        "gemini-3",
        "gemini-2.5-pro",  # Also a reasoning model
        "gemini-2.5-flash",  # Also a reasoning model
        "-thinking",
        "-reason",
    ]
    for pattern in thinking_model_patterns:
        if pattern in model_lower:
            return True
    return False


def _sanitize_params_for_thinking_model(model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove or adjust parameters not supported by thinking models.
    
    GPT-5, o1, o3 series and similar models only support temperature=1
    and don't support top_p, presence_penalty, frequency_penalty.
    """
    sanitized = dict(kwargs)
    
    unsupported_params = [
        "temperature",
        "top_p",
        "presence_penalty", 
        "frequency_penalty",
        "logit_bias",
    ]
    
    removed_params = []
    for param in unsupported_params:
        if param in sanitized:
            removed_params.append(f"{param}={sanitized[param]}")
            del sanitized[param]
    
    if removed_params:
        logger.info(f"[Thinking Model] {model}: removed unsupported params: {', '.join(removed_params)}")
    
    return sanitized


def _consume_stream(stream_response, model: str) -> Any:
    """
    Consume a litellm streaming response and reassemble into a normal ModelResponse.

    This is used when force_stream=True to work around proxies that return
    broken non-streaming responses (e.g. ikuncode drops content for GPT-5).
    The caller sees no difference — they get back the same object type as
    a non-streaming call.
    """
    content_parts = []
    reasoning_parts = []
    finish_reason = "stop"
    response_id = None
    response_model = model
    usage_info = None

    for chunk in stream_response:
        if not chunk.choices:
            # Final chunk sometimes has usage only
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage
            continue

        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            content_parts.append(delta.content)

        # Collect reasoning_content from extended thinking models
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_parts.append(delta.reasoning_content)
        # Also check provider_specific_fields for reasoning
        psf = getattr(delta, "provider_specific_fields", None)
        if psf and isinstance(psf, dict) and psf.get("reasoning_content"):
            reasoning_parts.append(psf["reasoning_content"])

        fr = getattr(chunk.choices[0], "finish_reason", None)
        if fr:
            finish_reason = fr

        if hasattr(chunk, "id") and chunk.id:
            response_id = chunk.id
        if hasattr(chunk, "model") and chunk.model:
            response_model = chunk.model
        if hasattr(chunk, "usage") and chunk.usage:
            usage_info = chunk.usage

    assembled_content = "".join(content_parts)
    assembled_reasoning = "".join(reasoning_parts)

    # Build a ModelResponse identical to what non-streaming would return
    from litellm import ModelResponse

    message_dict = {"role": "assistant", "content": assembled_content}
    if assembled_reasoning:
        message_dict["reasoning_content"] = assembled_reasoning

    response = ModelResponse(
        id=response_id or "stream-assembled",
        model=response_model,
        choices=[
            {
                "index": 0,
                "message": message_dict,
                "finish_reason": finish_reason,
            }
        ],
    )

    # Attach usage if available
    if usage_info:
        response.usage = usage_info

    logger.debug(
        f"[Stream] Assembled {len(content_parts)} chunks, "
        f"{len(assembled_content)} chars, finish={finish_reason}"
    )
    return response


def _route_model(
    model: str,
    messages: List[dict],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[str, Dict[str, Any]]:
    """
    Route model using llm_routes configuration.

    This function:
    1. Looks up the model in llm_routes.yaml
    2. Gets provider, api_base, api_key configuration
    3. For gemini provider, converts model name to 'gemini/model-name' format
    4. Merges all kwargs for litellm.completion()
    5. Sanitizes params for thinking models (GPT-5, o1, o3, etc.)
    6. Injects proxy-configured HTTPHandler for direct Gemini routing

    Returns:
        Tuple of (resolved_model_name, merged_kwargs)
    """
    # Extract route_mode_override if passed by caller (e.g., comparison script)
    route_mode_override = kwargs.pop("_route_mode_override", None)

    # Build base kwargs
    base_kwargs = dict(kwargs)
    base_kwargs["messages"] = messages
    base_kwargs["temperature"] = temperature
    if max_tokens is not None:
        base_kwargs["max_tokens"] = max_tokens

    # If routes not available, return original
    if get_litellm_kwargs is None:
        resolved_model = model
        merged = base_kwargs
    else:
        try:
            # Get routing config (with optional route_mode override)
            routed = get_litellm_kwargs(
                model, verbose=True, route_mode_override=route_mode_override
            )

            # Get resolved model name (with provider prefix for gemini)
            resolved_model = routed.get("resolved_model") or routed.get("model") or model

            # Clean up routed kwargs (remove model-related keys)
            routed_clean = dict(routed)
            for k in ("model", "resolved_model"):
                routed_clean.pop(k, None)

            # Merge: base_kwargs + routed_clean
            merged = dict(base_kwargs)
            merged.update(routed_clean)

        except Exception as e:
            logger.warning(f"Route failed for {model}: {e}, using original")
            resolved_model = model
            merged = base_kwargs

    # Strip temperature if route config says no_temperature
    # (e.g. Claude with extended thinking rejects temperature entirely)
    if merged.pop("_no_temperature", False):
        merged.pop("temperature", None)

    # Sanitize parameters for thinking models (GPT-5, o1, o3, Gemini-3, etc.)
    preserve_sampling = merged.pop('_preserve_sampling_parameters', False)
    if _is_thinking_model(resolved_model) and not preserve_sampling:
        merged = _sanitize_params_for_thinking_model(resolved_model, merged)

    # Extract internal routing markers (must not reach litellm)
    proxy_url = merged.pop("_proxy_url", None)
    route_mode = merged.pop("_route_mode", "default")

    # Inject proxy-configured client for direct API access
    # - For OpenAI-compatible endpoints (api.openai.com etc): build openai.OpenAI client with proxy httpx
    # - For Gemini direct: use HTTPHandler wrapper as before
    if proxy_url and route_mode == "direct":
        try:
            api_base = merged.get("api_base", "")
            is_openai_endpoint = "openai.com" in api_base or resolved_model.startswith("openai/")
            if is_openai_endpoint:
                # Build openai.OpenAI client with proxy-configured httpx.Client
                import httpx
                from openai import OpenAI
                openai_key = merged.get("api_key")
                httpx_client = httpx.Client(proxy=proxy_url, timeout=httpx.Timeout(600.0, connect=10.0))
                openai_client = OpenAI(api_key=openai_key, base_url=api_base, http_client=httpx_client)
                merged["client"] = openai_client
                logger.info(f"[Route] Direct mode (OpenAI): injecting OpenAI client via proxy {proxy_url}")
            else:
                from benchmark.utils.proxy_client import ProxyClientManager
                proxy_manager = ProxyClientManager.get_instance()
                proxy_client = proxy_manager.get_proxy_client(proxy_url)
                merged["client"] = proxy_client
                logger.info(f"[Route] Direct mode (Gemini): injecting HTTPHandler via proxy {proxy_url}")
        except Exception as e:
            logger.warning(f"[Route] Failed to create proxy client: {e}, falling back to relay")
            route_mode = "relay"

    # Store route_mode as metadata for metrics collection (popped by _completion_with_retry)
    merged["_route_mode_meta"] = route_mode

    logger.debug(
        f"[Route] {model} -> {resolved_model}, mode={route_mode}, "
        f"api_base={merged.get('api_base', 'default(googleapis)')}"
    )
    return resolved_model, merged


def _completion_with_retry(
    model: str,
    max_retries: int = RATE_LIMIT_MAX_RETRIES,
    base_wait: float = RATE_LIMIT_BASE_WAIT,
    max_wait: float = RATE_LIMIT_MAX_WAIT,
    **kwargs
) -> Any:
    """
    Call litellm.completion with retry logic for transient errors.

    Handles:
    - 429: Rate limit errors with exponential backoff
    - 503: Service unavailable / model overloaded
    - 500/502/504: Server errors
    - Empty choices response

    Also collects route metrics when _COLLECT_ROUTE_METRICS is enabled.
    """
    if not LITELLM_AVAILABLE:
        raise RuntimeError("LiteLLM is required. Install with: pip install litellm")

    # Pop internal metadata (must not reach litellm.completion)
    budget = kwargs.pop("_first_pass_budget", None)
    evidence = kwargs.pop("_first_pass_stage", None)
    route_mode = kwargs.pop("_route_mode_meta", "unknown")
    force_stream = kwargs.pop("_force_stream", False)
    if evidence is not None:
        kwargs["num_retries"] = 0  # Count retries here, not invisibly inside LiteLLM.

    # force_stream: use streaming and reassemble into a normal ModelResponse.
    # Works around proxies that return broken non-streaming responses.
    if force_stream:
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {})["include_usage"] = True

    last_error = None
    empty_choices_count = 0
    needs_sleep = False
    sleep_time = 0.0

    for attempt in range(max_retries + 1):
        # Sleep from previous iteration (AFTER semaphore was released by finally)
        if needs_sleep:
            time.sleep(sleep_time)
            needs_sleep = False

        if budget is not None:
            if hasattr(budget, "prepare_request"):
                budget.prepare_request(kwargs.get("messages", []), kwargs.get("max_tokens", 16384))
            budget.consume()
        acquired = False
        attempt_record = None
        try:
            # Acquire semaphore before making the LLM call
            if _LLM_CALL_SEMAPHORE is not None:
                _LLM_CALL_SEMAPHORE.acquire()
                acquired = True

            call_start = time.time()
            if evidence is not None:
                evidence["attempts"] += 1
                evidence["effective_model"] = model
                evidence["effective_args"] = {k: kwargs[k] for k in
                    ("temperature", "max_tokens", "max_completion_tokens", "top_p", "seed",
                     "reasoning_effort", "reasoning", "response_format", "timeout") if k in kwargs}
                # Only retain the known safe provider switch, never arbitrary
                # extra_body values (which may contain credentials).
                if "enable_thinking" in (kwargs.get("extra_body") or {}):
                    evidence["effective_args"]["extra_body"] = {
                        "enable_thinking": kwargs["extra_body"]["enable_thinking"]}
                attempt_record = dict(attempt=evidence["attempts"],
                    started_at_unix=call_start, response_model=None, tokens=None,
                    finish_reason=None, error=None, outcome="started")
                evidence.setdefault("request_attempts", []).append(attempt_record)
            raw_response = completion(model=model, **kwargs)

            # If streaming, consume chunks and reassemble into ModelResponse
            if force_stream:
                response = _consume_stream(raw_response, model)
            else:
                response = raw_response
            call_latency_ms = (time.time() - call_start) * 1000

            if evidence is not None:
                usage = getattr(response, "usage", None)
                attempt_record.update(
                    request_id=getattr(response, "id", None),
                    response_model=getattr(response, "model", None),
                    tokens=usage.model_dump() if hasattr(usage, "model_dump") else None,
                    finish_reason=(getattr(response.choices[0], "finish_reason", None)
                                   if response.choices else None),
                    outcome="response" if response.choices else "empty_choices",
                    latency_ms=call_latency_ms)

            # Check for empty choices
            if not response.choices or len(response.choices) == 0:
                if evidence is not None and getattr(budget, "reject_invalid_responses", False):
                    raise ValueError("First-pass response has empty choices")
                empty_choices_count += 1
                if empty_choices_count <= EMPTY_CHOICES_MAX_RETRIES:
                    logger.warning(
                        f"[Empty Choices] API returned empty choices "
                        f"(attempt {empty_choices_count}/{EMPTY_CHOICES_MAX_RETRIES + 1}), "
                        f"waiting {base_wait}s..."
                    )
                    needs_sleep = True
                    sleep_time = base_wait
                    continue  # finally releases semaphore, then sleep at top of next iteration
                else:
                    logger.error(f"[Empty Choices] {empty_choices_count} consecutive empty responses")
                    raise RuntimeError(f"API returned empty choices after {empty_choices_count} attempts")

            # Record success metric
            if _ROUTE_METRICS_EVENT.is_set():
                _record_metric(
                    model=model,
                    route_mode=route_mode,
                    latency_ms=call_latency_ms,
                    success=True,
                    response=response,
                )

            return response

        except Exception as e:
            call_latency_ms = (time.time() - call_start) * 1000 if 'call_start' in locals() else 0

            if evidence is not None:
                evidence.setdefault("attempt_errors", []).append(type(e).__name__)
                if attempt_record is not None:
                    attempt_record.update(error=type(e).__name__, latency_ms=call_latency_ms)
                    if attempt_record["outcome"] == "started":
                        attempt_record["outcome"] = "error"

            # Record failure metric
            if _ROUTE_METRICS_EVENT.is_set():
                _record_metric(
                    model=model,
                    route_mode=route_mode,
                    latency_ms=call_latency_ms,
                    success=False,
                    error=e,
                )

            # Fail-fast on account-level quota exhaustion before classifying
            # as retryable. This covers user-sim and evaluator paths that don't
            # go through api_agent's exception handler.
            if evidence is not None:
                if "free-models-per-day" in str(e).lower():
                    evidence["unavailable_reason"] = "daily_free_limit"
                    raise
                if _is_quota_exhausted_error(e):
                    evidence["unavailable_reason"] = "quota_exhausted"
                    raise
            _fail_fast_on_quota_exhausted(e)

            if evidence is not None and "missing credentials" in str(e).lower():
                raise

            # Check if this is a retryable error (rate limit, 503, server errors, etc.)
            if _is_retryable_error(e):
                last_error = e
                if attempt < max_retries:
                    wait_time = min(base_wait * (2 ** attempt), max_wait)
                    error_type = "Rate Limit" if _is_rate_limit_error(e) else "Retryable Error"
                    logger.warning(
                        f"[{error_type}] (attempt {attempt + 1}/{max_retries + 1}), "
                        f"waiting {wait_time:.1f}s... Error: {e}"
                    )
                    needs_sleep = True
                    sleep_time = wait_time
                    continue  # finally releases semaphore, then sleep at top of next iteration
                else:
                    logger.error(f"[Retry Exhausted] Max retries ({max_retries + 1}) reached. Error: {e}")
                    raise e
            else:
                # Non-retryable error, raise immediately
                raise e

        finally:
            if attempt_record is not None and hasattr(budget, "record_attempt"):
                budget.record_attempt(attempt_record)
            # Always release semaphore after each attempt (success, error, or continue)
            if acquired and _LLM_CALL_SEMAPHORE is not None:
                _LLM_CALL_SEMAPHORE.release()
                acquired = False

    raise last_error if last_error else RuntimeError("Unexpected error in retry loop")


def _record_metric(
    model: str,
    route_mode: str,
    latency_ms: float,
    success: bool,
    response: Any = None,
    error: Optional[Exception] = None,
):
    """Record a call metric to the RouteMetricsCollector."""
    try:
        from benchmark.utils.route_metrics import RouteMetricsCollector

        input_tokens = 0
        output_tokens = 0
        if response and hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        RouteMetricsCollector.get_instance().record_call(
            model=model,
            route_mode=route_mode,
            latency_ms=latency_ms,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error_type=type(error).__name__ if error else None,
            error_msg=str(error)[:200] if error else None,
        )
    except Exception as e:
        logger.debug(f"Failed to record route metric: {e}")


def generate_chat_completion(
    model: str,
    messages: List[dict],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    return_raw_response: bool = False,
    **kwargs
) -> Any:
    """
    Generate a chat completion using LiteLLM with automatic routing.
    
    This function automatically:
    1. Routes the model using llm_routes.yaml configuration
    2. Handles gemini/openai provider differences
    3. Retries on rate limits and empty responses
    
    Args:
        model: Model name (e.g., "gpt-4o-mini", "gemini-2.5-flash")
        messages: List of message dicts with "role" and "content"
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        return_raw_response: If True, return the raw LiteLLM response object
        **kwargs: Additional arguments passed to LiteLLM
        
    Returns:
        Generated text content (str) or raw response object if return_raw_response=True
        
    Raises:
        RuntimeError: If LiteLLM is not available
        Exception: If API call fails after retries
    """
    if not LITELLM_AVAILABLE:
        raise RuntimeError("LiteLLM is required for LLM calls. Install with: pip install litellm")
    
    budget = kwargs.pop("_first_pass_budget", None)
    evidence = kwargs.pop("_first_pass_stage", None)
    # Route model to get proper provider config
    resolved_model, routed_kwargs = _route_model(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    # The release first-pass contract outranks retry defaults in a model route.
    if evidence is not None and getattr(budget, "reject_invalid_responses", False):
        routed_kwargs["max_retries"] = 0
        routed_kwargs["num_retries"] = 0
    
    try:
        # Call with retry logic
        response = _completion_with_retry(model=resolved_model, _first_pass_stage=evidence, _first_pass_budget=budget, **routed_kwargs)
        
        if evidence is not None:
            evidence["request_id"] = getattr(response, "id", None)
            evidence["response_model"] = getattr(response, "model", None)
            usage = getattr(response, "usage", None)
            evidence["tokens"] = usage.model_dump() if hasattr(usage, "model_dump") else None
            evidence["finish_reason"] = (getattr(response.choices[0], "finish_reason", None)
                                         if response.choices else None)
        # Return raw response if requested (for APIAgent compatibility)
        if return_raw_response:
            return response
        
        # Extract content from response
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], 'finish_reason', None)
            if evidence is not None and getattr(budget, "reject_invalid_responses", False) and finish_reason == "length":
                evidence["raw_response"] = content
                raise ValueError("First-pass response was truncated")
            
            # Log empty content with finish_reason for debugging
            if not content:
                logger.warning(f"API returned empty content. finish_reason={finish_reason}")
                # Traces already retain usage and finish reason; avoid dumping
                # provider reasoning or a very large response into experiment logs.
                if evidence is None:
                    logger.debug(f"Full response: {response}")
            
            return (content or "") if evidence is not None else (content.strip() if content else "")
        else:
            logger.warning("API returned empty choices.")
            return ""
        
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise


def generate_chat_completion_with_retry(
    model: str,
    messages: List[dict],
    temperature: float = 0.7,
    max_retries: int = 3,
    **kwargs
) -> str:
    """
    Generate chat completion with retry logic.
    
    Args:
        model: Model name
        messages: List of messages
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        **kwargs: Additional arguments
        
    Returns:
        Generated text content
    """
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _generate():
        return generate_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
    
    return _generate()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text (approximate).
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer selection
        
    Returns:
        Approximate token count
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough estimate
        return len(text) // 4
    except Exception:
        return len(text) // 4


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Estimate API cost based on token counts.
    
    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    # Approximate pricing (as of 2024)
    pricing = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    # Default pricing if model not found
    default_pricing = {"input": 0.01, "output": 0.03}
    
    # Find matching pricing
    model_pricing = default_pricing
    model_lower = model.lower()
    for key, value in pricing.items():
        if key in model_lower:
            model_pricing = value
            break
    
    cost = (input_tokens / 1000 * model_pricing["input"] + 
            output_tokens / 1000 * model_pricing["output"])
    
    return cost


def validate_model_access(model: str) -> bool:
    """
    Validate that we have access to a model.
    
    Args:
        model: Model name to validate
        
    Returns:
        True if model is accessible
    """
    if not LITELLM_AVAILABLE:
        return False
    
    try:
        # Try a minimal call
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        return True
    except Exception as e:
        logger.warning(f"Model '{model}' not accessible: {e}")
        return False
