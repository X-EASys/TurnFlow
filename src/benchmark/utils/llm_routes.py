"""
LLM Routes - Model routing configuration for the benchmark

This module provides a flexible way to configure different LLM providers
and models through a YAML configuration file.

Supported providers:
- openai: OpenAI API (including compatible endpoints)
- anthropic: Anthropic Claude API
- gemini: Google Gemini API
- local: Local models via vLLM/HuggingFace

Usage:
    from benchmark.utils.llm_routes import get_litellm_kwargs, get_model_config
    
    # Get LiteLLM kwargs for a model
    kwargs = get_litellm_kwargs("gpt-4o-mini")
    
    # Use with LiteLLM
    response = litellm.completion(model=model_name, messages=messages, **kwargs)
"""

import os
import threading
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


# Global routes cache (protected by lock for thread safety)
_ROUTES: Optional[Dict] = None
_ROUTES_LOCK = threading.Lock()


def _get_default_routes_path() -> str:
    """Get the default path for llm_routes.yaml"""
    # Check environment variable first
    env_path = os.getenv("BENCHMARK_LLM_ROUTES")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Check in the benchmark package directory
    package_dir = Path(__file__).parent
    local_path = package_dir / "llm_routes.yaml"
    if local_path.exists():
        return str(local_path)
    
    # Check in the project root
    project_root = package_dir.parent.parent.parent
    root_path = project_root / "llm_routes.yaml"
    if root_path.exists():
        return str(root_path)
    
    return str(local_path)  # Return default even if not exists


def _load_routes(config_path: Optional[str] = None, force_reload: bool = False) -> Dict:
    """Load routes from YAML file
    
    Args:
        config_path: Custom path to the routes YAML file. If None, uses default.
        force_reload: Force reload even if cached
    """
    global _ROUTES

    # If custom path provided, always load from it (don't use cache)
    if config_path is not None:
        path = config_path
    else:
        with _ROUTES_LOCK:
            if _ROUTES is not None and not force_reload:
                return _ROUTES
        path = _get_default_routes_path()

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                routes = yaml.safe_load(f) or {}
            logger.debug(f"Loaded LLM routes from: {path}")

            # Only cache if using default path
            if config_path is None:
                with _ROUTES_LOCK:
                    _ROUTES = routes
            return routes
        except Exception as e:
            logger.warning(f"Failed to load LLM routes from {path}: {e}")
            return {}
    else:
        logger.debug(f"LLM routes file not found: {path}, using defaults")
        return {}


def get_model_config(model_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the configuration for a specific model.
    
    Args:
        model_name: The model name/identifier
        config_path: Custom path to routes YAML file
        
    Returns:
        Dict with model configuration (provider, api_base, api_key, etc.)
    """
    routes = _load_routes(config_path)
    models = routes.get("models", {})
    return models.get(model_name, {})


def get_litellm_kwargs(
    model_name: str,
    config_path: Optional[str] = None,
    verbose: bool = True,
    route_mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get LiteLLM kwargs for a specific model.

    This function looks up the model configuration and returns
    kwargs suitable for passing to litellm.completion().

    Args:
        model_name: The model name/identifier (can include provider prefix like "gemini/")
        config_path: Custom path to routes YAML file
        verbose: Whether to print routing info
        route_mode_override: Override route_mode from config ("relay", "direct", or "default").
            Used by comparison scripts to force a specific routing method.

    Returns:
        Dict with LiteLLM kwargs (custom_llm_provider, api_base, api_key, etc.)
        Internal keys (consumed by llm_utils._route_model, not passed to litellm):
            _proxy_url: HTTP proxy URL for direct mode
            _route_mode: "relay", "direct", or "default"
    """
    routes = _load_routes(config_path)
    models = routes.get("models") or {}

    # Handle models that already have provider prefix (e.g., "gemini/model-name")
    # Strip the prefix and look up the base model name
    lookup_name = model_name
    prefix_provider = None
    if "/" in model_name:
        parts = model_name.split("/", 1)
        prefix_provider = parts[0].lower()
        lookup_name = parts[1]

    # First try the original name, then try the stripped name
    model_config = models.get(model_name, {})
    if not model_config and lookup_name != model_name:
        model_config = models.get(lookup_name, {})

    # Get provider (default to openai, or use prefix if provided)
    provider = (model_config.get("provider") or prefix_provider or "openai").strip().lower()

    # Determine route mode: override > config > default ("default")
    route_mode = route_mode_override or model_config.get("route_mode", "default")

    # Get API key: priority is yaml.api_key > environment variable
    api_key = model_config.get("api_key")
    if not api_key:
        # Check for custom env var name
        env_name = model_config.get("api_key_env")
        if env_name:
            api_key = os.getenv(env_name)
        else:
            # Default env vars by provider
            if provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # Build kwargs
    kwargs = {}

    # Set provider if not default openai
    if provider != "openai":
        kwargs["custom_llm_provider"] = provider

    # Handle model name format with provider prefix for litellm
    # If model already has provider prefix, don't add it again
    # model_name config allows route key to differ from the actual model name
    # sent to the API (e.g., "gemini-3.1-pro-preview-ikun" key sends "gemini-3.1-pro-preview")
    actual_model = model_config.get("model_name") or lookup_name
    resolved_model = model_name
    if provider == "gemini":
        if not actual_model.startswith("gemini/"):
            resolved_model = f"gemini/{actual_model}"
        else:
            resolved_model = actual_model
        kwargs["resolved_model"] = resolved_model
    elif provider == "openai":
        # For openai provider (including vLLM/OpenAI-compatible endpoints),
        # add openai/ prefix so litellm knows which provider to use
        if not actual_model.startswith("openai/"):
            resolved_model = f"openai/{actual_model}"
        else:
            resolved_model = actual_model
        kwargs["resolved_model"] = resolved_model

    # Handle route mode
    if route_mode == "direct":
        # Direct mode: route via SSH-tunnel proxy to official endpoint.
        # For OpenAI: set api_base from config (e.g. https://api.openai.com/v1)
        # For Gemini: omit api_base, litellm defaults to googleapis
        # In both cases: inject proxy URL so the HTTP request goes through SSH tunnel.
        if model_config.get("api_base"):
            kwargs["api_base"] = model_config["api_base"]
        proxy_config = routes.get("proxy", {}).get("ssh_tunnel", {})
        proxy_url = proxy_config.get("proxy_url", "http://127.0.0.1:7897")
        kwargs["_proxy_url"] = proxy_url
        kwargs["_route_mode"] = "direct"
    elif route_mode == "relay":
        # Relay mode: use third-party proxy as api_base
        if model_config.get("api_base"):
            kwargs["api_base"] = model_config["api_base"]
        kwargs["_route_mode"] = "relay"
    else:
        # Default mode: use api_base URL directly, no proxy/SSH tunnel
        if model_config.get("api_base"):
            kwargs["api_base"] = model_config["api_base"]
        kwargs["_route_mode"] = "default"

    # Set API key
    if api_key:
        kwargs["api_key"] = api_key

    # Some proxies block the default OpenAI SDK User-Agent header.
    # mask_user_agent: true in the route config overrides it.
    if model_config.get("mask_user_agent"):
        kwargs.setdefault("extra_headers", {})["User-Agent"] = "python-httpx/0.27"

    # Pass extra_body params to the API (e.g. cache control for proxies)
    if "extra_body" in model_config:
        kwargs.setdefault("extra_body", {}).update(model_config["extra_body"])

    # force_stream: proxy returns broken non-streaming responses for some models
    # (e.g. ikuncode drops content for GPT-5). Streaming works, so we consume
    # the stream internally and reassemble a normal ModelResponse.
    if model_config.get("force_stream"):
        kwargs["_force_stream"] = True

    # no_temperature: some endpoints (e.g. Claude with extended thinking)
    # reject the temperature parameter entirely. Strip it from kwargs.
    if model_config.get("no_temperature"):
        kwargs.pop("temperature", None)
        kwargs["_no_temperature"] = True

    # Explicit frozen routes may have validated sampling parameters that the
    # legacy model-name heuristic would otherwise discard.
    if model_config.get("preserve_sampling_parameters") is True:
        kwargs["_preserve_sampling_parameters"] = True

    # Add any extra kwargs from config
    for key in ["timeout", "max_retries", "temperature", "max_tokens"]:
        if key in model_config:
            kwargs[key] = model_config[key]

    if verbose:
        logger.info(
            f"[llm_routes] model={model_name} provider={provider} "
            f"route_mode={route_mode} "
            f"api_base={kwargs.get('api_base', 'default(googleapis)')} "
            f"api_key_present={'yes' if api_key else 'no'} "
            f"resolved_model={resolved_model}"
        )

    return kwargs


def list_available_models(config_path: Optional[str] = None) -> list:
    """
    List all models configured in the routes file.
    
    Args:
        config_path: Custom path to routes YAML file
    
    Returns:
        List of model names
    """
    routes = _load_routes(config_path)
    return list(routes.get("models", {}).keys())


def get_routes_path() -> str:
    """Get the current routes file path"""
    return _get_default_routes_path()


def reload_routes() -> None:
    """Force reload the routes from file"""
    global _ROUTES
    with _ROUTES_LOCK:
        _ROUTES = None
    _load_routes(force_reload=True)


# Convenience function for creating LiteLLM completion calls
def create_completion_kwargs(
    model_name: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **extra_kwargs
) -> Dict[str, Any]:
    """
    Create complete kwargs for a LiteLLM completion call.
    
    Args:
        model_name: Model identifier
        messages: List of messages
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **extra_kwargs: Additional LiteLLM kwargs
        
    Returns:
        Complete kwargs dict for litellm.completion()
    """
    route_kwargs = get_litellm_kwargs(model_name, verbose=False)
    
    # Use resolved model name if available (for Gemini)
    actual_model = route_kwargs.pop("resolved_model", model_name)
    
    kwargs = {
        "model": actual_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **route_kwargs,
        **extra_kwargs,
    }

    # Strip internal routing markers (consumed by llm_utils._route_model, not litellm)
    for internal_key in ("_proxy_url", "_route_mode", "_force_stream"):
        kwargs.pop(internal_key, None)

    return kwargs


def get_proxy_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the proxy/SSH tunnel configuration from routes YAML.

    Returns:
        Dict with proxy config (proxy_url, ssh_host, ssh_port, etc.)
    """
    routes = _load_routes(config_path)
    return routes.get("proxy", {}).get("ssh_tunnel", {})


def get_route_mode(model_name: str, config_path: Optional[str] = None) -> str:
    """
    Get the configured route mode for a model.

    Returns:
        "relay", "direct", or "default"
    """
    config = get_model_config(model_name, config_path)
    return config.get("route_mode", "default")
