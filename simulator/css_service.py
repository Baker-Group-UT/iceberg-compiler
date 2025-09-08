# simulator/css_service.py
from __future__ import annotations

import os
import threading
from typing import Optional, Any
from dotenv import load_dotenv
load_dotenv()

try:
    import cirq_superstaq as css  # type: ignore
except Exception as exc:  # pragma: no cover
    css = None  # defer error until first use
    _import_error = exc
else:
    _import_error = None


class SuperstaqService:
    """
    Thread-safe singleton for cirq_superstaq.Service.

    Usage:
        svc = SuperstaqService.get()  # reads SUPERSTAQ_API_KEY from env
        # or:
        svc = SuperstaqService.get(api_key="sk-...", base_url="https://...")

    Notes:
      - Does NOT log or expose the key.
      - Call SuperstaqService.reset() in tests if you need a fresh instance.
    """
    _instance: Optional["css.Service"] = None
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        *,
        api_key: Optional[str] = None,
        **service_kwargs: Any,
    ) -> "css.Service":
        """
        Return the singleton Service. Creates it on first call.

        Args:
            api_key: API key override. If None, uses env var SUPERSTAQ_API_KEY.
            **service_kwargs: forwarded to css.Service (e.g., base_url, timeout).

        Raises:
            RuntimeError: if cirq_superstaq is not installed or no API key found.
        """
        if css is None:  # pragma: no cover
            raise RuntimeError(
                "cirq-superstaq is not installed or failed to import."
            ) from _import_error

        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            if cls._instance is None:
                key = api_key or os.environ.get("SUPERSTAQ_API_KEY")
                if not key:
                    raise RuntimeError(
                        "SUPERSTAQ_API_KEY not set and no api_key provided."
                    )
                cls._instance = css.Service(api_key=key, **service_kwargs)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton (useful for tests)."""
        with cls._lock:
            cls._instance = None


__all__ = ["SuperstaqService"]