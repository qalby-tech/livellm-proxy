"""OpenTelemetry tracing + thin logfire-shaped helpers.

Drop-in replacement for the bits of logfire we used:
- `tracer` / `span()` ─ open OTel spans following GenAI semantic conventions
- `info / warn / error / debug` ─ stdlib logging that also records on the
  current span via `add_event` / `record_exception`
- `configure_tracing()` ─ wires a `TracerProvider` with a single
  `BatchSpanProcessor(OTLPSpanExporter)` pointing at the cluster's
  otel-collector. Resource attributes come from a default set plus anything
  provided via `OTEL_RESOURCE_ATTRIBUTES`, so traces carry `service.name`,
  `deployment.environment.name`, etc.

Per-request *project* tagging is done via OTel baggage: the FastAPI middleware
reads the `X-Project` header (or a fallback) and stores it in baggage, which
the tracer copies onto each span as `project`. Tempo can then filter
`{ resource.service.name="livellm-proxy" && project="qalby" }`.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from opentelemetry import baggage, context as otel_context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

# Public: shared module-level logger. Keep `logfire`-style call sites cheap.
logger = logging.getLogger("livellm")

# Header / baggage key for cross-service project attribution.
PROJECT_HEADER = "x-project"
PROJECT_BAGGAGE_KEY = "project"

# Env var that gates whether prompt / completion text is recorded on spans.
_LOG_PROMPTS_ENV = "LOG_PROMPTS"


def log_prompts_enabled() -> bool:
    return os.environ.get(_LOG_PROMPTS_ENV, "false").lower() in ("1", "true", "yes")


# ----------------------------------------------------------------------------
# Tracer setup
# ----------------------------------------------------------------------------

_tracer: Optional[trace.Tracer] = None


def configure_tracing(
    service_name: str,
    otlp_endpoint: Optional[str],
    environment: Optional[str] = None,
    extra_resource_attrs: Optional[dict[str, Any]] = None,
) -> None:
    """Configure the global TracerProvider. Idempotent — safe to call once."""
    attrs: dict[str, Any] = {
        "service.name": service_name,
    }
    if environment:
        attrs["deployment.environment.name"] = environment
    if extra_resource_attrs:
        attrs.update(extra_resource_attrs)

    resource = Resource.create(attrs)
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        endpoint = otlp_endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )

    trace.set_tracer_provider(provider)
    global _tracer
    _tracer = trace.get_tracer("livellm-proxy")


def tracer() -> trace.Tracer:
    """Return the configured tracer (configure_tracing must have run)."""
    if _tracer is None:
        # Fallback so unit tests / scripts that never called configure_tracing
        # still get a working no-op tracer.
        return trace.get_tracer("livellm-proxy")
    return _tracer


# ----------------------------------------------------------------------------
# Span helper
# ----------------------------------------------------------------------------


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Open a span. Attribute values that are dicts/lists are JSON-stringified
    so the OTel SDK accepts them. Project from baggage is auto-attached."""
    safe_attrs = _sanitize_attrs(attrs)
    project = baggage.get_baggage(PROJECT_BAGGAGE_KEY)
    if project:
        safe_attrs[PROJECT_BAGGAGE_KEY] = str(project)
    with tracer().start_as_current_span(
        name, kind=SpanKind.CLIENT, attributes=safe_attrs
    ) as s:
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def set_attrs(span_obj: Optional[Span], **attrs: Any) -> None:
    """Set multiple attributes on a span, sanitizing non-primitive values."""
    if span_obj is None:
        span_obj = trace.get_current_span()
    for k, v in _sanitize_attrs(attrs).items():
        span_obj.set_attribute(k, v)


def _sanitize_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (str, bool, int, float)) for x in v
        ):
            out[k] = list(v)
        else:
            try:
                import json

                out[k] = json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                out[k] = str(v)
    return out


# ----------------------------------------------------------------------------
# Project / baggage helpers — used by the FastAPI middleware
# ----------------------------------------------------------------------------


def attach_project(project: str) -> object:
    """Attach a project name to the current OTel context via baggage.
    Returns a token that must be passed to `detach_project()` on cleanup."""
    ctx = baggage.set_baggage(PROJECT_BAGGAGE_KEY, project)
    return otel_context.attach(ctx)


def detach_project(token: object) -> None:
    otel_context.detach(token)  # type: ignore[arg-type]


def current_project() -> Optional[str]:
    val = baggage.get_baggage(PROJECT_BAGGAGE_KEY)
    return str(val) if val else None


# ----------------------------------------------------------------------------
# logfire-shaped logging shims — same surface, but routes through stdlib
# logging AND records an event on the current span (so logs show up next to
# their span in Tempo / Grafana).
# ----------------------------------------------------------------------------


def _log(level: int, msg: str, *args: Any, exc_info: Any = None, **_) -> None:
    logger.log(level, msg, *args, exc_info=exc_info)
    s = trace.get_current_span()
    if s is not None and s.is_recording():
        rendered = msg % args if args else msg
        s.add_event(
            "log",
            attributes={
                "log.severity": logging.getLevelName(level),
                "log.message": rendered[:4000],
            },
        )


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    _log(logging.INFO, msg, *args, **kwargs)


def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    _log(logging.DEBUG, msg, *args, **kwargs)


def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    _log(logging.WARNING, msg, *args, **kwargs)


# logfire spelled it `warn`; keep an alias.
def warn(msg: str, *args: Any, **kwargs: Any) -> None:
    warning(msg, *args, **kwargs)


def error(msg: str, *args: Any, exc_info: Any = None, **kwargs: Any) -> None:
    _log(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)
