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
import sys
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


def configure_mlflow_tracing(
    tracking_uri: str,
    experiment_name: Optional[str] = None,
) -> None:
    """Enable MLflow's native pydantic-ai tracing integration.

    Unlike the generic OTLP export in `configure_tracing()` (which targets a
    collector/Tempo via the global TracerProvider), MLflow Tracing runs its own
    OpenTelemetry pipeline and ships traces straight to the MLflow Tracking
    Server. `mlflow.pydantic_ai.autolog()` monkey-patches pydantic-ai's
    `Agent` / `InstrumentedModel` / tool / MCP calls and auto-sets
    `instrument=True`, so every agent run is captured — prompts, model params,
    tool runs, usage — and rendered in the MLflow Tracing UI with no per-call
    wiring. The two pipelines are independent and compose freely.

    `mlflow.set_experiment()` creates the experiment if it does not yet exist.
    mlflow is imported lazily so it stays an optional runtime cost when unused.

    Note: autolog records prompt/response content into MLflow regardless of the
    `LOG_PROMPTS` gate (which only governs the OTLP/pydantic-ai spans).
    """
    # Tracing must never be able to take down the proxy. `set_experiment()` is a
    # synchronous call to the tracking server, so if MLflow is unreachable at
    # startup it would raise here and crash the worker. Degrade gracefully: log
    # and continue without tracing. (Per-request trace export is already async /
    # best-effort via MLFLOW_ENABLE_ASYNC_TRACE_LOGGING, on by default.)
    try:
        # MLflow independently reads OTEL_EXPORTER_OTLP_ENDPOINT and tries to
        # dual-export traces + metrics to the OTel collector through a gRPC OTLP
        # exporter that isn't installed (and a payload OTLP rejects: span_type is
        # None). That raises inside tracer-provider init and silently degrades
        # EVERY MLflow span to a no-op — no traces are recorded, with no error.
        # `configure_tracing()` has already captured the endpoint for the proxy's
        # own exporter, so hide it from MLflow here and let MLflow keep its
        # default *isolated* provider that ships only to its tracking server.
        for _otlp_var in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        ):
            os.environ.pop(_otlp_var, None)

        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        mlflow.pydantic_ai.autolog()
    except Exception as e:
        warning(
            "MLflow tracing setup failed (tracking_uri=%s); continuing without "
            "it. Error: %s",
            tracking_uri,
            e,
        )
        return

    global _mlflow_tracing_enabled
    _mlflow_tracing_enabled = True
    info(
        "MLflow pydantic-ai tracing enabled (tracking_uri=%s, experiment=%s)",
        tracking_uri,
        experiment_name or "<default>",
    )


# Set once MLflow tracing is successfully configured. Gates `mlflow_span()` so
# non-pydantic-ai paths only create spans when MLflow is actually active.
_mlflow_tracing_enabled: bool = False


@contextmanager
def mlflow_span(
    name: str,
    span_type: Optional[str] = None,
    inputs: Optional[dict[str, Any]] = None,
):
    """Open an MLflow span for code that MLflow's autolog doesn't cover (TTS / ASR).

    Yields the span (so callers can `set_outputs`) when MLflow tracing is active,
    otherwise yields ``None`` and does nothing — so the audio paths never hard-
    depend on MLflow being configured. Never raises into the caller.
    """
    if not _mlflow_tracing_enabled:
        yield None
        return
    try:
        import mlflow
    except Exception:
        yield None
        return
    # Open the span defensively. MLflow's span machinery touches the tracking
    # server on close (export), and start_span itself can raise — neither must
    # ever propagate into the audio path. On any failure we degrade to a no-op
    # span (yield None); callers already guard with `if span is not None`.
    cm = span = None
    try:
        cm = mlflow.start_span(name=name, span_type=span_type)
        span = cm.__enter__()
        if inputs:
            try:
                span.set_inputs(inputs)
            except Exception:
                pass
    except Exception as e:
        debug("MLflow start_span failed; tracing this block as a no-op: %s", e)
        if cm is not None:
            try:
                cm.__exit__(*sys.exc_info())
            except Exception:
                pass
        yield None
        return
    try:
        yield span
    except Exception:
        # Record the error on the span, then re-raise so the caller's control
        # flow is unchanged. Span-close errors are swallowed.
        try:
            cm.__exit__(*sys.exc_info())
        except Exception:
            pass
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception as e:
            debug("MLflow span close failed (ignored): %s", e)


def configure_pydantic_ai_instrumentation() -> None:
    """Turn on pydantic-ai's native OTel GenAI instrumentation.

    pydantic-ai already follows the OpenTelemetry GenAI semantic convention
    (v2): it emits a `chat <model>` span with `gen_ai.input.messages`,
    `gen_ai.output.messages`, `gen_ai.system_instructions`, plus tool-call
    parts and usage metrics. Without this call, no content is recorded and
    Tempo / Grafana / Phoenix / Logfire UIs show empty trace details.

    Whether to include message content is gated by `LOG_PROMPTS` — the same
    env var the rest of this module honours.
    """
    from pydantic_ai import Agent, InstrumentationSettings

    settings = InstrumentationSettings(
        include_content=log_prompts_enabled(),
        include_binary_content=log_prompts_enabled(),
        version=2,
    )
    Agent.instrument_all(settings)


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
