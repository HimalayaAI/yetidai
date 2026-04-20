"""
core/tool_registry.py — Central registry for all YetiDai tools.

Plugins call `get_registry().register(spec, handler)` at import time.
bot.py calls `registry.openai_tools()` to build the Sarvam `tools` array
and `registry.execute(name, ctx, args)` when the LLM returns tool_calls.
"""
from __future__ import annotations

import logging
from typing import Any

from core.tool_contracts import ToolSpec, ToolResult, ToolContext, ToolHandler

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Thread-safe registry for tool specs and their async handlers.

    Typical lifecycle:
        1. Plugin imports → calls `register(spec, handler)`
        2. Bot startup   → calls `openai_tools()` to get Sarvam schema
        3. Per-message   → calls `execute(name, ctx, args)` for each LLM tool_call
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}         # keyed by spec.name
        self._handlers: dict[str, ToolHandler] = {}   # keyed by spec.name

    # ── Registration ──────────────────────────────────────────────

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a tool spec + async handler. Overwrites if name exists."""
        if spec.name in self._specs:
            logger.warning("Tool '%s' re-registered (overwriting).", spec.name)
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler
        logger.info(
            "Registered tool: %s (id=%s, category=%s)",
            spec.name,
            spec.tool_id,
            spec.category.value,
        )

    def unregister(self, name: str) -> bool:
        """Remove a tool by function name. Returns True if it existed."""
        removed = name in self._specs
        self._specs.pop(name, None)
        self._handlers.pop(name, None)
        if removed:
            logger.info("Unregistered tool: %s", name)
        return removed

    # ── Queries ───────────────────────────────────────────────────

    def list_tools(self, *, enabled_only: bool = True) -> list[ToolSpec]:
        """Return all registered tool specs."""
        specs = list(self._specs.values())
        if enabled_only:
            specs = [s for s in specs if s.enabled]
        return specs

    def get_spec(self, name: str) -> ToolSpec | None:
        """Look up a single spec by function name."""
        return self._specs.get(name)

    def has(self, name: str) -> bool:
        return name in self._specs

    def openai_tools(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        """
        Build the `tools` array for `chat.completions(tools=...)`.

        Output format matches what Sarvam expects — see
        tests/test_sarvam_tool_calling_live.py for reference.
        """
        return [
            s.to_openai_tool()
            for s in self.list_tools(enabled_only=enabled_only)
        ]

    # ── Execution ─────────────────────────────────────────────────

    async def execute(
        self,
        name: str,
        ctx: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Run the handler for the named tool.

        Returns a ToolResult in all cases — never raises to the caller.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                tool_id=name,
                success=False,
                error=f"Unknown tool: '{name}'. Registered: {list(self._specs.keys())}",
            )

        spec = self._specs[name]
        if not spec.enabled:
            return ToolResult(
                tool_id=spec.tool_id,
                success=False,
                error=f"Tool '{name}' is currently disabled.",
            )

        try:
            return await handler(ctx, arguments)
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception.", name)
            return ToolResult(
                tool_id=spec.tool_id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )


# ── Global singleton ──────────────────────────────────────────────

_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Return (or create) the global ToolRegistry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
