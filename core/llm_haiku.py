"""
core/llm_haiku.py — Local Claude Haiku subprocess shim.

Drop-in replacement for `AsyncSarvamAI` that talks to the `claude` CLI
(Haiku 4.5, low effort). Exposes the same
`client.chat.completions(model=..., messages=..., tools=..., tool_choice=...)`
interface the Sarvam SDK offers, so `bot.py`'s multi-turn tool loop works
unchanged.

Tool calling is simulated via JSON-schema-constrained output: Haiku is asked
to emit either a `tool_calls` list OR a `final_answer` in a single JSON
object. The shim parses that into OpenAI-style objects with `finish_reason`,
`message.content`, and `message.tool_calls`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# ── JSON schema forced on Haiku output ────────────────────────────

def _build_tool_item_schema(tool_names: list[str]) -> dict[str, Any]:
    """Build the tool_calls[] item schema. When tool_names is non-empty,
    constrain `name` to that enum so the model can't hallucinate built-in
    tools (e.g. Claude Code's native WebSearch)."""
    name_schema: dict[str, Any] = {"type": "string"}
    if tool_names:
        name_schema["enum"] = tool_names
    return {
        "type": "object",
        "properties": {
            "name": name_schema,
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
    }


# Legacy unconstrained item schema (kept for reference / no-tools mode).
_TOOL_CALL_ITEM_SCHEMA: dict[str, Any] = _build_tool_item_schema([])

HAIKU_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "One short sentence (internal) explaining whether a tool is "
                "needed. Not shown to the user."
            ),
        },
        "tool_calls": {
            "type": "array",
            "description": (
                "Tools to invoke before answering. Leave empty when you can "
                "answer directly (greetings, chit-chat, already-cached facts)."
            ),
            "items": _TOOL_CALL_ITEM_SCHEMA,
        },
        "final_answer": {
            "type": "string",
            "description": (
                "Final Nepali (Devanagari) reply to the user. MUST be empty "
                "string when tool_calls is non-empty."
            ),
        },
    },
    "required": ["reasoning", "tool_calls", "final_answer"],
    "additionalProperties": False,
}

# Strict retry schema: at least one tool call, no final answer. Used when we
# need to force Haiku off a premature MODE B for a clearly factual query.
HAIKU_FORCE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "minItems": 1,
            "items": _TOOL_CALL_ITEM_SCHEMA,
        },
        "final_answer": {"type": "string", "enum": [""]},
    },
    "required": ["reasoning", "tool_calls", "final_answer"],
    "additionalProperties": False,
}


# ── Fake OpenAI/Sarvam response objects ───────────────────────────


class _ToolFn:
    def __init__(self, name: str, arguments_json: str) -> None:
        self.name = name
        self.arguments = arguments_json


class _ToolCall:
    def __init__(self, id_: str, name: str, args: dict[str, Any]) -> None:
        self.id = id_
        self.type = "function"
        self.function = _ToolFn(name, json.dumps(args, ensure_ascii=False))


class _Message:
    def __init__(self, content: str, tool_calls: list[_ToolCall]) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, choices: list[_Choice]) -> None:
        self.choices = choices


# ── Rendering helpers ─────────────────────────────────────────────


def _render_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""
    lines: list[str] = ["# AVAILABLE TOOLS", ""]
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {})
        lines.append(f"## {fn['name']}")
        lines.append(fn["description"])
        lines.append("")
        props = params.get("properties", {})
        if props:
            lines.append("Parameters:")
            required_set = set(params.get("required", []))
            for pname, pinfo in props.items():
                req = "required" if pname in required_set else "optional"
                extras = []
                if pinfo.get("enum"):
                    extras.append(f"enum={pinfo['enum']}")
                if pinfo.get("examples"):
                    extras.append(f"examples={pinfo['examples']}")
                extra_str = f" [{', '.join(extras)}]" if extras else ""
                lines.append(
                    f"  - {pname} ({pinfo.get('type', 'string')}, {req}){extra_str}: "
                    f"{pinfo.get('description', '')}"
                )
        lines.append("")
    return "\n".join(lines)


def _render_transcript(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Split messages into (system_text, user_transcript)."""
    system_parts: list[str] = []
    transcript: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            transcript.append(f"USER: {content}")
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                serialized = []
                for tc in tcs:
                    fn = tc["function"]
                    args = fn["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    serialized.append({"name": fn["name"], "arguments": args})
                transcript.append(
                    "ASSISTANT (called tools): "
                    + json.dumps(serialized, ensure_ascii=False)
                )
            if content:
                transcript.append(f"ASSISTANT: {content}")
        elif role == "tool":
            tcid = m.get("tool_call_id", "")
            transcript.append(f"TOOL_RESULT[{tcid}]: {content}")
    return "\n\n".join(system_parts), "\n\n".join(transcript)


# Heuristic markers for "this question needs a tool call". Matched against the
# latest user message. Used to nudge Haiku back on track when it picks MODE B
# for a clearly factual question. Covers Nepali + Romanized + English.
_FACTUAL_MARKERS: tuple[str, ...] = (
    # Nepali who-is
    "को हुनुहुन्छ", "को हो", "को हुन्", "नाम के",
    # Nepali macro/news/govt
    "मुद्रास्फीति", "महंगाई", "मूल्यवृद्धि", "रेमिट्यान्स", "विदेशी मुद्रा",
    "सञ्चिति", "ऋण", "सरकार", "प्रधानमन्त्री", "मन्त्री", "NEPSE", "नेप्से",
    "शेयर", "आयात", "निर्यात", "समाचार", "भइरहेको", "भैरहेको", "घोषणा",
    "बजेट", "विधेयक", "संसद", "पर्यटन", "जित्यो", "जीत्यो", "विजेता",
    # Romanized Nepali (common Discord style)
    "kasle", "jityo", "k bhako", "k bhayo", "ko ho", "ko hun",
    "hijo", "gaeko hapta", "aghillo mahina", "abako",
    # English factual question words
    "who is", "who are", "who won", "winner", "who was",
    "what is", "what was", "when did", "when was",
    "current ", "latest ", "recent ",
    # English Nepal topics
    "inflation", "remittance", "nepse", "debt", "gdp", "minister",
    "parliament", "reserves", "trade balance", "news",
    # English world topics (Bucket 3)
    "champions league", "world cup", "premier league", "uefa", "fifa",
    "olympics", "nba", "president of", "prime minister of", "ceo of",
)


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _has_tool_result(messages: list[dict[str, Any]]) -> bool:
    return any(m.get("role") == "tool" for m in messages)


def _looks_factual(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _FACTUAL_MARKERS)


NUDGE_SUFFIX = """

# RETRY NUDGE (CRITICAL)
Your previous attempt picked MODE B for a factual question without calling any
tool. That was WRONG. This user question clearly requires fresh data — you
MUST emit MODE A with a tool_call this turn. Pick `get_nepal_live_context` for
Nepal macro/NEPSE/debt/parliament/govt/news topics, or `internet_search` for
identifying people (ministers, politicians, CEOs) or world facts.
Do NOT output final_answer. Only output tool_calls."""


OUTPUT_CONTRACT = """# OUTPUT CONTRACT (STRICT)
Respond with ONE JSON object matching the supplied schema. Two modes:

MODE A — call tools (REQUIRED for factual/current/Nepal-specific questions):
  {"reasoning": "<why tools are needed>",
   "tool_calls": [{"name": "<tool_name>", "arguments": {...}}],
   "final_answer": ""}

MODE B — answer directly (ONLY for greetings, chit-chat, identity, or
answering AFTER tool results have been provided above):
  {"reasoning": "<why no tools needed, or how tool data is used>",
   "tool_calls": [],
   "final_answer": "<complete Nepali Devanagari reply>"}

# DECISION SHORTCUT
- If the transcript contains NO prior TOOL_RESULT and the user asks about
  current Nepal data, ministers, macro numbers, or recent events → MODE A.
- If the transcript already contains a TOOL_RESULT → MODE B (write the
  Nepali answer using that data, add `स्रोत:` line).
- Greetings / "who are you" / chit-chat → MODE B, no tools.

NEVER emit both tool_calls AND final_answer. NEVER wrap the JSON in
markdown fences. Output the JSON object and nothing else."""


# ── Public client ─────────────────────────────────────────────────


class HaikuChatCompletions:
    def __init__(self, client: "HaikuClient") -> None:
        self._client = client

    async def __call__(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> _Response:
        return await self._client._run(messages=messages, tools=tools or [])


class HaikuChat:
    def __init__(self, client: "HaikuClient") -> None:
        self.completions = HaikuChatCompletions(client)


class HaikuClient:
    """Local Haiku (via `claude` CLI) mimicking the Sarvam async client."""

    def __init__(
        self,
        *,
        model: str = "haiku",
        effort: str = "low",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.chat = HaikuChat(self)

    async def _run(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> _Response:
        sys_text, transcript = _render_transcript(messages)
        tools_block = _render_tools(tools)
        base_system = "\n\n".join(
            part for part in (sys_text, tools_block, OUTPUT_CONTRACT) if part
        )

        tool_names = [t["function"]["name"] for t in tools if t.get("function")]
        item_schema = _build_tool_item_schema(tool_names)

        default_schema: dict[str, Any] = {
            **HAIKU_RESPONSE_SCHEMA,
            "properties": {
                **HAIKU_RESPONSE_SCHEMA["properties"],
                "tool_calls": {
                    **HAIKU_RESPONSE_SCHEMA["properties"]["tool_calls"],
                    "items": item_schema,
                },
            },
        }
        force_tool_schema: dict[str, Any] = {
            **HAIKU_FORCE_TOOL_SCHEMA,
            "properties": {
                **HAIKU_FORCE_TOOL_SCHEMA["properties"],
                "tool_calls": {
                    **HAIKU_FORCE_TOOL_SCHEMA["properties"]["tool_calls"],
                    "items": item_schema,
                },
            },
        }

        resp = await self._invoke(base_system, transcript, default_schema)

        # Retry nudge: if Haiku chose MODE B for a clearly factual first-turn
        # question, force it to MODE A via a stricter schema (minItems:1).
        # Weak/low-effort models default to "I don't have info" without this.
        needs_nudge = (
            resp.choices[0].finish_reason != "tool_calls"
            and tools
            and not _has_tool_result(messages)
            and _looks_factual(_latest_user_message(messages))
        )
        if needs_nudge:
            logger.info("Haiku picked MODE B for factual query; retrying with force-tool schema.")
            resp = await self._invoke(
                base_system + NUDGE_SUFFIX,
                transcript,
                force_tool_schema,
            )

        return resp

    async def _invoke(
        self,
        full_system: str,
        transcript: str,
        schema: dict[str, Any],
    ) -> _Response:

        # Filter CLAUDECODE so we don't trip the nested-session guard when
        # this shim runs inside another Claude Code session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        # We omit `--bare` so the CLI can use the user's OAuth session
        # (Max plan) — `--bare` would force ANTHROPIC_API_KEY only.
        # `--system-prompt` replaces the default Claude Code system prompt,
        # so Yeti's rules aren't mixed with Claude Code's.
        # `--no-session-persistence` avoids writing sessions to disk.
        cmd = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            "--system-prompt",
            full_system,
            transcript or "(begin)",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("Haiku subprocess timed out")

        if proc.returncode != 0:
            raise RuntimeError(
                f"Haiku CLI failed ({proc.returncode}): "
                f"{stderr_b.decode('utf-8', 'replace')[:500]}"
            )

        raw = stdout_b.decode("utf-8", "replace").strip()

        # `claude -p --output-format json` returns an envelope. When
        # `--json-schema` is set, the validated object lives under
        # `structured_output` (and `result` is empty). Fall back to `result`
        # for plain text mode.
        parsed: dict[str, Any] | None = None
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            outer = None

        if isinstance(outer, dict):
            if outer.get("is_error"):
                raise RuntimeError(
                    f"Haiku API error: {outer.get('result') or outer}"
                )
            so = outer.get("structured_output")
            if isinstance(so, dict):
                parsed = so
            elif outer.get("result"):
                try:
                    maybe = json.loads(outer["result"])
                    if isinstance(maybe, dict):
                        parsed = maybe
                except json.JSONDecodeError:
                    parsed = {
                        "reasoning": "",
                        "tool_calls": [],
                        "final_answer": outer["result"],
                    }

        if parsed is None:
            logger.warning("Haiku returned non-JSON payload; treating as text.")
            parsed = {"reasoning": "", "tool_calls": [], "final_answer": raw}

        tool_calls_raw = parsed.get("tool_calls") or []
        final = parsed.get("final_answer") or ""

        if tool_calls_raw:
            tcs = [
                _ToolCall(
                    f"call_{uuid.uuid4().hex[:8]}",
                    tc.get("name", ""),
                    tc.get("arguments") or {},
                )
                for tc in tool_calls_raw
                if isinstance(tc, dict) and tc.get("name")
            ]
            if tcs:
                return _Response([_Choice(_Message("", tcs), "tool_calls")])

        return _Response([_Choice(_Message(final, []), "stop")])
