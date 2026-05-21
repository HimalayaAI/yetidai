"""
Run a live prompt-suite evaluation for the Yeti small profile.

Usage:
    python tests/eval_yeti_small_live.py
    python tests/eval_yeti_small_live.py --suite tests/yeti_small_prompt_suite.json

Requires:
    - API_KEY (or OPENAI_API_KEY)
    - MODEL_NAME (or OPENAI_MODEL_NAME / OPENAI_MODEL)
Optionally:
    - BASE_URL / OPENAI_BASE_URL
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Ensure project root is on path for local script execution.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bot_helpers import (  # noqa: E402
    ensure_sources_line,
    extract_urls,
    normalize_digits,
    rewrite_sources_as_markdown,
)
from core.output_validator import validate_answer  # noqa: E402
from core.output_validator import build_fix_message  # noqa: E402
from core.preflight import plan_preflight  # noqa: E402
from core.prompt_profiles import (  # noqa: E402
    DEFAULT_SMALL_TOOL_ALLOWLIST,
    SMALL_PROFILE_NAME,
    build_runtime_system_prompt,
)
from core.tool_contracts import ToolContext  # noqa: E402
from core.tool_registry import get_registry  # noqa: E402
import tools.fetch.plugin as fetch_plugin  # noqa: E402
import tools.osint.plugin as osint_plugin  # noqa: E402
import tools.search.plugin as search_plugin  # noqa: E402


MAX_TOOL_ROUNDS = 4
DEFAULT_MAX_TOKENS = 260
DEFAULT_TEMP = 0.0
VALIDATOR_LEAK_MARKERS = (
    "मुख्य जवाफ देवनागरी",
    "उही तथ्य-सूचनालाई",
    "कृपया तुरुन्त पुनः लेख्नुहोस्",
)


@dataclass
class EvalCaseResult:
    case_id: str
    prompt: str
    final_text: str
    rounds: int
    tools_called: list[str]
    tool_used: bool
    citation_urls_count: int
    validator_issues: list[str]
    has_devanagari: bool
    leaked_validator_text: bool
    repeated_lines: bool
    pass_basic: bool
    error: str | None


def _has_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def _has_repeated_lines(text: str) -> bool:
    """Simple repetition detector: same normalized non-trivial line appears >= 2."""
    counts: dict[str, int] = {}
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip().lower()
        if len(line) < 18:
            continue
        counts[line] = counts.get(line, 0) + 1
        if counts[line] >= 2:
            return True
    return False


def _leaked_validator_text(text: str) -> bool:
    normalized = text or ""
    return any(marker in normalized for marker in VALIDATOR_LEAK_MARKERS)


def _select_small_tools() -> list[dict[str, Any]]:
    registry = get_registry()
    allowlist_raw = os.getenv("YETI_SMALL_TOOLS", DEFAULT_SMALL_TOOL_ALLOWLIST)
    allowlist = {
        item.strip()
        for item in allowlist_raw.split(",")
        if item.strip()
    }
    if not allowlist:
        allowlist = {
            item.strip()
            for item in DEFAULT_SMALL_TOOL_ALLOWLIST.split(",")
            if item.strip()
        }
    return [
        tool for tool in registry.openai_tools()
        if (tool.get("function", {}) or {}).get("name") in allowlist
    ]


async def _run_case(
    client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    enable_validator_retry: bool,
) -> EvalCaseResult:
    registry = get_registry()
    tools = _select_small_tools()
    allowed_tool_names = {
        (tool.get("function", {}) or {}).get("name")
        for tool in tools
        if (tool.get("function", {}) or {}).get("name")
    }
    today = datetime.date.today()
    system_prompt = build_runtime_system_prompt(SMALL_PROFILE_NAME)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"Today: {today.isoformat()} AD."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    rounds = 0
    tools_called: list[str] = []
    tool_used = False
    citation_urls: list[str] = []
    final_text = ""

    try:
        # Preflight mirror (same intent as bot.py): execute deterministic
        # route before first LLM turn so small models get tool data upfront.
        preflight = plan_preflight(prompt)
        if preflight is not None:
            pf_name, pf_args = preflight
            if pf_name in allowed_tool_names:
                pf_ctx = ToolContext(
                    query=prompt,
                    history=[],
                    llm_client=client,
                    channel_id=0,
                    user_id=0,
                )
                pf_result = await registry.execute(pf_name, pf_ctx, pf_args)
                pf_tc_id = f"preflight_{uuid.uuid4().hex[:8]}"
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": pf_tc_id,
                        "type": "function",
                        "function": {
                            "name": pf_name,
                            "arguments": json.dumps(pf_args, ensure_ascii=False),
                        },
                    }],
                })
                messages.append(pf_result.to_tool_message(pf_tc_id))
                tools_called.append(pf_name)
                if pf_result.success and (pf_result.content or "").strip():
                    tool_used = True
                citation_urls.extend(extract_urls(pf_result.content))

                # Optional preflight fallback (same decision rule as bot.py).
                if (
                    pf_result.trigger_fallback
                    and pf_result.fallback_tool
                    and pf_result.fallback_tool in allowed_tool_names
                ):
                    fb_args = pf_result.fallback_args or {}
                    fb_result = await registry.execute(pf_result.fallback_tool, pf_ctx, fb_args)
                    fb_tc_id = f"preflight_fb_{uuid.uuid4().hex[:8]}"
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": fb_tc_id,
                            "type": "function",
                            "function": {
                                "name": pf_result.fallback_tool,
                                "arguments": json.dumps(fb_args, ensure_ascii=False),
                            },
                        }],
                    })
                    messages.append(fb_result.to_tool_message(fb_tc_id))
                    tools_called.append(pf_result.fallback_tool)
                    if fb_result.success and (fb_result.content or "").strip():
                        tool_used = True
                    citation_urls.extend(extract_urls(fb_result.content))

        for _ in range(MAX_TOOL_ROUNDS):
            rounds += 1
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None) or []
            if choice.finish_reason != "tool_calls" or not tool_calls:
                final_text = (choice.message.content or "").strip()
                break

            messages.append({
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            ctx = ToolContext(
                query=prompt,
                history=[],
                llm_client=client,
                channel_id=0,
                user_id=0,
            )
            for tc in tool_calls:
                tool_name = tc.function.name
                tools_called.append(tool_name)
                args = json.loads(tc.function.arguments or "{}")
                result = await registry.execute(tool_name, ctx, args)
                messages.append(result.to_tool_message(tc.id))
                if result.success and (result.content or "").strip():
                    tool_used = True
                citation_urls.extend(extract_urls(result.content))

        # Mirror bot.py deterministic post-fixes for a fair "final output" eval.
        if final_text:
            final_text = normalize_digits(final_text)
            if tool_used:
                final_text = ensure_sources_line(final_text, citation_urls)
            final_text = rewrite_sources_as_markdown(final_text)

        validator_issues = validate_answer(
            final_text,
            tool_was_used=tool_used,
            github_tool_was_used=("analyze_github_repo" in tools_called or "list_github_repos" in tools_called),
        )
        # Mirror bot.py validator retry: ask the model to rewrite once with
        # explicit issue list when checks still fail.
        if final_text and validator_issues and enable_validator_retry:
            try:
                retry_messages = messages + [
                    {"role": "assistant", "content": final_text},
                    {"role": "system", "content": build_fix_message(validator_issues)},
                ]
                retry_resp = await client.chat.completions.create(
                    model=model_name,
                    messages=retry_messages,
                    tools=None,
                    tool_choice=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                retry_choice = retry_resp.choices[0]
                retry_text = (retry_choice.message.content or "").strip()
                if retry_text:
                    retry_text = normalize_digits(retry_text)
                    if tool_used:
                        retry_text = ensure_sources_line(retry_text, citation_urls)
                    retry_text = rewrite_sources_as_markdown(retry_text)
                    final_text = retry_text
                    validator_issues = validate_answer(
                        final_text,
                        tool_was_used=tool_used,
                        github_tool_was_used=(
                            "analyze_github_repo" in tools_called
                            or "list_github_repos" in tools_called
                        ),
                    )
            except Exception:
                pass

        has_devanagari = _has_devanagari(final_text)
        leaked = _leaked_validator_text(final_text)
        repeated = _has_repeated_lines(final_text)
        pass_basic = bool(final_text) and has_devanagari and not leaked and not repeated and not validator_issues
        return EvalCaseResult(
            case_id="",
            prompt=prompt,
            final_text=final_text,
            rounds=rounds,
            tools_called=tools_called,
            tool_used=tool_used,
            citation_urls_count=len(set(citation_urls)),
            validator_issues=validator_issues,
            has_devanagari=has_devanagari,
            leaked_validator_text=leaked,
            repeated_lines=repeated,
            pass_basic=pass_basic,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return EvalCaseResult(
            case_id="",
            prompt=prompt,
            final_text="",
            rounds=rounds,
            tools_called=tools_called,
            tool_used=tool_used,
            citation_urls_count=0,
            validator_issues=[],
            has_devanagari=False,
            leaked_validator_text=False,
            repeated_lines=False,
            pass_basic=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv(dotenv_path=".env")
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        is_local_base = bool(base_url) and (
            "localhost" in base_url or "127.0.0.1" in base_url
        )
        if is_local_base:
            api_key = "dummy-local-key"
            print("INFO: API key not set; using dummy key for local base_url.")
        else:
            print("ERROR: API_KEY/OPENAI_API_KEY missing. Add it to .env and rerun.")
            return 1

    model_name = (
        os.getenv("MODEL_NAME")
        or os.getenv("OPENAI_MODEL_NAME")
        or os.getenv("OPENAI_MODEL")
    )
    if not model_name:
        print("ERROR: MODEL_NAME/OPENAI_MODEL_NAME/OPENAI_MODEL missing in env.")
        return 1

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Register only the tool set we expect for yeti_small eval.
    osint_plugin.register()
    search_plugin.register()
    fetch_plugin.register()

    suite_path = Path(args.suite)
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    max_tokens = int(args.max_tokens)
    temperature = float(args.temperature)
    validator_retry_raw = (args.validator_retry or "auto").strip().lower()
    if validator_retry_raw == "on":
        enable_validator_retry = True
    elif validator_retry_raw == "off":
        enable_validator_retry = False
    else:
        # Mirror bot.py default for yeti_small.
        enable_validator_retry = (
            os.getenv("YETI_ENABLE_VALIDATOR_RETRY", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )

    print("=" * 88)
    print("Yeti Small Prompt-Suite Evaluation")
    print(f"model={model_name}")
    print(f"base_url={base_url or 'default'}")
    print(f"profile={SMALL_PROFILE_NAME}")
    print(f"temperature={temperature} max_tokens={max_tokens}")
    print(f"validator_retry={enable_validator_retry}")
    print(f"suite={suite_path}")
    print("=" * 88)

    all_results: list[EvalCaseResult] = []
    for idx, case in enumerate(data, start=1):
        case_id = str(case.get("id", f"case_{idx}"))
        prompt = str(case.get("prompt", "")).strip()
        if not prompt:
            print(f"[{idx}] {case_id}: skipped (empty prompt)")
            continue
        print(f"\n[{idx}] {case_id}")
        print(f"Q: {prompt}")
        result = await _run_case(
            client,
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_validator_retry=enable_validator_retry,
        )
        result.case_id = case_id
        all_results.append(result)
        if result.error:
            print(f"ERROR: {result.error}")
            continue

        print(f"PASS={result.pass_basic} rounds={result.rounds} tool_used={result.tool_used}")
        print(f"tools_called={result.tools_called}")
        print(
            f"signals: devanagari={result.has_devanagari} "
            f"validator_leak={result.leaked_validator_text} "
            f"repeated_lines={result.repeated_lines} "
            f"validator_issues={len(result.validator_issues)}"
        )
        preview = result.final_text.replace("\n", " ")[:320]
        print(f"A: {preview}")
        if result.validator_issues:
            print(f"validator_issues={result.validator_issues}")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.pass_basic)
    failed = total - passed
    print("\n" + "=" * 88)
    print(f"SUMMARY: total={total} pass={passed} fail={failed}")
    if failed:
        print("Failed cases:")
        for r in all_results:
            if r.pass_basic:
                continue
            why = []
            if r.error:
                why.append(f"error={r.error}")
            if r.leaked_validator_text:
                why.append("validator_leak")
            if r.repeated_lines:
                why.append("repetition")
            if r.validator_issues:
                why.append(f"validator={len(r.validator_issues)}")
            if not r.has_devanagari:
                why.append("non_nepali")
            print(f"- {r.case_id}: {', '.join(why) if why else 'unknown'}")
    print("=" * 88)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate yeti_small on a prompt suite.")
    parser.add_argument(
        "--suite",
        default=str(ROOT / "tests" / "yeti_small_prompt_suite.json"),
        help="Path to suite JSON file.",
    )
    parser.add_argument(
        "--max-tokens",
        default=str(DEFAULT_MAX_TOKENS),
        help="Max completion tokens for each turn.",
    )
    parser.add_argument(
        "--temperature",
        default=str(DEFAULT_TEMP),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--validator-retry",
        default="auto",
        choices=["auto", "on", "off"],
        help="Whether to run one LLM validator rewrite pass after deterministic fixes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(asyncio.run(main_async(parsed)))
