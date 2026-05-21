"""Prompt-profile helpers for YetiDai runtime tuning."""

from __future__ import annotations

SMALL_PROFILE_NAME = "yeti_small"
LARGE_PROFILE_NAME = "yeti_large"
DEFAULT_SMALL_MODEL_NAME = "himalaya-bf16"
DEFAULT_SMALL_TOOL_ALLOWLIST = (
    "get_nepal_live_context,internet_search,fetch_url"
)


def is_small_model_name(model_name: str | None) -> bool:
    """Heuristic model-size detector for prompt/runtime tuning."""
    lowered = (model_name or "").strip().lower()
    if not lowered:
        return False
    size_hints = (
        "0.5b",
        "0_5b",
        "himalayagpt-0.5b",
        "himalaya-bf16",
        "himalaya-q8",
        "himalaya-q4",
        "tiny",
        "small",
    )
    return any(hint in lowered for hint in size_hints)


def resolve_prompt_profile(explicit_profile: str | None, model_name: str | None) -> str:
    """Return either 'yeti_small' or 'yeti_large'."""
    raw = (explicit_profile or "auto").strip().lower()
    if raw in {"small", SMALL_PROFILE_NAME}:
        return SMALL_PROFILE_NAME
    if raw in {"large", LARGE_PROFILE_NAME}:
        return LARGE_PROFILE_NAME
    return SMALL_PROFILE_NAME if is_small_model_name(model_name) else LARGE_PROFILE_NAME


def resolve_runtime_model(
    profile: str,
    configured_model: str | None,
    small_model_name: str | None = None,
) -> str:
    """Return runtime model name, forcing a stable small-model backend."""
    configured = (configured_model or "").strip()
    if profile == SMALL_PROFILE_NAME:
        forced = (small_model_name or DEFAULT_SMALL_MODEL_NAME).strip()
        return forced or DEFAULT_SMALL_MODEL_NAME
    return configured or "gpt-4.1-mini"


def build_small_runtime_prompt() -> str:
    """Ultra-short profile for tiny local models (e.g. 0.5B)."""
    return (
        "तिमी Yeti हौ। नेपाली (देवनागरी) मा छोटो र स्पष्ट उत्तर देऊ।\n"
        "हालको तथ्य अनुमान नगर; आवश्यक परे tool call गर।\n"
        "नेपाल: get_nepal_live_context, विश्व: internet_search, URL: fetch_url।\n"
        "समाचार चाहिँदा बढीमा ४ बुँदा, प्रत्येक १ लाइन; एउटै कुरा नदोहोर्याऊ।\n"
        "tool प्रयोग भएको छ भने अन्त्यमा स्रोत: राख।"
    )


def build_large_runtime_prompt() -> str:
    """Compact-but-rich profile for larger instruction-following models."""
    return (
        "You are Yeti (यती), a warm, witty Nepali-language Discord assistant "
        "built by Himalaya AI Research Lab (HARL).\n\n"
        "Reply only in Nepali (Devanagari). Keep small talk short and human. "
        "Use relaxed Nepali, not corporate chatbot language. Use ASCII digits "
        "only inside URLs; otherwise convert digits to Devanagari in the final "
        "answer.\n\n"
        "Core rules: never guess current Nepal facts from memory; use tools for "
        "anything that could have changed. Prefer NepalOSINT for Nepal facts. "
        "Use internet_search for non-Nepal current events. Use fetch_url for a "
        "specific pasted URL. For GitHub, never invent repo URLs; use "
        "analyze_github_repo or list_github_repos. For the Discord social "
        "feed, AI handles, or recent scraped tweets, use get_social_media_feed.\n\n"
        "Routing: small talk needs no tool. Nepal factual questions use "
        "get_nepal_live_context with the most specific intent "
        "(macro, trading, debt, government, parliament, general_news, who_is, "
        "history). Comparative Nepal/world questions may use multiple tool "
        "calls in parallel.\n\n"
        "Freshness: if NepalOSINT returns stale data, acknowledge it and use "
        "web results as primary. Never pad counts with duplicates. If the user "
        "corrects you, re-route instead of repeating the old answer.\n\n"
        "Tool policy: never announce intent, never refuse a factual question "
        "before calling a tool, and always cite tool-backed answers with a "
        "short स्रोत: line containing real URLs or endpoint names.\n\n"
        "Tone: warm, polite, sharp on facts. Short replies for small talk, "
        "concise factual replies elsewhere."
    )


def build_runtime_system_prompt(profile: str) -> str:
    """Return runtime prompt according to the selected profile."""
    if profile == SMALL_PROFILE_NAME:
        return build_small_runtime_prompt()
    return build_large_runtime_prompt()


def parse_stop_tokens(raw: str | None) -> list[str] | None:
    """Parse comma-separated stop tokens from env var input."""
    if not raw:
        return None
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    return tokens or None
