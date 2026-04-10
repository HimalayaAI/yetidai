import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


GENERAL_KEYWORDS = {
    "what is happening",
    "what's happening",
    "today",
    "tonight",
    "latest",
    "recent",
    "breaking",
    "brief",
    "briefing",
    "update",
    "updates",
    "summary",
    "news",
    "समाचार",
    "आज",
    "अहिले",
    "के भइरहेको",
    "के भैरहेको",
    "अपडेट",
    "सारांश",
}

MACRO_KEYWORDS = {
    "nrb",
    "macro",
    "economy",
    "economic",
    "inflation",
    "remittance",
    "reserves",
    "reserve",
    "trade",
    "imports",
    "exports",
    "fx",
    "foreign exchange",
    "current account",
    "tourism",
    "tourist",
    "migrant",
    "migration",
    "broad money",
    "narrow money",
    "treasury bill",
    "t-bill",
    "interbank",
    "bank rate",
    "मुद्रास्फीति",
    "रेमिट्यान्स",
    "विदेशी मुद्रा",
    "पर्यटन",
    "अर्थतन्त्र",
    "अर्थव्यवस्था",
    "निर्यात",
    "आयात",
}

GOVT_KEYWORDS = {
    "cabinet",
    "government decision",
    "govt decision",
    "announcement",
    "announcements",
    "ministry",
    "official notice",
    "सरकार",
    "sarkar",
    "sarkarle",
    "sarkar le",
    "मन्त्रालय",
    "निर्णय",
    "घोषणा",
    "सूचना",
    "मन्त्रिपरिषद्",
    "क्याबिनेट",
}

DEBT_KEYWORDS = {
    "debt",
    "loan",
    "loans",
    "rin",
    "rinn",
    "karja",
    "sovereign",
    "adb",
    "world bank",
    "imf",
    "financing",
    "public debt",
    "ऋण",
    "कर्जा",
    "सार्वजनिक ऋण",
    "loan amount",
}

PARLIAMENT_KEYWORDS = {
    "parliament",
    "hor",
    "na",
    "sansad",
    "bill",
    "bills",
    "session",
    "sessions",
    "house of representatives",
    "national assembly",
    "संसद",
    "प्रतिनिधि सभा",
    "राष्ट्रिय सभा",
    "विधेयक",
    "बैठक",
    "अधिवेशन",
}

TRADING_KEYWORDS = {
    "nepse",
    "share",
    "shares",
    "stock market",
    "market news",
    "stock",
    "stocks",
    "ipo",
    "hydro",
    "dividend",
    "bonus share",
    "right share",
    "market",
    "trading",
    "शेयर",
    "सेयर",
    "आईपिओ",
    "नेप्से",
    "बोनस",
    "लाभांश",
    "हकप्रद",
}

HISTORY_KEYWORDS = {
    "yesterday",
    "last week",
    "last month",
    "history",
    "historical",
    "previous",
    "गत हप्ता",
    "हिजो",
    "अघिल्लो",
    "इतिहास",
    "पहिले",
}

ROMANIZED_GENERAL_PATTERNS = (
    "aja ko news",
    "aaja ko news",
    "today ko news",
    "news bhanus",
    "news bhanus",
    "news summary",
    "daily briefing",
    "news briefing",
    "k bhairacha",
    "ke bhairacha",
    "k bhaira cha",
    "ke bhaira cha",
    "k bhayo",
    "ke bhayo",
)


@dataclass
class RoutePlan:
    use_nepalosint: bool
    intents: list[str] = field(default_factory=list)
    wants_history: bool = False
    history_start_date: str | None = None
    history_end_date: str | None = None
    history_category: str | None = None


def _contains_any(normalized_query: str, keywords: set[str]) -> bool:
    return any(keyword in normalized_query for keyword in keywords)


def _extract_history_range(query: str) -> tuple[str | None, str | None]:
    normalized = query.lower()
    today = date.today()

    iso_dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    if len(iso_dates) >= 2:
        return iso_dates[0], iso_dates[1]

    if "yesterday" in normalized or "हिजो" in normalized:
        target = today - timedelta(days=1)
        label = target.isoformat()
        return label, label

    if "last week" in normalized or "गत हप्ता" in normalized:
        return (today - timedelta(days=7)).isoformat(), today.isoformat()

    if "last month" in normalized:
        return (today - timedelta(days=30)).isoformat(), today.isoformat()

    return None, None


def route_query(query: str) -> RoutePlan:
    normalized = query.lower().strip()
    intents: list[str] = []

    has_romanized_general = any(pattern in normalized for pattern in ROMANIZED_GENERAL_PATTERNS)

    is_macro = _contains_any(normalized, MACRO_KEYWORDS)
    is_government = _contains_any(normalized, GOVT_KEYWORDS)
    is_debt = _contains_any(normalized, DEBT_KEYWORDS)
    is_parliament = _contains_any(normalized, PARLIAMENT_KEYWORDS)
    is_ticker_like = bool(re.search(r"\b[A-Z]{2,6}\b", query)) and not (is_macro or is_debt or is_parliament or is_government)
    is_trading = _contains_any(normalized, TRADING_KEYWORDS) or is_ticker_like
    is_general_news = _contains_any(normalized, GENERAL_KEYWORDS) or has_romanized_general
    wants_history = _contains_any(normalized, HISTORY_KEYWORDS)

    start_date, end_date = _extract_history_range(query) if wants_history else (None, None)

    if is_macro:
        intents.append("macro")
    if is_government:
        intents.append("government")
    if is_debt:
        intents.append("debt")
    if is_parliament:
        intents.append("parliament")
    if is_trading:
        intents.append("trading")
    if is_general_news or not intents:
        if is_general_news or wants_history:
            intents.append("general_news")

    wants_briefing = any(term in normalized for term in ("brief", "briefing", "summary", "सारांश", "समरी"))
    if is_general_news and wants_briefing and "government" not in intents:
        intents.append("government")

    deduped_intents = list(dict.fromkeys(intents))
    history_category = "economic" if is_macro or is_trading else None
    use_nepalosint = bool(deduped_intents or wants_history)

    return RoutePlan(
        use_nepalosint=use_nepalosint,
        intents=deduped_intents,
        wants_history=wants_history,
        history_start_date=start_date,
        history_end_date=end_date,
        history_category=history_category,
    )


async def _safe_fetch(name: str, coroutine: Any) -> tuple[str, Any, str | None]:
    try:
        return name, await coroutine, None
    except Exception as exc:  # pragma: no cover - defensive integration path
        return name, None, f"{type(exc).__name__}: {exc}"


async def fetch_context_bundle(client: Any, query: str, plan: RoutePlan) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    max_items = getattr(client, "max_context_items", 8)

    if "macro" in plan.intents:
        tasks["economy_snapshot"] = client.get_economy_snapshot()
        tasks["economy_bootstrap"] = client.get_dashboard_bootstrap("economy")

    if "government" in plan.intents:
        tasks["govt_decisions"] = client.get_govt_decisions_latest(limit=min(10, max_items), dedupe=True)
        tasks["announcements"] = client.get_announcements_summary(limit=min(10, max_items))

    if "debt" in plan.intents:
        tasks["debt_clock"] = client.get_debt_clock()
        tasks["debt_search"] = client.search_unified(query, limit=min(8, max_items))

    if "parliament" in plan.intents:
        tasks["verbatim_summary"] = client.get_verbatim_summary()
        tasks["parliament_bills"] = client.get_parliament_bills(limit=min(6, max_items))

    if "trading" in plan.intents:
        tasks["embedding_search"] = client.search_embeddings(query, top_k=min(8, max_items))
        tasks["trading_search"] = client.search_unified(query, limit=min(10, max_items))

    if "general_news" in plan.intents and "trading_search" not in tasks and "debt_search" not in tasks:
        tasks["general_search"] = client.search_unified(query, limit=min(8, max_items))

    if plan.wants_history and plan.history_start_date and plan.history_end_date:
        tasks["history"] = client.get_consolidated_history(
            start_date=plan.history_start_date,
            end_date=plan.history_end_date,
            limit=min(8, max_items),
            category=plan.history_category,
        )

    has_specific_intent = any(
        intent in plan.intents
        for intent in ("macro", "government", "debt", "parliament", "trading")
    )

    if "general_news" in plan.intents and not has_specific_intent:
        tasks["recent_news"] = client.get_consolidated_recent(hours=24, limit=min(12, max_items))

    results = await asyncio.gather(*[_safe_fetch(name, task) for name, task in tasks.items()])

    payloads: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, payload, error in results:
        if error:
            errors[name] = error
        elif payload is not None:
            payloads[name] = payload

    return {
        "query": query,
        "plan": plan,
        "payloads": payloads,
        "errors": errors,
    }
