from __future__ import annotations

from typing import Any


def _truncate(text: str | None, limit: int = 220) -> str:
    if not text:
        return ""
    compact = " ".join(text.replace("\n", " ").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _pick_metrics(metrics: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    lowered = query.lower()
    targeted_keywords = {
        "inflation": ("inflation",),
        "मुद्रास्फीति": ("inflation",),
        "remittance": ("remittance",),
        "रेमिट्यान्स": ("remittance",),
        "reserve": ("reserve", "fx reserves"),
        "विदेशी मुद्रा": ("reserve", "fx reserves"),
        "trade": ("exports", "imports", "trade balance", "current account"),
        "आयात": ("imports",),
        "निर्यात": ("exports",),
        "tourism": ("tourist",),
        "पर्यटन": ("tourist",),
        "migrant": ("permit",),
        "bank": ("bank", "interbank", "broad money", "narrow money"),
        "debt": ("debt",),
    }

    matched_terms: set[str] = set()
    for needle, labels in targeted_keywords.items():
        if needle in lowered:
            matched_terms.update(labels)

    if not matched_terms:
        return metrics[:4]

    selected = [
        metric for metric in metrics
        if any(term in (metric.get("label") or "").lower() for term in matched_terms)
    ]
    return selected[:4] if selected else metrics[:4]


def _format_metric(metric: dict[str, Any]) -> str:
    value = metric.get("display_value") or metric.get("raw_value")
    change_pct = metric.get("change_pct")
    if change_pct is None:
        return f"- {metric.get('label')}: {value}"
    sign = "+" if change_pct >= 0 else ""
    return f"- {metric.get('label')}: {value} ({sign}{change_pct:.2f}% vs अघिल्लो महिना)"


def _append_block(blocks: list[str], title: str, lines: list[str], limit: int) -> bool:
    content = "\n".join([title, *lines]).strip()
    candidate = ("\n\n".join(blocks + [content])).strip()
    if len(candidate) > limit:
        return False
    blocks.append(content)
    return True


def build_context_brief(bundle: dict[str, Any], max_chars: int = 1800) -> str | None:
    payloads = bundle.get("payloads", {})
    errors = bundle.get("errors", {})
    query = bundle.get("query", "")
    plan = bundle.get("plan")
    blocks: list[str] = []
    sources: list[str] = []
    lowered_query = query.lower()
    broad_news_query = any(term in lowered_query for term in ("के भइरहेको", "के भैरहेको", "what is happening", "latest", "recent", "समाचार"))

    history_items = payloads.get("history", {}).get("items", [])
    if history_items and plan and getattr(plan, "wants_history", False):
        lines = [
            f"- {item.get('canonical_headline')} — {item.get('source_name')}"
            for item in history_items[:5]
        ]
        if _append_block(blocks, "Historical news", lines, max_chars):
            for item in history_items[:5]:
                if item.get("source_name"):
                    sources.append(item["source_name"])

    recent_news = payloads.get("recent_news")
    if recent_news and broad_news_query:
        items = recent_news[:5] if isinstance(recent_news, list) else []
        lines = [
            f"- {item.get('canonical_headline')} — {item.get('source_name')}"
            for item in items
        ]
        if lines and _append_block(blocks, "Relevant news", lines, max_chars):
            for item in items:
                if item.get("source_name"):
                    sources.append(item["source_name"])

    economy_snapshot = payloads.get("economy_snapshot")
    if economy_snapshot:
        relevant_metrics: list[dict[str, Any]] = []
        for section in economy_snapshot.get("sections", {}).values():
            relevant_metrics.extend(section.get("metrics", []))
        chosen = _pick_metrics(relevant_metrics, query)
        lines = [
            f"अवधि: {economy_snapshot.get('as_of_label')}",
            *[_format_metric(metric) for metric in chosen],
        ]
        if _append_block(blocks, "Macro snapshot", lines, max_chars):
            sources.append("NRB / NepalOSINT macro snapshot")

    debt_clock = payloads.get("debt_clock")
    if debt_clock:
        lines = [
            f"- कुल सार्वजनिक ऋण: NPR {debt_clock.get('debt_now_npr', 0):,.0f}",
            f"- बाह्य ऋण: NPR {debt_clock.get('external_debt_npr', 0):,.0f}",
            f"- आन्तरिक ऋण: NPR {debt_clock.get('domestic_debt_npr', 0):,.0f}",
            f"- Debt/GDP: {debt_clock.get('debt_gdp_pct', 0):.2f}%",
            f"- सन्दर्भ अवधि: {debt_clock.get('debt_as_of_label') or debt_clock.get('updated_label')}",
        ]
        if _append_block(blocks, "Debt snapshot", lines, max_chars):
            sources.append("PDMO / NepalOSINT debt clock")

    govt_decisions = payloads.get("govt_decisions", {}).get("items", [])
    if govt_decisions:
        lines = [
            f"- {item.get('title')} ({item.get('status')}) — {item.get('office')}"
            for item in govt_decisions[:3]
        ]
        if _append_block(blocks, "Recent government decisions", lines, max_chars):
            for item in govt_decisions[:3]:
                if item.get("sourceName"):
                    sources.append(item["sourceName"])

    announcements = payloads.get("announcements", {}).get("latest", [])
    if announcements:
        lines = [
            f"- {item.get('title')} — {item.get('source_name')}"
            for item in announcements[:2]
        ]
        if _append_block(blocks, "Official announcements", lines, max_chars):
            for item in announcements[:2]:
                if item.get("source_name"):
                    sources.append(item["source_name"])

    verbatim_summary = payloads.get("verbatim_summary")
    if verbatim_summary:
        sessions = verbatim_summary.get("recent_sessions", [])
        if sessions:
            latest = sessions[0]
            summary = _truncate(latest.get("session_summary"), 360)
            bills = latest.get("bills_discussed", [])[:2]
            lines = [
                f"- पछिल्लो बैठक: {latest.get('title_ne') or latest.get('session_date')}",
                f"- सार: {summary}",
            ]
            if bills:
                lines.append(f"- मुख्य विधेयक/विषय: {', '.join(bills)}")
            if _append_block(blocks, "Parliament update", lines, max_chars):
                sources.append("Federal Parliament / NepalOSINT verbatim summary")

    parliament_bills = payloads.get("parliament_bills", {}).get("items", [])
    if parliament_bills:
        lines = [
            f"- {item.get('title_en')} ({item.get('status')})"
            for item in parliament_bills[:2]
        ]
        if _append_block(blocks, "Tracked bills", lines, max_chars):
            sources.append("Federal Parliament bills")

    if history_items and not (plan and getattr(plan, "wants_history", False)):
        lines = [
            f"- {item.get('canonical_headline')} — {item.get('source_name')}"
            for item in history_items[:4]
        ]
        if _append_block(blocks, "Historical news", lines, max_chars):
            for item in history_items[:4]:
                if item.get("source_name"):
                    sources.append(item["source_name"])

    search_payload = (
        payloads.get("trading_search")
        or payloads.get("general_search")
        or payloads.get("debt_search")
    )
    if search_payload and not (broad_news_query and recent_news):
        story_items = search_payload.get("categories", {}).get("stories", {}).get("items", [])[:4]
        if story_items:
            lines = [
                f"- {item.get('title')} — {item.get('source_name')}"
                for item in story_items
            ]
            if _append_block(blocks, "Relevant news", lines, max_chars):
                for item in story_items:
                    if item.get("source_name"):
                        sources.append(item["source_name"])

        social_items = search_payload.get("categories", {}).get("social_signals", {}).get("items", [])[:2]
        if social_items:
            lines = [
                f"- {item.get('text', '')[:120]} — @{item.get('author_username')}"
                for item in social_items
            ]
            if _append_block(blocks, "Social signals", lines, max_chars):
                for item in social_items:
                    username = item.get("author_username")
                    if username:
                        sources.append(f"@{username}")

    recent_news = payloads.get("recent_news")
    if recent_news and not search_payload and not broad_news_query:
        items = recent_news[:4] if isinstance(recent_news, list) else []
        lines = [
            f"- {item.get('canonical_headline')} — {item.get('source_name')}"
            for item in items
        ]
        if lines and _append_block(blocks, "Relevant news", lines, max_chars):
            for item in items:
                if item.get("source_name"):
                    sources.append(item["source_name"])

    embedding_hits = payloads.get("embedding_search", {}).get("results", [])
    if embedding_hits:
        lines = [
            f"- {item.get('title')} — {item.get('source_name')}"
            for item in embedding_hits[:3]
        ]
        if _append_block(blocks, "Semantic market hits", lines, max_chars):
            for item in embedding_hits[:3]:
                if item.get("source_name"):
                    sources.append(item["source_name"])

    if not blocks and errors:
        lines = [
            "NepalOSINT को live context अहिले ल्याउन सकिएन।",
            "हालको सार्वजनिक तथ्य चाहिएको हो भने उत्तरमा यो सीमाबारे स्पष्ट भन्नुहोस्।",
        ]
        return "\n".join(lines)

    if not blocks:
        return None

    deduped_sources: list[str] = []
    for source in sources:
        if source and source not in deduped_sources:
            deduped_sources.append(source)

    source_line = ", ".join(deduped_sources[:4]) if deduped_sources else "NepalOSINT live context"
    footer = [
        "उत्तर दिने नियम:",
        "- माथिको सन्दर्भ प्रयोग भएको छ भने नेपालीमै छोटो र स्पष्ट उत्तर दिनुहोस्।",
        "- उत्तरको अन्त्यमा अनिवार्य रूपमा `स्रोत:` शीर्षक राखेर २ देखि ४ स्रोत उल्लेख गर्नुहोस्।",
        f"- प्राथमिक स्रोतहरू: {source_line}",
    ]
    if plan and getattr(plan, "wants_history", False):
        footer.append("- इतिहाससम्बन्धी प्रश्नमा समयसीमा स्पष्ट गरेर उत्तर दिनुहोस्।")

    combined = "\n\n".join(blocks + ["\n".join(footer)])
    return combined[:max_chars].rstrip()
