# Required: pip install discord.py gspread google-auth python-dotenv aiohttp

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import discord
import gspread
from discord.ext import tasks
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials


ENV_PATH = Path(".env")
STATE_PATH = Path("introsync_state.json")
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
MAX_CHUNK_CHARS = 5000
MAX_MESSAGES_PER_CHUNK = 8

REQUIRED_ENV_VARS = [
    "DISCORD_TOKEN",
    "SARVAM_API_KEY",
    "SARVAM_ROUTER_MODEL",
    "INTRO_CHANNEL_ID",
    "GUILD_ID",
    "GOOGLE_SHEET_ID",
    "GOOGLE_SERVICE_ACCOUNT_EMAIL",
    "GOOGLE_PROJECT_ID",
    "GOOGLE_CLIENT_ID",
    "GOOGLE_PRIVATE_KEY",
]

SHEET_HEADERS = [
    "Username",
    "Real Name",
    "Background",
    "Experience",
    "Skills",
    "Location",
    "Goals",
    "Summary",
    "Date Posted",
    "Message URL",
]

COLUMN_WIDTHS = [20, 20, 25, 20, 30, 15, 30, 40, 22, 40]
COL_USERNAME = 0
COL_REAL_NAME = 1
COL_BACKGROUND = 2
COL_EXPERIENCE = 3
COL_SKILLS = 4
COL_LOCATION = 5
COL_GOALS = 6
COL_SUMMARY = 7
COL_DATE_POSTED = 8
COL_MESSAGE_URL = 9

SYSTEM_PROMPT = (
    "You are a strict JSON-only classifier. You analyze Discord messages and "
    "identify ONLY messages where a real person is genuinely introducing "
    "THEMSELVES - sharing their own name, background, skills, or experience.\n\n"
    "You must respond with ONLY a raw JSON array. No thinking, no explanation, "
    "no markdown, no code fences. Start with [ and end with ]."
)

USER_PROMPT_TEMPLATE = (
    "Analyze each Discord message below and classify it strictly.\n\n"
    "A message IS an introduction ONLY if:\n"
    "  - The author is talking about THEMSELVES\n"
    "  - They share personal details like their name, profession, skills, \n"
    "    background, education, experience, or what they are working on\n"
    "  - Examples: someone saying who they are, what they do, where they are from,\n"
    "    what technologies they use, what they are studying or building\n\n"
    "A message is NOT an introduction if it is any of these - skip it:\n"
    "  - A welcome message inviting others to introduce themselves\n"
    "  - A greeting or reaction to someone else's introduction (e.g. 'welcome!', 'great!')\n"
    "  - A question, meme, joke, or random chat\n"
    "  - A bot message or server announcement\n"
    "  - Someone asking for help or posting a link\n"
    "  - Short filler messages like 'hello', 'hi', 'gm', 'aaaaa', 'yes boss'\n"
    "  - Someone welcoming a new member\n"
    "  - Any message that is NOT the author describing themselves\n\n"
    "For each message respond with one object in a JSON array:\n\n"
    "If NOT a real self-introduction:\n"
    '{ "id": "MSG_001", "is_introduction": false }\n\n'
    "If IS a genuine self-introduction:\n"
    "{\n"
    '  "id": "MSG_001",\n'
    '  "is_introduction": true,\n'
    '  "real_name": "...",\n'
    '  "experience": "...",\n'
    '  "skills": "...",\n'
    '  "background": "...",\n'
    '  "location": "...",\n'
    '  "goals": "...",\n'
    '  "summary": "..."\n'
    "}\n\n"
    "Use null for any field not mentioned. Summary must be 2 sentences max.\n"
    "Return exactly one object per message. Array length must match message count.\n"
    "IMPORTANT: Start your response with [ and end with ]. Nothing else.\n\n"
    "Messages:\n"
    "__CHUNK_CONTENT__"
)

SUMMARY_MERGE_SYSTEM_PROMPT = (
    "You are a JSON-only assistant. Respond with only a raw JSON object. "
    "No markdown, no explanation, no code fences."
)


@dataclass
class IntroSyncConfig:
    discord_token: str
    sarvam_api_key: str
    sarvam_model: str
    intro_channel_id: int
    guild_id: int
    google_sheet_id: str


class SarvamChunkError(Exception):
    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_str()}] {message}", flush=True)


def load_config() -> IntroSyncConfig:
    load_dotenv(str(ENV_PATH))

    missing = []
    for name in REQUIRED_ENV_VARS:
        value = os.getenv(name)
        if value is None or not value.strip():
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Missing or empty required environment variables in .env: "
            + ", ".join(missing)
        )

    try:
        intro_channel_id = int(os.getenv("INTRO_CHANNEL_ID", ""))
        guild_id = int(os.getenv("GUILD_ID", ""))
    except ValueError as exc:
        raise RuntimeError(
            "INTRO_CHANNEL_ID and GUILD_ID must be valid Discord ID integers."
        ) from exc

    return IntroSyncConfig(
        discord_token=os.getenv("DISCORD_TOKEN", ""),
        sarvam_api_key=os.getenv("SARVAM_API_KEY", ""),
        sarvam_model=os.getenv("SARVAM_ROUTER_MODEL", ""),
        intro_channel_id=intro_channel_id,
        guild_id=guild_id,
        google_sheet_id=os.getenv("GOOGLE_SHEET_ID", ""),
    )


def read_sync_state(state_path: Path = STATE_PATH) -> dict[str, str] | None:
    if not state_path.exists():
        return None

    try:
        raw = state_path.read_text(encoding="utf-8").strip()
        if not raw:
            log("State file is empty; treating as first run.")
            return None

        payload: dict[str, Any] = json.loads(raw)
        last_message_id = str(payload["last_message_id"])
        last_message_timestamp = str(payload["last_message_timestamp"])
        int(last_message_id)
        return {
            "last_message_id": last_message_id,
            "last_message_timestamp": last_message_timestamp,
        }
    except Exception:
        log("Malformed introsync_state.json; treating as first run.")
        return None


def write_sync_state(
    last_message_id: str | int | None,
    last_message_timestamp: str | None,
    state_path: Path = STATE_PATH,
) -> None:
    payload = {
        "last_message_id": str(last_message_id) if last_message_id is not None else "",
        "last_message_timestamp": last_message_timestamp or "",
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_sheet(config: IntroSyncConfig) -> gspread.Worksheet:
    credentials_dict = {
        "type": "service_account",
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY", "").replace("\\n", "\n"),
        "client_email": os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    credentials = Credentials.from_service_account_info(
        credentials_dict,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(config.google_sheet_id)
    return spreadsheet.sheet1


def apply_sheet_formatting(worksheet: gspread.Worksheet) -> None:
    requests = [
        {
            "repeatCell": {
                "range": {
                    "sheetId": worksheet.id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": len(SHEET_HEADERS),
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {
                            "red": 0.129,
                            "green": 0.588,
                            "blue": 0.953,
                        },
                        "textFormat": {
                            "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                            "bold": True,
                        },
                    }
                },
                "fields": (
                    "userEnteredFormat.backgroundColor,"
                    "userEnteredFormat.textFormat.foregroundColor,"
                    "userEnteredFormat.textFormat.bold"
                ),
            }
        },
        {
            "updateSheetProperties": {
                "properties": {"sheetId": worksheet.id, "gridProperties": {"frozenRowCount": 1}},
                "fields": "gridProperties.frozenRowCount",
            }
        },
    ]

    for col_index, width in enumerate(COLUMN_WIDTHS):
        requests.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet.id,
                        "dimension": "COLUMNS",
                        "startIndex": col_index,
                        "endIndex": col_index + 1,
                    },
                    "properties": {"pixelSize": width},
                    "fields": "pixelSize",
                }
            }
        )

    worksheet.spreadsheet.batch_update({"requests": requests})


def reset_sheet_with_headers(worksheet: gspread.Worksheet) -> None:
    worksheet.clear()
    worksheet.update(range_name="A1:J1", values=[SHEET_HEADERS])
    apply_sheet_formatting(worksheet)


def append_intro_rows(worksheet: gspread.Worksheet, rows: list[list[str]]) -> None:
    if rows:
        worksheet.append_rows(rows, value_input_option="USER_ENTERED")


def parse_date(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def is_present(value: str) -> bool:
    value_clean = (value or "").strip()
    return bool(value_clean and value_clean.lower() != "null")


def first_non_null(values: list[str]) -> str:
    for value in values:
        if is_present(value):
            return value.strip()
    return ""


def split_unique_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        if not is_present(value):
            continue
        parts = re.split(r",|\n|;", value)
        for part in parts:
            token = part.strip()
            if not token:
                continue
            token_key = token.casefold()
            if token_key in seen:
                continue
            seen.add(token_key)
            merged.append(token)
    return merged


def extract_model_text_from_response(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message_obj = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    if not isinstance(message_obj, dict):
        return ""
    content = message_obj.get("content")
    if content is None or (isinstance(content, str) and not content.strip()):
        content = message_obj.get("reasoning_content", "")
    if content is None or (isinstance(content, str) and not content.strip()):
        content = message_obj.get("thinking", "")
    return str(content or "")


def extract_merged_summary_from_text(content: str) -> str | None:
    cleaned = content.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Direct parse first.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and is_present(coerce_text(parsed.get("merged_summary"))):
            return coerce_text(parsed.get("merged_summary")).strip()
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and is_present(coerce_text(item.get("merged_summary"))):
                    return coerce_text(item.get("merged_summary")).strip()
    except Exception:
        pass

    # Parse any valid JSON objects embedded in surrounding text.
    for obj in extract_json_objects_from_text(cleaned):
        if is_present(coerce_text(obj.get("merged_summary"))):
            return coerce_text(obj.get("merged_summary")).strip()

    # Last-resort tolerant extraction for malformed key quoting.
    match = re.search(r"merged_summary\s*[:=]\s*\"([^\"]+)\"", cleaned)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    return None


async def request_merged_summary(
    session: aiohttp.ClientSession,
    config: IntroSyncConfig,
    username: str,
    summaries: list[str],
) -> str:
    valid_summaries = [item.strip() for item in summaries if is_present(item)]
    if len(valid_summaries) <= 1:
        return valid_summaries[0] if valid_summaries else ""

    summary_lines = "\n".join(f"- {item}" for item in valid_summaries)
    user_prompt = (
        f"The following are multiple summary snippets about the same person named {username}. "
        "Merge them into a single cohesive summary of 2 sentences max that captures everything "
        "important about them.\n\n"
        f"Summaries:\n{summary_lines}\n\n"
        "Respond with only:\n"
        '{ "merged_summary": "..." }'
    )

    payload = {
        "model": config.sarvam_model,
        "messages": [
            {"role": "system", "content": SUMMARY_MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 512,
        "budget_tokens": 256,
    }
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": config.sarvam_api_key,
    }

    async def send_summary_merge_request(request_payload: dict[str, Any]) -> str:
        timeout = aiohttp.ClientTimeout(total=90)
        async with session.post(
            SARVAM_URL,
            json=request_payload,
            headers=headers,
            timeout=timeout,
        ) as response:
            raw_text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {raw_text[:300]}")
            response_json = json.loads(raw_text)
            return extract_model_text_from_response(response_json).strip()

    try:
        content = await send_summary_merge_request(payload)

        merged_summary = extract_merged_summary_from_text(content)
        if is_present(merged_summary or ""):
            return (merged_summary or "").strip()

        # Retry once with a tighter response contract.
        retry_payload = {
            "model": config.sarvam_model,
            "messages": [
                {"role": "system", "content": SUMMARY_MERGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Merge these summaries about {username} into one 2-sentence summary.\n\n"
                        f"Summaries:\n{summary_lines}\n\n"
                        'Return EXACTLY one JSON object and nothing else:\n{"merged_summary":"..."}'
                    ),
                },
            ],
            "max_tokens": 300,
            "budget_tokens": 128,
        }
        retry_content = await send_summary_merge_request(retry_payload)
        retry_merged = extract_merged_summary_from_text(retry_content)
        if is_present(retry_merged or ""):
            return (retry_merged or "").strip()
        raise RuntimeError("merged_summary missing in response")
    except Exception as exc:
        log(f"[WARN] Summary merge failed for {username}; using first summary ({exc})")
        return valid_summaries[0]


async def merge_user_rows(
    config: IntroSyncConfig,
    username: str,
    rows: list[list[str]],
) -> list[str]:
    real_name = first_non_null([row[COL_REAL_NAME] for row in rows])
    background = first_non_null([row[COL_BACKGROUND] for row in rows])
    experience = first_non_null([row[COL_EXPERIENCE] for row in rows])
    location = first_non_null([row[COL_LOCATION] for row in rows])
    skills = ", ".join(split_unique_terms([row[COL_SKILLS] for row in rows]))
    goals = ", ".join(split_unique_terms([row[COL_GOALS] for row in rows]))

    summaries = [row[COL_SUMMARY] for row in rows if is_present(row[COL_SUMMARY])]
    if len(summaries) > 1:
        async with aiohttp.ClientSession() as session:
            summary = await request_merged_summary(session, config, username, summaries)
    else:
        summary = summaries[0] if summaries else ""

    earliest_row = rows[0]
    earliest_dt = parse_date(earliest_row[COL_DATE_POSTED])
    for row in rows[1:]:
        candidate_dt = parse_date(row[COL_DATE_POSTED])
        if earliest_dt is None:
            earliest_dt = candidate_dt
            earliest_row = row
            continue
        if candidate_dt is not None and candidate_dt < earliest_dt:
            earliest_dt = candidate_dt
            earliest_row = row

    return [
        username,
        real_name,
        background,
        experience,
        skills,
        location,
        goals,
        summary,
        earliest_row[COL_DATE_POSTED],
        earliest_row[COL_MESSAGE_URL],
    ]


async def deduplicate_intro_rows(
    config: IntroSyncConfig,
    intro_rows: list[list[str]],
) -> list[list[str]]:
    groups: dict[str, list[list[str]]] = {}
    for row in intro_rows:
        username = row[COL_USERNAME].strip()
        if not username:
            continue
        groups.setdefault(username, []).append(row)

    merged_rows: list[list[str]] = []
    for username, rows in groups.items():
        if len(rows) > 1:
            log(f"[MERGED] {username} — combined {len(rows)} messages into 1 row")
        merged_rows.append(await merge_user_rows(config=config, username=username, rows=rows))

    merged_rows.sort(key=lambda item: parse_date(item[COL_DATE_POSTED]) or datetime.max)
    log(
        "Deduplication complete. "
        f"{len(merged_rows)} unique members from {len(intro_rows)} total introduction messages."
    )
    return merged_rows


def read_existing_sheet_rows(worksheet: gspread.Worksheet) -> tuple[dict[str, int], dict[str, list[str]]]:
    all_values = worksheet.get_all_values()
    username_to_row_number: dict[str, int] = {}
    username_to_row_data: dict[str, list[str]] = {}
    if len(all_values) <= 1:
        return username_to_row_number, username_to_row_data

    for row_number, row in enumerate(all_values[1:], start=2):
        padded = (row + [""] * len(SHEET_HEADERS))[: len(SHEET_HEADERS)]
        username = padded[COL_USERNAME].strip()
        if not username:
            continue
        username_to_row_number[username] = row_number
        username_to_row_data[username] = padded
    return username_to_row_number, username_to_row_data


async def upsert_rows_into_sheet(
    config: IntroSyncConfig,
    worksheet: gspread.Worksheet,
    new_rows: list[list[str]],
) -> None:
    if not new_rows:
        return

    username_to_row_number, username_to_row_data = read_existing_sheet_rows(worksheet)
    rows_to_append: list[list[str]] = []

    for new_row in new_rows:
        username = new_row[COL_USERNAME].strip()
        if not username:
            continue

        if username in username_to_row_number and username in username_to_row_data:
            existing_row = username_to_row_data[username]
            merged = await merge_user_rows(config=config, username=username, rows=[existing_row, new_row])
            target_row = username_to_row_number[username]
            worksheet.update(
                range_name=f"A{target_row}:J{target_row}",
                values=[merged],
            )
            username_to_row_data[username] = merged
            log(f"[UPDATED ROW] {username} — new info merged into existing row")
        else:
            rows_to_append.append(new_row)
            username_to_row_data[username] = new_row
            log(f"[NEW ROW] {username} — added as new entry")

    append_intro_rows(worksheet, rows_to_append)


def build_chunks(
    messages: list[discord.Message],
) -> list[tuple[str, dict[str, discord.Message]]]:
    chunks: list[tuple[str, dict[str, discord.Message]]] = []
    current_parts: list[str] = []
    current_map: dict[str, discord.Message] = {}
    current_len = 0

    chunk_message_count = 0
    for index, message in enumerate(messages, start=1):
        msg_id = f"MSG_{index:03d}"
        content = message.content.strip() if message.content else ""
        if not content:
            content = "[no text content]"

        block = (
            f"[{msg_id}]\n"
            f"Author: {message.author.name}\n"
            f"Date: {message.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Content: {content}\n"
        )

        block_len = len(block) + 1
        if current_parts and (
            current_len + block_len > MAX_CHUNK_CHARS
            or chunk_message_count >= MAX_MESSAGES_PER_CHUNK
        ):
            chunks.append(("\n".join(current_parts), current_map))
            current_parts = []
            current_map = {}
            current_len = 0
            chunk_message_count = 0

        current_parts.append(block)
        current_map[msg_id] = message
        current_len += block_len
        chunk_message_count += 1

    if current_parts:
        chunks.append(("\n".join(current_parts), current_map))

    return chunks


def sanitize_model_output(content: str) -> str:
    return (
        content.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


async def call_sarvam_for_chunk(
    session: aiohttp.ClientSession,
    config: IntroSyncConfig,
    chunk_content: str,
) -> tuple[str, str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.replace("__CHUNK_CONTENT__", chunk_content)

    payload = {
        "model": config.sarvam_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "max_tokens": 4096,
        "budget_tokens": 2048,
    }
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": config.sarvam_api_key,
    }

    timeout = aiohttp.ClientTimeout(total=120)
    async with session.post(
        SARVAM_URL,
        json=payload,
        headers=headers,
        timeout=timeout,
    ) as response:
        raw_text = await response.text()
        if response.status >= 400:
            raise SarvamChunkError(f"HTTP {response.status}", raw_response=raw_text)
        try:
            response_json = json.loads(raw_text)
        except Exception as exc:
            raise SarvamChunkError(
                f"Invalid API JSON: {exc}",
                raw_response=raw_text,
            ) from exc
        choices = response_json.get("choices") if isinstance(response_json, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        message_obj = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        finish_reason = (
            first_choice.get("finish_reason", "")
            if isinstance(first_choice, dict)
            else ""
        )

        if not isinstance(message_obj, dict):
            return "", raw_text, str(finish_reason)

        content = message_obj.get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            content = message_obj.get("reasoning_content", "")
        if content is None or (isinstance(content, str) and not content.strip()):
            content = message_obj.get("thinking", "")

        return str(content or ""), raw_text, str(finish_reason)


def extract_json_objects_from_text(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    snippet = text[start : i + 1]
                    try:
                        parsed = json.loads(snippet)
                    except Exception:
                        continue
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    start = -1

    return objects


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def intro_row_from_result(
    result: dict[str, Any],
    message: discord.Message,
    config: IntroSyncConfig,
) -> list[str]:
    message_url = (
        f"https://discord.com/channels/{config.guild_id}/"
        f"{config.intro_channel_id}/{message.id}"
    )
    return [
        message.author.name,
        coerce_text(result.get("real_name")),
        coerce_text(result.get("background")),
        coerce_text(result.get("experience")),
        coerce_text(result.get("skills")),
        coerce_text(result.get("location")),
        coerce_text(result.get("goals")),
        coerce_text(result.get("summary")),
        message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        message_url,
    ]


async def classify_messages(
    config: IntroSyncConfig, messages: list[discord.Message]
) -> tuple[list[list[str]], int, discord.Message | None]:
    chunks = build_chunks(messages)
    log(f"Built {len(chunks)} chunks. Sending to Sarvam...")

    intro_rows: list[list[str]] = []
    analyzed_count = 0
    successful_chunks = 0
    checkpoint_message: discord.Message | None = None
    can_advance_checkpoint = True

    async with aiohttp.ClientSession() as session:
        for chunk_index, (chunk_content, id_map) in enumerate(chunks, start=1):
            chunk_messages = list(id_map.values())
            chunk_last_message = chunk_messages[-1] if chunk_messages else None
            try:
                model_content, raw_api_response, finish_reason = await call_sarvam_for_chunk(
                    session=session,
                    config=config,
                    chunk_content=chunk_content,
                )
            except Exception as exc:
                raw_hint = ""
                if isinstance(exc, SarvamChunkError) and exc.raw_response:
                    raw_hint = f" | Raw: {exc.raw_response[:300]}"
                log(
                    f"[API ERROR] Chunk {chunk_index}/{len(chunks)} failed — "
                    f"{exc.__class__.__name__}: {exc}{raw_hint}"
                )
                can_advance_checkpoint = False
                if chunk_index < len(chunks):
                    await asyncio.sleep(1)
                continue

            content = model_content.strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            content = content.strip()
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                content = content[start : end + 1]

            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    result_array = [parsed]
                else:
                    result_array = parsed
            except Exception:
                fallback_objects = extract_json_objects_from_text(content)
                if fallback_objects:
                    result_array = fallback_objects
                    log(
                        f"[WARN] Chunk {chunk_index}/{len(chunks)} used fallback JSON "
                        f"object extraction ({len(result_array)} objects)."
                    )
                else:
                    log(
                        f"[API ERROR] Chunk {chunk_index}/{len(chunks)} failed — "
                        f"JSONDecodeError: invalid JSON after cleaning | "
                        f"Raw: {(raw_api_response or '')[:300]}"
                    )
                    can_advance_checkpoint = False
                    if chunk_index < len(chunks):
                        await asyncio.sleep(1)
                    continue

            if not isinstance(result_array, list):
                log(
                    f"[API ERROR] Chunk {chunk_index}/{len(chunks)} — skipped "
                    "(response is not a JSON array)."
                )
                can_advance_checkpoint = False
                if chunk_index < len(chunks):
                    await asyncio.sleep(1)
                continue
            successful_chunks += 1
            if can_advance_checkpoint and chunk_last_message is not None:
                checkpoint_message = chunk_last_message

            if finish_reason == "length":
                log(
                    f"[WARN] Chunk {chunk_index}/{len(chunks)} hit model length limit; "
                    "partial extraction may apply."
                )

            for result in result_array:
                if not isinstance(result, dict):
                    continue

                result_id = result.get("id")
                is_intro = result.get("is_introduction")
                if not isinstance(result_id, str) or not isinstance(is_intro, bool):
                    continue

                message = id_map.get(result_id)
                if message is None:
                    continue

                analyzed_count += 1
                if is_intro:
                    row = intro_row_from_result(result=result, message=message, config=config)
                    intro_rows.append(row)
                    log(
                        f"[INTRO FOUND] {message.author.name} — "
                        f"{coerce_text(result.get('real_name')) or 'Unknown'}"
                    )
                else:
                    log(f"[NOT INTRO] {message.author.name} — skipped")

            log(
                f"Chunk {chunk_index}/{len(chunks)} processed — {analyzed_count} "
                f"messages analyzed, {len(intro_rows)} intros found so far..."
            )

            if chunk_index < len(chunks):
                await asyncio.sleep(1)

    return intro_rows, successful_chunks, checkpoint_message


async def fetch_non_bot_messages(
    channel: discord.TextChannel, after_obj: discord.Object | None = None
) -> list[discord.Message]:
    if after_obj is None:
        messages = [msg async for msg in channel.history(limit=None)]
    else:
        messages = [msg async for msg in channel.history(after=after_obj, limit=None)]
    filtered = [msg for msg in messages if not msg.author.bot]
    filtered.sort(key=lambda msg: msg.created_at)
    return filtered


class IntroSyncClient(discord.Client):
    def __init__(self, config: IntroSyncConfig) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(intents=intents)
        self.config = config

    async def on_ready(self) -> None:
        log("introsync ready. Starting first sync...")
        if not self.sync_loop.is_running():
            self.sync_loop.start()

    @tasks.loop(hours=6)
    async def sync_loop(self) -> None:
        try:
            await self.run_sync_cycle()
        except Exception as exc:
            log(f"[SYNC ERROR] Unexpected cycle failure: {exc}")
        finally:
            log("Next sync in 6 hours.")

    async def run_sync_cycle(self) -> None:
        channel_obj = self.get_channel(self.config.intro_channel_id)
        if channel_obj is None:
            try:
                channel_obj = await self.fetch_channel(self.config.intro_channel_id)
            except Exception as exc:
                log(f"Channel not found or inaccessible; cycle aborted ({exc})")
                return

        if not isinstance(channel_obj, discord.TextChannel):
            log("Configured INTRO_CHANNEL_ID is not a text channel; cycle aborted.")
            return

        state = read_sync_state()
        first_run = not state or not state.get("last_message_id")

        if first_run:
            messages = await fetch_non_bot_messages(channel_obj)
        else:
            try:
                anchor = discord.Object(id=int(state["last_message_id"]))
            except Exception:
                log("Invalid last_message_id in state; falling back to first run behavior.")
                messages = await fetch_non_bot_messages(channel_obj)
                first_run = True
            else:
                messages = await fetch_non_bot_messages(channel_obj, after_obj=anchor)

        log(f"Fetched {len(messages)} messages. Building chunks...")

        try:
            worksheet = get_sheet(self.config)
        except Exception as exc:
            log(f"Google API failure; cycle aborted ({exc})")
            return

        if first_run:
            if messages:
                intro_rows, successful_chunks, checkpoint_message = await classify_messages(
                    config=self.config,
                    messages=messages,
                )
            else:
                intro_rows = []
                successful_chunks = 0
                checkpoint_message = None

            if messages and successful_chunks == 0:
                log("[WARN] All chunks failed. State not saved. Will retry full scan next cycle.")
                return

            deduped_rows = await deduplicate_intro_rows(self.config, intro_rows)
            reset_sheet_with_headers(worksheet)
            append_intro_rows(worksheet, deduped_rows)

            if checkpoint_message is not None:
                write_sync_state(checkpoint_message.id, checkpoint_message.created_at.isoformat())
                if checkpoint_message.id != messages[-1].id:
                    log(
                        "[WARN] Partial chunk failure detected. "
                        "State saved to last contiguous successful chunk only."
                    )
            else:
                write_sync_state(None, None)

            log(
                f"Sync complete. {len(deduped_rows)} introductions out of "
                f"{len(messages)} messages."
            )
            return

        if not messages:
            log("No new messages. Waiting for next cycle.")
            return

        intro_rows, successful_chunks, checkpoint_message = await classify_messages(
            config=self.config,
            messages=messages,
        )
        if successful_chunks == 0:
            log("[WARN] All chunks failed. State not saved. Will retry full scan next cycle.")
            return

        deduped_rows = await deduplicate_intro_rows(self.config, intro_rows)
        if deduped_rows:
            await upsert_rows_into_sheet(self.config, worksheet, deduped_rows)

        if checkpoint_message is not None:
            write_sync_state(checkpoint_message.id, checkpoint_message.created_at.isoformat())
            if checkpoint_message.id != messages[-1].id:
                log(
                    "[WARN] Partial chunk failure detected. "
                    "State saved to last contiguous successful chunk only."
                )
        else:
            log("[WARN] First chunk failed. State not advanced; pending messages will be retried.")
            return

        log(
            f"Sync complete. {len(deduped_rows)} introductions out of "
            f"{len(messages)} messages."
        )


def main() -> None:
    try:
        config = load_config()
    except Exception as exc:
        log(f"Startup failed: {exc}")
        raise SystemExit(1) from exc

    client = IntroSyncClient(config=config)
    client.run(config.discord_token)


if __name__ == "__main__":
    main()
