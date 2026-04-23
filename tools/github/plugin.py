"""
tools/github/plugin.py — public GitHub repo analyzer.

Uses GitHub's REST API (https://api.github.com) with no auth by default.
Set `GITHUB_TOKEN` in the env to bump the rate limit from 60 → 5000 req/h.

Why not `git clone`?
  * Startup cost + disk I/O for a chat turn is too high.
  * Most questions are answered by metadata + README + tree, all of which
    the REST API exposes cheaply.
  * Skip lfs, submodules, git history — the LLM rarely asks about those.

What we surface:
  1. Repo metadata: full name, description, primary language, stars,
     forks, open issues, license, default branch, last push timestamp.
  2. Top-level tree (truncated) — gives the LLM a sense of structure.
  3. README (first ~3 KB) — highest-signal file for "what is this".
  4. Optional `file_path` argument → fetch that specific file directly.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import urllib.parse
from typing import Any

import httpx

from core.tool_contracts import (
    ToolCategory,
    ToolContext,
    ToolParam,
    ToolResult,
    ToolSpec,
)
from core.tool_registry import get_registry

logger = logging.getLogger("yetidai.github")


API_BASE = "https://api.github.com"
REQUEST_TIMEOUT = 10.0
MAX_README_CHARS = 3200
MAX_TREE_ENTRIES = 60
MAX_FILE_CHARS = 4000


GITHUB_SPEC = ToolSpec(
    tool_id="github.analyze_repo",
    name="analyze_github_repo",
    description=(
        "Inspect a public GitHub repository via the REST API. Returns "
        "metadata + a truncated file tree + the README (or a specific "
        "file if `file_path` is given).\n\n"
        "USE FOR:\n"
        "  • 'What is this repo about?' / 'summarize this GitHub project'.\n"
        "  • Comparing two projects (call twice in parallel).\n"
        "  • Reading a specific file without cloning: pass `file_path`.\n\n"
        "ACCEPTED INPUTS FOR `repo`:\n"
        "  • `owner/name` — e.g. `anthropics/claude-code`.\n"
        "  • Full GitHub URL — e.g. `https://github.com/anthropics/claude-code`.\n"
        "  • Branch URL — `https://github.com/owner/name/tree/<branch>`.\n\n"
        "LIMITS:\n"
        "  • README capped at ~3 KB, tree at 60 entries, file at 4 KB.\n"
        "  • Rate-limited to 60 req/h without GITHUB_TOKEN; 5000/h with one.\n"
        "  • Private repos are not supported (would need a user token)."
    ),
    category=ToolCategory.UTILITY,
    parameters=[
        ToolParam(
            name="repo",
            type="string",
            description="Repo as `owner/name` or a full github.com URL.",
            required=True,
            examples=[
                "anthropics/claude-code",
                "https://github.com/HimalayaAI/yetidai",
                "https://github.com/HimalayaAI/yetidai/tree/feat/yeti-robustness-v2",
            ],
        ),
        ToolParam(
            name="file_path",
            type="string",
            description=(
                "Optional path inside the repo to read directly, "
                "e.g. `bot.py` or `core/tool_registry.py`. When set, the "
                "tool returns the file contents instead of the README."
            ),
            required=False,
            examples=["README.md", "package.json", "src/index.ts"],
        ),
        ToolParam(
            name="branch",
            type="string",
            description=(
                "Optional branch or tag name. Defaults to the repo's "
                "default branch."
            ),
            required=False,
        ),
    ],
    timeout_seconds=20.0,
)


# ── URL / identifier parsing ──────────────────────────────────────

_GITHUB_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)"
    r"(?:\.git)?(?:/tree/([^/?#]+))?/?(?:[?#].*)?$",
    re.IGNORECASE,
)
_OWNER_REPO_RE = re.compile(r"^([\w.-]+)/([\w.-]+)$")


def _parse_repo(raw: str) -> tuple[str, str, str | None]:
    """Return (owner, repo, branch) from a URL or owner/name string."""
    raw = (raw or "").strip().rstrip("/")
    m = _GITHUB_URL_RE.match(raw)
    if m:
        owner, name, branch = m.group(1), m.group(2), m.group(3)
        return owner, name, branch
    m = _OWNER_REPO_RE.match(raw)
    if m:
        return m.group(1), m.group(2), None
    raise ValueError(f"unrecognised repo identifier: {raw!r}")


# ── API helpers ───────────────────────────────────────────────────

def _auth_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "yetidai/1.0",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _api_get(
    client: httpx.AsyncClient, path: str, *, params: dict[str, Any] | None = None,
) -> httpx.Response:
    url = f"{API_BASE}{path}"
    return await client.get(
        url,
        headers=_auth_headers(),
        params=params,
        timeout=REQUEST_TIMEOUT,
    )


def _format_metadata(meta: dict[str, Any]) -> str:
    license_name = "—"
    if isinstance(meta.get("license"), dict):
        license_name = meta["license"].get("spdx_id") or meta["license"].get("name") or "—"
    return (
        f"Repository: {meta.get('full_name')}\n"
        f"Description: {meta.get('description') or '—'}\n"
        f"Language: {meta.get('language') or '—'} | "
        f"License: {license_name} | "
        f"Default branch: {meta.get('default_branch')}\n"
        f"Stars: {meta.get('stargazers_count'):,} | "
        f"Forks: {meta.get('forks_count'):,} | "
        f"Open issues: {meta.get('open_issues_count'):,} | "
        f"Last push: {meta.get('pushed_at')}\n"
        f"URL: {meta.get('html_url')}"
    )


def _format_tree(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "(tree unavailable)"
    lines: list[str] = []
    for entry in entries[:MAX_TREE_ENTRIES]:
        path = entry.get("path", "")
        kind = "📁" if entry.get("type") == "tree" else "📄"
        size = entry.get("size")
        if isinstance(size, int) and size > 0 and entry.get("type") != "tree":
            lines.append(f"  {kind} {path}  ({size} B)")
        else:
            lines.append(f"  {kind} {path}")
    more = len(entries) - MAX_TREE_ENTRIES
    if more > 0:
        lines.append(f"  … ({more} more entries)")
    return "\n".join(lines)


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + "\n\n… (truncated)"


def _decode_content(payload: dict[str, Any]) -> str:
    raw = payload.get("content") or ""
    encoding = payload.get("encoding")
    if encoding == "base64":
        try:
            return base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return raw


# ── Handler ───────────────────────────────────────────────────────

async def handle_github(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    repo_raw = arguments.get("repo")
    if not repo_raw or not isinstance(repo_raw, str):
        return ToolResult(tool_id=GITHUB_SPEC.tool_id, success=False, error="Missing repo")

    try:
        owner, name, branch_from_url = _parse_repo(repo_raw)
    except ValueError as exc:
        return ToolResult(tool_id=GITHUB_SPEC.tool_id, success=False, error=str(exc))

    branch = (arguments.get("branch") or branch_from_url or "").strip() or None
    file_path = (arguments.get("file_path") or "").strip() or None

    try:
        async with httpx.AsyncClient() as client:
            meta_resp = await _api_get(client, f"/repos/{owner}/{name}")
            if meta_resp.status_code == 404:
                return ToolResult(
                    tool_id=GITHUB_SPEC.tool_id,
                    success=False,
                    error=f"repo not found or private: {owner}/{name}",
                )
            if meta_resp.status_code == 403:
                rl = meta_resp.headers.get("x-ratelimit-remaining")
                return ToolResult(
                    tool_id=GITHUB_SPEC.tool_id,
                    success=False,
                    error=(
                        f"GitHub rate limit hit (remaining={rl}). "
                        "Set GITHUB_TOKEN to raise the limit."
                    ),
                )
            meta_resp.raise_for_status()
            meta = meta_resp.json()
            branch = branch or meta.get("default_branch") or "main"

            # Specific file mode — skip tree + README to save rate-limit tokens.
            if file_path:
                file_resp = await _api_get(
                    client,
                    f"/repos/{owner}/{name}/contents/{urllib.parse.quote(file_path)}",
                    params={"ref": branch},
                )
                if file_resp.status_code == 404:
                    return ToolResult(
                        tool_id=GITHUB_SPEC.tool_id,
                        success=False,
                        error=f"file not found: {file_path} on {branch}",
                    )
                file_resp.raise_for_status()
                decoded = _decode_content(file_resp.json())
                block = (
                    f"{_format_metadata(meta)}\n\n"
                    f"──── {file_path} @ {branch} ────\n"
                    f"{_truncate(decoded, MAX_FILE_CHARS)}"
                )
                return ToolResult(
                    tool_id=GITHUB_SPEC.tool_id,
                    success=True,
                    content=block,
                    meta={"owner": owner, "repo": name, "branch": branch, "file": file_path},
                )

            # Default mode: metadata + tree + README, all in parallel.
            tree_url = f"/repos/{owner}/{name}/git/trees/{urllib.parse.quote(branch)}"
            readme_url = f"/repos/{owner}/{name}/readme"

            import asyncio as _asyncio
            tree_resp, readme_resp = await _asyncio.gather(
                _api_get(client, tree_url, params={"recursive": "0"}),
                _api_get(client, readme_url, params={"ref": branch}),
                return_exceptions=False,
            )

            tree_entries: list[dict[str, Any]] = []
            if tree_resp.status_code == 200:
                payload = tree_resp.json()
                tree_entries = payload.get("tree") or []

            readme_text = ""
            if readme_resp.status_code == 200:
                readme_text = _decode_content(readme_resp.json())

            block = (
                f"{_format_metadata(meta)}\n\n"
                f"Top-level tree ({branch}):\n{_format_tree(tree_entries)}\n\n"
                f"README:\n{_truncate(readme_text, MAX_README_CHARS) if readme_text else '(no README)'}"
            )
            return ToolResult(
                tool_id=GITHUB_SPEC.tool_id,
                success=True,
                content=block,
                meta={
                    "owner": owner,
                    "repo": name,
                    "branch": branch,
                    "stars": meta.get("stargazers_count"),
                    "language": meta.get("language"),
                },
            )
    except httpx.HTTPStatusError as exc:
        logger.info(
            "github API HTTP %s for %s/%s",
            exc.response.status_code, owner, name,
        )
        return ToolResult(
            tool_id=GITHUB_SPEC.tool_id,
            success=False,
            error=f"GitHub API HTTP {exc.response.status_code}",
        )
    except Exception as exc:
        logger.exception("analyze_github_repo failed")
        return ToolResult(
            tool_id=GITHUB_SPEC.tool_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def register() -> None:
    get_registry().register(GITHUB_SPEC, handle_github)
