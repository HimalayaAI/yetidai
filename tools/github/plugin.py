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
MAX_LISTED_REPOS = 30

# Default org for "our code / the repo" style questions. Keeping it here
# (not in the system prompt) means the LLM cannot be tricked into asking
# for an arbitrary org's repos by prompt injection — list_github_repos
# only answers for this org unless the caller explicitly overrides.
DEFAULT_ORG = "HimalayaAI"


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
                "default branch. Also parsed automatically from "
                "`/tree/<branch>` and `/blob/<branch>/...` URLs."
            ),
            required=False,
        ),
        ToolParam(
            name="commit_sha",
            type="string",
            description=(
                "Optional commit SHA. When set (or when `repo` is a "
                "`.../commit/<sha>` URL), the tool returns commit "
                "metadata + message + changed-file list instead of the "
                "repo overview. Use this for 'what changed in this "
                "commit' questions."
            ),
            required=False,
            examples=["cb3a55be60ee0c9ffaad01e93b4493f694849731"],
        ),
    ],
    timeout_seconds=20.0,
)


LIST_REPOS_SPEC = ToolSpec(
    tool_id="github.list_org_repos",
    name="list_github_repos",
    description=(
        "List public repositories for a GitHub organization or user. "
        f"Defaults to the official HARL org (`{DEFAULT_ORG}`). Call this "
        "when the user asks for 'your repo / the code / HARL github / "
        "सबै repo / कोड कहाँ छ / list of repos'. Returns name + URL + "
        "short description for each repo, sorted by recent push.\n\n"
        "USE THIS — NEVER fabricate a `github.com/<org>/<repo>` URL from "
        "memory. `github.com/HimalayaAI` is only the org index page; the "
        "repos under it must come from this tool."
    ),
    category=ToolCategory.UTILITY,
    parameters=[
        ToolParam(
            name="org",
            type="string",
            description=(
                f"GitHub org or username. Defaults to `{DEFAULT_ORG}` when "
                "omitted. Pass a different value only if the user "
                "explicitly asks about another org."
            ),
            required=False,
            examples=[DEFAULT_ORG, "anthropics"],
        ),
    ],
    timeout_seconds=15.0,
)


# ── URL / identifier parsing ──────────────────────────────────────

# Tolerant URL parser. Accepts all the GitHub URL shapes users paste:
#   github.com/owner/name
#   github.com/owner/name/tree/<branch>
#   github.com/owner/name/blob/<branch>/path/to/file
#   github.com/owner/name/commit/<sha>
#   github.com/owner/name/pull/<n>
#   github.com/owner/name/issues/<n>
_GITHUB_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[\w.-]+)/"
    r"(?P<name>[\w.-]+?)(?:\.git)?"
    r"(?:/(?P<kind>tree|blob|commit|commits|pull|pulls|issues)/(?P<ref>[^/?#]+)(?:/(?P<sub>[^?#]*))?)?"
    r"/?(?:[?#].*)?$",
    re.IGNORECASE,
)
_OWNER_REPO_RE = re.compile(r"^([\w.-]+)/([\w.-]+)$")


def _parse_repo(raw: str) -> tuple[str, str, str | None, str | None, str | None]:
    """Return (owner, name, branch, commit_sha, file_path).

    Commit URLs set `commit_sha` (branch/file_path stay None). `blob/<ref>/path`
    URLs set both `branch` and `file_path`. Pull-request / issue URLs set
    nothing extra — the caller falls back to the default repo view.
    """
    raw = (raw or "").strip().rstrip("/")
    m = _GITHUB_URL_RE.match(raw)
    if m:
        owner = m.group("owner")
        name = m.group("name")
        kind = (m.group("kind") or "").lower()
        ref = m.group("ref")
        sub = m.group("sub")
        branch: str | None = None
        commit_sha: str | None = None
        file_path: str | None = None
        if kind == "tree":
            branch = ref
        elif kind == "blob":
            branch = ref
            file_path = sub
        elif kind in ("commit", "commits"):
            commit_sha = ref
        # pull/issues: we only keep owner/name; the specific number isn't
        # actionable through the repo endpoints, and analyze_github_repo
        # doesn't claim to read PR/issue bodies.
        return owner, name, branch, commit_sha, file_path
    m = _OWNER_REPO_RE.match(raw)
    if m:
        return m.group(1), m.group(2), None, None, None
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


# ── Org-listing helpers ───────────────────────────────────────────

async def _fetch_org_repos(
    client: httpx.AsyncClient, owner: str,
) -> tuple[bool, list[dict[str, Any]], str | None]:
    """Return (ok, repos, error).

    Hits `/orgs/<owner>/repos` first (the canonical endpoint for orgs).
    If that 404s — meaning `<owner>` is a user account, not an org —
    falls back to `/users/<owner>/repos`. Either way the caller gets
    the full public-repo list without caring which account type it is.
    """
    for path in (f"/orgs/{owner}/repos", f"/users/{owner}/repos"):
        resp = await _api_get(
            client, path, params={"per_page": 100, "type": "public", "sort": "pushed"},
        )
        if resp.status_code == 200:
            data = resp.json()
            return True, data if isinstance(data, list) else [], None
        if resp.status_code == 404:
            continue
        if resp.status_code == 403:
            rl = resp.headers.get("x-ratelimit-remaining")
            return False, [], (
                f"GitHub rate limit hit (remaining={rl}). "
                "Set GITHUB_TOKEN to raise the limit."
            )
        return False, [], f"GitHub API HTTP {resp.status_code} for {path}"
    return False, [], f"account not found: {owner}"


def _format_repo_list(owner: str, repos: list[dict[str, Any]]) -> str:
    if not repos:
        return (
            f"No public repositories found under {owner}. "
            f"Org page: https://github.com/{owner}"
        )
    lines = [
        f"Public repositories under https://github.com/{owner} "
        f"(showing {min(len(repos), MAX_LISTED_REPOS)} of {len(repos)}, "
        "newest-pushed first):"
    ]
    for r in repos[:MAX_LISTED_REPOS]:
        name = r.get("name", "")
        html_url = r.get("html_url", f"https://github.com/{owner}/{name}")
        desc = (r.get("description") or "").strip()
        lang = r.get("language") or ""
        stars = r.get("stargazers_count") or 0
        meta_bits = [b for b in (lang, f"★{stars}" if stars else "") if b]
        meta_str = f" [{' · '.join(meta_bits)}]" if meta_bits else ""
        if desc:
            lines.append(f"- {name}{meta_str} — {html_url} — {desc}")
        else:
            lines.append(f"- {name}{meta_str} — {html_url}")
    if len(repos) > MAX_LISTED_REPOS:
        lines.append(
            f"… ({len(repos) - MAX_LISTED_REPOS} more; see "
            f"https://github.com/{owner})"
        )
    lines.append(f"\nOfficial org page: https://github.com/{owner}")
    return "\n".join(lines)


async def handle_list_repos(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    org = (arguments.get("org") or DEFAULT_ORG).strip() or DEFAULT_ORG
    # Guard: only allow the GitHub username char-set. Rejects weird inputs
    # that could be confused with API paths.
    if not re.fullmatch(r"[\w.-]+", org):
        return ToolResult(
            tool_id=LIST_REPOS_SPEC.tool_id,
            success=False,
            error=f"invalid org name: {org!r}",
        )
    try:
        async with httpx.AsyncClient() as client:
            ok, repos, err = await _fetch_org_repos(client, org)
        if not ok:
            return ToolResult(
                tool_id=LIST_REPOS_SPEC.tool_id,
                success=False,
                error=err or f"could not fetch repos for {org}",
            )
        return ToolResult(
            tool_id=LIST_REPOS_SPEC.tool_id,
            success=True,
            content=_format_repo_list(org, repos),
            meta={"org": org, "count": len(repos)},
        )
    except Exception as exc:
        logger.exception("list_github_repos failed")
        return ToolResult(
            tool_id=LIST_REPOS_SPEC.tool_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ── Handler ───────────────────────────────────────────────────────

async def handle_github(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    repo_raw = arguments.get("repo")
    if not repo_raw or not isinstance(repo_raw, str):
        return ToolResult(tool_id=GITHUB_SPEC.tool_id, success=False, error="Missing repo")

    try:
        owner, name, branch_from_url, commit_from_url, path_from_url = _parse_repo(repo_raw)
    except ValueError as exc:
        return ToolResult(tool_id=GITHUB_SPEC.tool_id, success=False, error=str(exc))

    branch = (arguments.get("branch") or branch_from_url or "").strip() or None
    file_path = (arguments.get("file_path") or path_from_url or "").strip() or None
    commit_sha = (arguments.get("commit_sha") or commit_from_url or "").strip() or None

    try:
        async with httpx.AsyncClient() as client:
            meta_resp = await _api_get(client, f"/repos/{owner}/{name}")
            if meta_resp.status_code == 404:
                # Self-correction: a repo under a known account often 404s
                # because the LLM fabricated the name. Include the real
                # sibling-repo list so the next turn can recover without
                # another tool round-trip.
                siblings_ok, siblings, _ = await _fetch_org_repos(client, owner)
                hint = (
                    "\n\n" + _format_repo_list(owner, siblings)
                    if siblings_ok else ""
                )
                # Success=True so the content (not the error) reaches the
                # LLM via to_tool_message — ToolResult serialises `content`
                # only on success. The meta flag lets callers still see it
                # was a miss.
                return ToolResult(
                    tool_id=GITHUB_SPEC.tool_id,
                    success=True,
                    content=(
                        f"Repo DOES NOT EXIST: {owner}/{name} is not a "
                        f"public repository. Do NOT fabricate a URL for "
                        f"it in the final answer.{hint}"
                    ),
                    meta={"owner": owner, "repo": name, "not_found": True},
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

            # Commit mode — return commit metadata + message + changed-file list.
            # Triggered when the URL is .../commit/<sha>, or when the caller
            # passed commit_sha explicitly. Answers "यो commit मा के change छ?".
            if commit_sha:
                commit_resp = await _api_get(
                    client, f"/repos/{owner}/{name}/commits/{commit_sha}",
                )
                if commit_resp.status_code == 404:
                    return ToolResult(
                        tool_id=GITHUB_SPEC.tool_id,
                        success=False,
                        error=f"commit not found: {commit_sha}",
                    )
                commit_resp.raise_for_status()
                commit = commit_resp.json()
                info = commit.get("commit", {}) or {}
                author = (info.get("author") or {}).get("name") or "—"
                date_s = (info.get("author") or {}).get("date") or "—"
                message = info.get("message") or ""
                stats = commit.get("stats") or {}
                files = commit.get("files") or []
                file_lines: list[str] = []
                for f in files[:30]:
                    status = f.get("status", "?")
                    path = f.get("filename", "?")
                    additions = f.get("additions", 0)
                    deletions = f.get("deletions", 0)
                    file_lines.append(
                        f"  [{status}] {path}  (+{additions} / -{deletions})"
                    )
                if len(files) > 30:
                    file_lines.append(f"  … ({len(files) - 30} more files)")
                block = (
                    f"{_format_metadata(meta)}\n\n"
                    f"──── commit {commit_sha[:12]} ────\n"
                    f"Author: {author}  |  Date: {date_s}\n"
                    f"Stats: +{stats.get('additions', 0)} / "
                    f"-{stats.get('deletions', 0)}  |  "
                    f"{len(files)} files changed\n\n"
                    f"Message:\n{_truncate(message, 1200)}\n\n"
                    f"Files:\n" + ("\n".join(file_lines) if file_lines else "  (no files listed)")
                )
                return ToolResult(
                    tool_id=GITHUB_SPEC.tool_id,
                    success=True,
                    content=block,
                    meta={
                        "owner": owner,
                        "repo": name,
                        "commit_sha": commit_sha,
                        "files_changed": len(files),
                    },
                )

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
    get_registry().register(LIST_REPOS_SPEC, handle_list_repos)
