"""
GitHub MCP Server
-----------------
An MCP server that gives AI agents access to the GitHub REST API.
Covers: repo search, issue listing, file reading, issue creation,
PR summarization, and stateful session context.
 
Setup:
    pip install mcp httpx python-dotenv
    export GITHUB_TOKEN=your_personal_access_token
    python server.py
"""
 
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Optional
 
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator
 
load_dotenv()
 
# ── Constants ────────────────────────────────────────────────────────────────
 
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
DEFAULT_PER_PAGE = 20
MAX_PER_PAGE = 100
 
# ── Session state (stateful context — week 3 skill) ──────────────────────────
# Stores the "active" repo so agents don't repeat it every call.
 
_session: dict[str, str] = {
    "owner": "",
    "repo": "",
}
 
# ── Shared HTTP client (created once, reused across all tools) ───────────────
 
@asynccontextmanager
async def app_lifespan():
    """Create a single async HTTP client for the lifetime of the server."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
 
    async with httpx.AsyncClient(
        base_url=GITHUB_API_BASE,
        headers=headers,
        timeout=15.0,
    ) as client:
        yield {"client": client}
 
 
mcp = FastMCP("github_mcp", lifespan=app_lifespan)
 
 
# ── Shared helpers ────────────────────────────────────────────────────────────
 
def _handle_error(e: Exception) -> str:
    """Turn HTTP/network errors into clear, actionable messages."""
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 401:
            return "Error: Unauthorized. Check that GITHUB_TOKEN is set and valid."
        if code == 403:
            return "Error: Forbidden. Your token may lack the required scope."
        if code == 404:
            return "Error: Not found. Double-check the owner, repo, or path."
        if code == 422:
            detail = e.response.json().get("message", "")
            return f"Error: Unprocessable request — {detail}"
        if code == 429:
            return "Error: Rate limit hit. Wait a moment before retrying."
        return f"Error: GitHub API returned {code}."
    if isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. GitHub may be slow — try again."
    return f"Error: Unexpected error — {type(e).__name__}: {e}"
 
 
def _resolve(owner: Optional[str], repo: Optional[str]) -> tuple[str, str]:
    """
    Return (owner, repo), falling back to the active session values.
    Raises ValueError if neither the call nor the session provides them.
    """
    resolved_owner = owner or _session["owner"]
    resolved_repo = repo or _session["repo"]
    if not resolved_owner or not resolved_repo:
        raise ValueError(
            "No owner/repo provided and no active session. "
            "Call github_set_context first, or pass owner and repo explicitly."
        )
    return resolved_owner, resolved_repo
 
 
def _format_repo(r: dict[str, Any]) -> dict[str, Any]:
    """Pick the fields that matter from a raw GitHub repo object."""
    return {
        "full_name": r.get("full_name"),
        "description": r.get("description"),
        "stars": r.get("stargazers_count"),
        "language": r.get("language"),
        "open_issues": r.get("open_issues_count"),
        "url": r.get("html_url"),
        "updated_at": r.get("updated_at"),
    }
 
 
def _format_issue(i: dict[str, Any]) -> dict[str, Any]:
    """Pick the fields that matter from a raw GitHub issue object."""
    return {
        "number": i.get("number"),
        "title": i.get("title"),
        "state": i.get("state"),
        "author": i.get("user", {}).get("login"),
        "labels": [lb["name"] for lb in i.get("labels", [])],
        "comments": i.get("comments"),
        "created_at": i.get("created_at"),
        "url": i.get("html_url"),
        "body_preview": (i.get("body") or "")[:300],
    }
 
 
# ── Pydantic input models ─────────────────────────────────────────────────────
 
class SetContextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
 
    owner: str = Field(..., description="GitHub username or org (e.g. 'BerriAI')", min_length=1, max_length=100)
    repo: str = Field(..., description="Repository name (e.g. 'litellm')", min_length=1, max_length=100)
 
    @field_validator("owner", "repo")
    @classmethod
    def no_slashes(cls, v: str) -> str:
        if "/" in v:
            raise ValueError("Pass owner and repo separately — no slashes.")
        return v
 
 
class SearchReposInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
 
    query: str = Field(..., description="GitHub search query (e.g. 'LLM gateway stars:>1000')", min_length=1, max_length=256)
    limit: int = Field(default=10, description="Max results to return", ge=1, le=MAX_PER_PAGE)
 
 
class ListIssuesInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
 
    owner: Optional[str] = Field(default=None, description="GitHub username or org. Uses session if omitted.")
    repo: Optional[str] = Field(default=None, description="Repository name. Uses session if omitted.")
    state: str = Field(default="open", description="Filter by state: 'open', 'closed', or 'all'")
    label: Optional[str] = Field(default=None, description="Filter by label name (e.g. 'bug', 'enhancement')")
    limit: int = Field(default=DEFAULT_PER_PAGE, description="Max results", ge=1, le=MAX_PER_PAGE)
 
    @field_validator("state")
    @classmethod
    def valid_state(cls, v: str) -> str:
        if v not in {"open", "closed", "all"}:
            raise ValueError("state must be 'open', 'closed', or 'all'")
        return v
 
 
class ReadFileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
 
    path: str = Field(..., description="File path inside the repo (e.g. 'README.md', 'src/main.py')", min_length=1)
    owner: Optional[str] = Field(default=None, description="GitHub username or org. Uses session if omitted.")
    repo: Optional[str] = Field(default=None, description="Repository name. Uses session if omitted.")
    ref: Optional[str] = Field(default=None, description="Branch, tag, or commit SHA. Defaults to default branch.")
 
    @field_validator("path")
    @classmethod
    def no_traversal(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("Path traversal ('..') is not allowed.")
        return v.lstrip("/")
 
 
class CreateIssueInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
 
    title: str = Field(..., description="Issue title", min_length=1, max_length=256)
    body: Optional[str] = Field(default=None, description="Issue body (Markdown supported)", max_length=65536)
    labels: Optional[list[str]] = Field(default=None, description="Label names to apply", max_length=10)
    owner: Optional[str] = Field(default=None, description="GitHub username or org. Uses session if omitted.")
    repo: Optional[str] = Field(default=None, description="Repository name. Uses session if omitted.")
 
 
class SummarizePRInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
 
    pr_number: int = Field(..., description="Pull request number", ge=1)
    owner: Optional[str] = Field(default=None, description="GitHub username or org. Uses session if omitted.")
    repo: Optional[str] = Field(default=None, description="Repository name. Uses session if omitted.")
 
 
# ── Tools ─────────────────────────────────────────────────────────────────────
 
@mcp.tool(
    name="github_set_context",
    annotations={
        "title": "Set active repository context",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def github_set_context(params: SetContextInput) -> str:
    """
    Set the active GitHub repository for this session.
 
    Once set, all other tools will use this owner/repo by default
    so you don't have to repeat it every call.
 
    Args:
        params.owner: GitHub username or org (e.g. 'BerriAI')
        params.repo:  Repository name (e.g. 'litellm')
 
    Returns:
        Confirmation string with the active context.
    """
    _session["owner"] = params.owner
    _session["repo"] = params.repo
    return f"Active context set to {params.owner}/{params.repo}."
 
 
@mcp.tool(
    name="github_search_repos",
    annotations={
        "title": "Search GitHub repositories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_search_repos(params: SearchReposInput, ctx: Any) -> str:
    """
    Search GitHub for repositories matching a query.
 
    Supports GitHub's full search syntax:
    'LLM gateway language:python stars:>500'
 
    Args:
        params.query: GitHub search query string
        params.limit: Maximum number of results (1-100)
 
    Returns:
        JSON list of repositories with name, description, stars, language, URL.
    """
    client: httpx.AsyncClient = ctx.request_context.lifespan_state["client"]
    try:
        resp = await client.get(
            "/search/repositories",
            params={"q": params.query, "per_page": params.limit, "sort": "stars"},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return json.dumps([_format_repo(r) for r in items], indent=2)
    except Exception as e:
        return _handle_error(e)
 
 
@mcp.tool(
    name="github_list_issues",
    annotations={
        "title": "List issues in a repository",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_list_issues(params: ListIssuesInput, ctx: Any) -> str:
    """
    List issues for a GitHub repository.
 
    Falls back to the session context if owner/repo are not provided.
 
    Args:
        params.owner:  GitHub username or org (optional if session is set)
        params.repo:   Repository name (optional if session is set)
        params.state:  'open', 'closed', or 'all'
        params.label:  Filter by label name (optional)
        params.limit:  Max results to return
 
    Returns:
        JSON list of issues with number, title, state, author, labels, URL.
    """
    client: httpx.AsyncClient = ctx.request_context.lifespan_state["client"]
    try:
        owner, repo = _resolve(params.owner, params.repo)
        query: dict[str, Any] = {"state": params.state, "per_page": params.limit}
        if params.label:
            query["labels"] = params.label
 
        resp = await client.get(f"/repos/{owner}/{repo}/issues", params=query)
        resp.raise_for_status()
        issues = [i for i in resp.json() if "pull_request" not in i]  # exclude PRs
        return json.dumps([_format_issue(i) for i in issues], indent=2)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return _handle_error(e)
 
 
@mcp.tool(
    name="github_read_file",
    annotations={
        "title": "Read a file from a repository",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_read_file(params: ReadFileInput, ctx: Any) -> str:
    """
    Read the contents of a file from a GitHub repository.
 
    Falls back to the session context if owner/repo are not provided.
    Binary files are not supported — text/code files only.
 
    Args:
        params.path:  File path inside the repo (e.g. 'README.md')
        params.owner: GitHub username or org (optional if session is set)
        params.repo:  Repository name (optional if session is set)
        params.ref:   Branch, tag, or commit SHA (optional)
 
    Returns:
        Raw file content as a string, or an error message.
    """
    client: httpx.AsyncClient = ctx.request_context.lifespan_state["client"]
    try:
        owner, repo = _resolve(params.owner, params.repo)
        query = {}
        if params.ref:
            query["ref"] = params.ref
 
        resp = await client.get(
            f"/repos/{owner}/{repo}/contents/{params.path}",
            params=query,
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        resp.raise_for_status()
        return resp.text
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return _handle_error(e)
 
 
@mcp.tool(
    name="github_create_issue",
    annotations={
        "title": "Create a new issue",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def github_create_issue(params: CreateIssueInput, ctx: Any) -> str:
    """
    Create a new issue in a GitHub repository.
 
    Falls back to the session context if owner/repo are not provided.
    Requires a token with 'repo' scope.
 
    Args:
        params.title:  Issue title (required)
        params.body:   Issue body in Markdown (optional)
        params.labels: List of label names to apply (optional)
        params.owner:  GitHub username or org (optional if session is set)
        params.repo:   Repository name (optional if session is set)
 
    Returns:
        JSON with the new issue number, title, and URL.
    """
    client: httpx.AsyncClient = ctx.request_context.lifespan_state["client"]
    try:
        owner, repo = _resolve(params.owner, params.repo)
        payload: dict[str, Any] = {"title": params.title}
        if params.body:
            payload["body"] = params.body
        if params.labels:
            payload["labels"] = params.labels
 
        resp = await client.post(f"/repos/{owner}/{repo}/issues", json=payload)
        resp.raise_for_status()
        issue = resp.json()
        return json.dumps({
            "number": issue["number"],
            "title": issue["title"],
            "url": issue["html_url"],
            "state": issue["state"],
        }, indent=2)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return _handle_error(e)
 
 
@mcp.tool(
    name="github_summarize_pr",
    annotations={
        "title": "Summarize a pull request",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def github_summarize_pr(params: SummarizePRInput, ctx: Any) -> str:
    """
    Fetch and summarize a GitHub pull request: metadata, changed files, and comments.
 
    This tool chains three API calls (PR details, files, comments) to give
    a complete picture in one response.
 
    Falls back to the session context if owner/repo are not provided.
 
    Args:
        params.pr_number: Pull request number
        params.owner:     GitHub username or org (optional if session is set)
        params.repo:      Repository name (optional if session is set)
 
    Returns:
        JSON summary with title, author, state, changed files, and recent comments.
    """
    client: httpx.AsyncClient = ctx.request_context.lifespan_state["client"]
    try:
        owner, repo = _resolve(params.owner, params.repo)
        base = f"/repos/{owner}/{repo}/pulls/{params.pr_number}"
 
        # Chain three API calls concurrently
        import asyncio
        pr_resp, files_resp, comments_resp = await asyncio.gather(
            client.get(base),
            client.get(f"{base}/files", params={"per_page": 30}),
            client.get(f"{base}/comments", params={"per_page": 10}),
        )
        for r in (pr_resp, files_resp, comments_resp):
            r.raise_for_status()
 
        pr = pr_resp.json()
        files = files_resp.json()
        comments = comments_resp.json()
 
        summary = {
            "number": pr["number"],
            "title": pr["title"],
            "state": pr["state"],
            "author": pr["user"]["login"],
            "base_branch": pr["base"]["ref"],
            "head_branch": pr["head"]["ref"],
            "mergeable": pr.get("mergeable"),
            "additions": pr["additions"],
            "deletions": pr["deletions"],
            "changed_files": [
                {
                    "filename": f["filename"],
                    "status": f["status"],
                    "changes": f["changes"],
                }
                for f in files
            ],
            "recent_comments": [
                {
                    "author": c["user"]["login"],
                    "body_preview": c["body"][:200],
                    "path": c.get("path"),
                }
                for c in comments[:5]
            ],
            "url": pr["html_url"],
        }
        return json.dumps(summary, indent=2)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return _handle_error(e)
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("⚠️  Warning: GITHUB_TOKEN not set. Unauthenticated requests are rate-limited to 60/hr.")
    print("Starting GitHub MCP server (stdio transport)...")
    mcp.run()