"""Backend-independent ranked web search and explicit keyless lookup."""

import json
import os
import socket
import ssl
from collections import deque
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPSHandler, HTTPRedirectHandler, Request, build_opener

SEARCH_TIMEOUT = 15
SEARCH_RESPONSE_LIMIT = 1_000_000
PROVIDERS = ("tavily", "instant-answer")


class _NoRedirect(HTTPRedirectHandler):
    # Never forward a search query or API key to a redirected host.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _setting(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name].strip()
    from dotenv import dotenv_values

    value = dotenv_values(Path(__file__).resolve().with_name(".env"), interpolate=False).get(name)
    return value.strip() if value is not None else default


def _text(value, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _normalize(rows, max_results: int) -> list[dict[str, str]]:
    results = []
    seen = set()
    for title, url, snippet in rows:
        if not isinstance(url, str) or len(url) > 2000 or any(c.isspace() for c in url):
            continue
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username is not None:
                continue
        except ValueError:
            continue
        # Fragments don't identify distinct pages; keep the original link for citations.
        identity = parsed._replace(fragment="").geturl()
        if identity in seen:
            continue
        seen.add(identity)
        results.append({"title": _text(title, 300), "url": url, "snippet": _text(snippet, 1000)})
        if len(results) == max_results:
            break
    return results


def _array(value) -> list:
    if not isinstance(value, list):
        raise ValueError("Expected an array")
    return value


def _tavily_rows(data):
    for item in _array(data.get("results")):
        if isinstance(item, dict):
            yield item.get("title"), item.get("url"), item.get("content")


def _instant_rows(data):
    direct = _array(data.get("Results", []))
    related = _array(data.get("RelatedTopics", []))
    yield data.get("Heading"), data.get("AbstractURL"), data.get("AbstractText")
    yield data.get("Heading"), data.get("DefinitionURL"), data.get("Definition")
    pending = deque(direct + related)
    while pending:
        item = pending.popleft()
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("Topics"), list):
            pending.extendleft(reversed(item["Topics"]))
        else:
            yield item.get("Text"), item.get("FirstURL"), item.get("Text")


def search_web(query: str, max_results: int = 5) -> dict:
    """Agent tool: use the configured provider without a model-controlled override."""
    return search(query, max_results)


def search(query: str, max_results: int = 5, *, provider: str | None = None) -> dict:
    """Search using app configuration or an explicit provider for comparisons."""
    def error(code, message, *, retryable=False):
        return {"provider": provider, "status": "error", "error_code": code,
                "error": message, "retryable": retryable}

    if not isinstance(query, str) or not query.strip():
        return error("invalid_query", "query must be a non-empty string")
    query = query.strip()
    if len(query) > 400 or len(query.split()) > 50:
        return error("invalid_query", "query must be at most 400 characters and 50 words")
    if type(max_results) is not int or not 1 <= max_results <= 10:
        return error("invalid_count", "max_results must be an integer from 1 to 10")

    try:
        if provider is None:
            provider = _setting("SEARCH_PROVIDER", "tavily")
        if provider not in PROVIDERS:
            return error("invalid_provider", "Set SEARCH_PROVIDER to tavily or instant-answer")
        headers = {"User-Agent": "hello-world-agent/1.0", "Accept": "application/json"}
        if provider == "tavily":
            key = _setting("TAVILY_API_KEY")
            if not key or any(not 33 <= ord(c) <= 126 for c in key):
                return error("missing_credentials", "Set a valid TAVILY_API_KEY or explicitly select SEARCH_PROVIDER=instant-answer for limited keyless lookup")
            headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            body = json.dumps({"query": query, "max_results": max_results, "search_depth": "basic",
                               "auto_parameters": False, "include_answer": False,
                               "include_raw_content": False}).encode("utf-8")
            request = Request("https://api.tavily.com/search", data=body, headers=headers, method="POST")
        else:
            params = {"q": query, "format": "json", "no_html": 1, "no_redirect": 1}
            request = Request("https://api.duckduckgo.com/?" + urlencode(params), headers=headers)

        import certifi

        context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or certifi.where())
        opener = build_opener(HTTPSHandler(context=context), _NoRedirect())

        with opener.open(request, timeout=SEARCH_TIMEOUT) as response:
            raw = response.read(SEARCH_RESPONSE_LIMIT + 1)
        if len(raw) > SEARCH_RESPONSE_LIMIT:
            return error("response_too_large", "Search response exceeded size limit")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        rows = _tavily_rows(data) if provider == "tavily" else _instant_rows(data)
        results = _normalize(rows, max_results)
        observation = {"provider": provider, "status": "ok" if results else "no_results",
                       "query": query, "results": results}
        if provider == "instant-answer":
            answer = _text(data.get("Answer"), 1000)
            observation.update(answer=answer, answer_type=_text(data.get("AnswerType"), 100),
                               note="Instant Answer is limited lookup, not ranked web search.")
            if answer:
                observation["status"] = "ok"
        return observation
    except (ValueError, TypeError, RecursionError):
        return error("invalid_response", "Search provider returned an invalid response")
    except ImportError:
        return error("missing_dependency", "Run python3 -m pip install -r requirements.txt to enable web search")
    except HTTPError as exc:
        exc.close()
        if provider == "tavily":
            if exc.code in (401, 403):
                return error("auth", f"Tavily returned HTTP {exc.code}; check the API key and account access")
            if exc.code in (432, 433):
                return error("quota", f"Tavily usage limit reached (HTTP {exc.code}); check the account's credits and limits")
        if exc.code == 429:
            return error("rate_limit", "Search provider rate limit reached (HTTP 429)", retryable=True)
        return error("http", f"Search provider returned HTTP {exc.code}", retryable=500 <= exc.code <= 599)
    except (URLError, OSError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        if isinstance(reason, ssl.SSLCertVerificationError):
            return error("tls", "TLS certificate verification failed; check certifi or SSL_CERT_FILE")
        if isinstance(reason, socket.gaierror):
            return error("dns", "Search provider DNS lookup failed")
        if isinstance(reason, TimeoutError):
            return error("timeout", "Search provider timed out", retryable=True)
        return error("connection", "Search connection or configuration failed; check the network and SSL_CERT_FILE")
