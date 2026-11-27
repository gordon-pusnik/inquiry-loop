"""Interactive model discovery and downloads through the Hugging Face Hub."""

from itertools import zip_longest
import re

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled,
)


SEARCH_LIMIT = 5
_GGUF_SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})(\.gguf)", re.IGNORECASE)
_GGUF_AUXILIARY = re.compile(r"(?:^|[-_.])mmproj(?:[-_.]|$)|(?:^|[-_.])imatrix\.gguf$")
_SEARCH_FORMATS = {"--mlx": "mlx", "--gguf": "gguf"}


def _search(api, arguments):
    formats = ("mlx", "gguf")
    if arguments and arguments[0] in _SEARCH_FORMATS:
        formats = (_SEARCH_FORMATS[arguments[0]],)
        arguments = arguments[1:]
    if not arguments or arguments[0].startswith("--"):
        raise ValueError("Usage: /search [--mlx|--gguf] TERM")

    query = " ".join(arguments)
    matches = [
        [(model.id, model_format) for model in api.list_models(
            search=query, filter=model_format, limit=SEARCH_LIMIT, sort="downloads")]
        for model_format in formats
    ]
    # Alternate formats so popularity in either format cannot fill the shortlist.
    shown = set()
    for row in zip_longest(*matches):
        for match in row:
            if match is None or match[0] in shown:
                continue
            repo_id, model_format = match
            shown.add(repo_id)
            print(f"{len(shown)}. {repo_id} ({model_format})")
            if len(shown) == SEARCH_LIMIT:
                return
    if not shown:
        print("No matching Hugging Face models found.")


def _choose_gguf(files, repo_id):
    candidates = []
    for filename in files:
        basename = filename.rsplit("/", 1)[-1].lower()
        if not basename.endswith(".gguf") or _GGUF_AUXILIARY.search(basename):
            continue
        shard = _GGUF_SHARD.fullmatch(filename)
        if shard is None or shard[2] == "00001":
            candidates.append(filename)
    if len(candidates) > 1:
        print(f"{repo_id} contains multiple GGUF variants. Choose one with /download {repo_id} FILENAME:")
        for filename in sorted(candidates):
            print(f"  {filename}")
        return None
    if candidates:
        return candidates[0]
    if any(name.lower().endswith(".gguf") for name in files):
        raise ValueError("No complete GGUF entry file found. Select a file explicitly with /download OWNER/REPO FILENAME.")
    return ""


def _download(api, repo_id, filename=None):
    model = api.model_info(repo_id)
    files = {item.rfilename for item in model.siblings or []}
    if filename is None:
        filename = _choose_gguf(files, repo_id)
        if filename is None:
            return
    elif filename not in files:
        raise ValueError(f"File {filename!r} was not found in {repo_id}.")

    if not filename:
        print(f"Downloading {repo_id} to the Hugging Face cache...", flush=True)
        path = api.snapshot_download(repo_id, revision=model.sha)
        print(f"Cached model: {path}")
        return

    shard = _GGUF_SHARD.fullmatch(filename)
    filenames = [filename]
    if shard:
        stem, _, count, extension = shard.groups()
        filenames = [f"{stem}-{index:05d}-of-{count}{extension}" for index in range(1, int(count) + 1)]
        if not filenames or not files.issuperset(filenames):
            raise ValueError(f"{repo_id} is missing one or more shards for {filename}.")
    print(f"Downloading {repo_id}/{filenames[0]} to the Hugging Face cache...", flush=True)
    path = api.hf_hub_download(repo_id, filenames[0], revision=model.sha)
    for other in filenames[1:]:
        api.hf_hub_download(repo_id, other, revision=model.sha)
    print(f"Cached file: {path}")


def run_model_command(command, arguments):
    api = HfApi()
    try:
        if command == "/search":
            _search(api, arguments)
        elif not arguments or arguments[0].count("/") != 1:
            raise ValueError("Usage: /download OWNER/REPO [FILENAME]")
        else:
            filename = " ".join(arguments[1:]) or None
            _download(api, arguments[0], filename)
    except GatedRepoError:
        raise ValueError("Access is gated. Request access on Hugging Face and sign in with hf auth login.") from None
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise ValueError(f"Hugging Face rejected the request (HTTP {status}). Check the model ID and your access.") from None
    except (httpx.RequestError, LocalEntryNotFoundError, OfflineModeIsEnabled):
        raise ValueError("Could not reach Hugging Face. Check connectivity or offline mode.") from None
    except (OSError, RuntimeError):
        raise ValueError("Hugging Face model operation failed. Check disk space, cache permissions, and connectivity.") from None
