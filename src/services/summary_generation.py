import io
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import requests
from openai import OpenAI
from pypdf import PdfReader
import dotenv

dotenv.load_dotenv()

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover - optional runtime dep
    boto3 = None  # type: ignore

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - optional runtime dep
    storage = None  # type: ignore


S3_URL_PATTERN = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")
GCS_URL_PATTERN = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<blob>.+)$")


class SummaryGenerationError(Exception):
    pass


@dataclass
class SummaryConfig:
    model: str = "gpt-4o-mini"
    max_input_chars_per_chunk: int = 12000
    overlap_chars: int = 800
    max_reduce_passes: int = 3
    temperature: float = 0.2


def _read_bytes_from_http(url: str, timeout_sec: int = 60) -> bytes:
    response = requests.get(url, timeout=timeout_sec)
    response.raise_for_status()
    return response.content


def _read_bytes_from_s3(url: str) -> bytes:
    if boto3 is None:
        raise SummaryGenerationError(
            "boto3 not installed. Add boto3 to requirements or install it to read s3 URLs."
        )
    match = S3_URL_PATTERN.match(url)
    if not match:
        raise SummaryGenerationError(f"Invalid s3 URL: {url}")
    bucket = match.group("bucket")
    key = match.group("key")
    session = boto3.session.Session()
    s3 = session.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def _read_bytes_from_gcs(url: str) -> bytes:
    if storage is None:
        raise SummaryGenerationError(
            "google-cloud-storage not installed. Install it to read gs URLs."
        )
    match = GCS_URL_PATTERN.match(url)
    if not match:
        raise SummaryGenerationError(f"Invalid gs URL: {url}")
    bucket_name = match.group("bucket")
    blob_name = match.group("blob")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def _fetch_bytes(url: str) -> bytes:
    if url.startswith("s3://"):
        return _read_bytes_from_s3(url)
    if url.startswith("gs://"):
        return _read_bytes_from_gcs(url)
    if url.startswith("http://") or url.startswith("https://"):
        return _read_bytes_from_http(url)
    # Local path fallback
    if os.path.exists(url):
        with open(url, "rb") as f:
            return f.read()
    raise SummaryGenerationError(
        "Unsupported URL scheme. Use http(s)://, s3://, gs://, or a local file path."
    )


def _is_pdf(data: bytes, url: str) -> bool:
    if data[:5] == b"%PDF-":
        return True
    return url.lower().endswith(".pdf")


def _is_text(data: bytes, url: str) -> bool:
    if url.lower().endswith(".txt"):
        return True
    # Heuristic: if decodable as utf-8 and contains mostly text
    try:
        text = data.decode("utf-8")
        # If too many control characters, likely not plain text
        control_ratio = sum(1 for c in text if ord(c) < 9 or (13 < ord(c) < 32)) / max(
            1, len(text)
        )
        return control_ratio < 0.01
    except Exception:
        return False


def _extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        pages.append(page_text)
    return "\n\n".join(pages).strip()


def _extract_text(data: bytes, url: str) -> str:
    if _is_pdf(data, url):
        return _extract_text_from_pdf(data)
    if _is_text(data, url):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception as exc:  # Fallback
            raise SummaryGenerationError(f"Failed to decode text: {exc}")
    raise SummaryGenerationError(
        "Unsupported file type. Only PDF and UTF-8 text are supported currently."
    )


def _chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    end = len(text)
    while start < end:
        chunk_end = min(end, start + max_chars)
        chunk = text[start:chunk_end]
        chunks.append(chunk)
        if chunk_end >= end:
            break
        start = chunk_end - overlap
        if start < 0:
            start = 0
    return chunks


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummaryGenerationError(
            "OPENAI_API_KEY is not set. Please set it in the environment."
        )
    return OpenAI(api_key=api_key)


def _map_summarize(client: OpenAI, model: str, chunk: str, temperature: float) -> str:
    system_prompt = (
        "You are a senior analyst and expert narrator. Write vivid, structured summaries "
        "optimized for spoken narration. Use concise sentences, logical flow, and clear "
        "headings with short bullet points where helpful. Avoid redundancy."
    )
    user_prompt = (
        "Summarize the following content for narration. Capture key ideas, decisions, "
        "data points, and any processes described. Provide a strong narrative arc, a "
        "one-sentence high-level overview, and callouts for critical quotes or figures.\n\n"
        f"CONTENT:\n{chunk}"
    )
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def _reduce_summaries(
    client: OpenAI,
    model: str,
    partial_summaries: List[str],
    temperature: float,
) -> str:
    system_prompt = (
        "You merge partial narration summaries into a single, cohesive narrative. "
        "Preserve fidelity to the source, remove redundancy, and maintain a clear "
        "beginning, middle, and end. Optimize for voice-over narration."
    )
    joined = "\n\n---\n\n".join(partial_summaries)
    user_prompt = (
        "Merge the following partial summaries into one comprehensive narration-ready "
        "summary with: (1) high-level overview, (2) thematic sections with clear "
        "headings, (3) concise bullet points for details, (4) conclusion with key "
        "takeaways.\n\nPARTIAL SUMMARIES:\n\n" + joined
    )
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def generate_narration_summary(url: str, config: Optional[SummaryConfig] = None) -> str:
    cfg = config or SummaryConfig()

    # data = _fetch_bytes(url)
    # text = _extract_text(data, url)
    '''TODO: Uncomment this when we have a PDF file or s3 gs url and remove the hardcoded path'''
    data = open(
        "/home/mathewvkariath/Desktop/keycode_25/standin-ai/confluence.txt", "rb"
    ).read()
    text = data.decode("utf-8", errors="ignore")
    if not text.strip():
        raise SummaryGenerationError("No extractable text found in the provided document.")

    chunks = _chunk_text(text, cfg.max_input_chars_per_chunk, cfg.overlap_chars)
    client = _get_openai_client()

    # Map step: summarize each chunk
    partials: List[str] = []
    for chunk in chunks:
        partials.append(_map_summarize(client, cfg.model, chunk, cfg.temperature))

    # Reduce step: iteratively merge until one remains or max passes reached
    passes = 0
    while len(partials) > 1 and passes < cfg.max_reduce_passes:
        merged: List[str] = []
        for i in range(0, len(partials), 4):
            group = partials[i : i + 4]
            merged.append(_reduce_summaries(client, cfg.model, group, cfg.temperature))
        partials = merged
        passes += 1

    final_summary = partials[0] if partials else ""
    if not final_summary:
        raise SummaryGenerationError("Failed to generate summary.")
    return final_summary


def _print_usage() -> None:
    print(
        "Usage: python -m src.services.summary_generation <url_or_path> [model]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(2)
    target = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else None
    cfg = SummaryConfig(model=model) if model else SummaryConfig()
    try:
        summary = generate_narration_summary(target, cfg)
        print(summary)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
