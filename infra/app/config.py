"""Central configuration. Override via environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

# Load .env from the package root (two levels up from this file)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_bytes().decode("utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


@dataclass
class Settings:
    # Teammate's retrieval API
    pipeline_url: str = os.getenv("PIPELINE_URL", "http://54.184.212.11:8000")

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Haiku 4.5 is the sweet spot for structured extraction: fast + cheap.
    # Budget math: ~1.5k in + 0.5k out per query ≈ $0.0032/query on Haiku 4.5.
    # With $25 that's ~7,800 queries, plenty for a datathon.
    extractor_model: str = os.getenv("EXTRACTOR_MODEL", "claude-haiku-4-5-20251001")
    # Sonnet 4.6 for final explanations where writing quality matters.
    # Used sparingly (top-K only) so cost stays low.
    explainer_model: str = os.getenv("EXPLAINER_MODEL", "claude-sonnet-4-6")

    # Ranking knobs
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "20"))
    # How many candidates we pull from the API before re-ranking.
    # Bigger = better reranking but slower. 50 is a good default.
    rerank_pool_size: int = int(os.getenv("RERANK_POOL_SIZE", "50"))

    # AWS / profile store (optional)
    profile_table_name: str = os.getenv("PROFILE_TABLE_NAME", "robin-user-profiles")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")


settings = Settings()
