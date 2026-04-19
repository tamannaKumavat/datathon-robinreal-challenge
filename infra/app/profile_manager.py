"""User profile persistence + rewriting.

Two storage backends:
  - InMemoryProfileStore  (dev / testing)
  - DynamoDBProfileStore  (prod on AWS)

Plus a ClaudeProfileRewriter that updates the profile based on a new query
and the user's reaction to results (favourites, hides, clicks). The rewriter
is conservative: it only moves durable preferences, never stores ephemeral
things like "I want a place available next June".
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from typing import Optional, Protocol

import anthropic

from .config import settings
from .schemas import ExtractedQuery, UserProfile

log = logging.getLogger(__name__)


# ---------- Storage ----------

class ProfileStore(Protocol):
    def get(self, user_id: str) -> Optional[UserProfile]: ...
    def put(self, profile: UserProfile) -> None: ...


class InMemoryProfileStore:
    def __init__(self):
        self._data: dict[str, UserProfile] = {}
        self._lock = threading.Lock()

    def get(self, user_id: str) -> Optional[UserProfile]:
        with self._lock:
            return self._data.get(user_id)

    def put(self, profile: UserProfile) -> None:
        with self._lock:
            profile.last_updated = dt.datetime.utcnow().isoformat()
            self._data[profile.user_id] = profile


class DynamoDBProfileStore:
    """Simple DynamoDB-backed store.

    Table schema: partition key `user_id` (string), single item per user,
    profile JSON in attribute `profile`.
    """
    def __init__(self, table_name: str = None, region: str = None):
        import boto3  # local import to keep dev lightweight
        self.table = boto3.resource(
            "dynamodb",
            region_name=region or settings.aws_region,
        ).Table(table_name or settings.profile_table_name)

    def get(self, user_id: str) -> Optional[UserProfile]:
        try:
            resp = self.table.get_item(Key={"user_id": user_id})
        except Exception as e:
            log.warning("DynamoDB get failed for %s: %s", user_id, e)
            return None
        item = resp.get("Item")
        if not item:
            return None
        try:
            return UserProfile.model_validate_json(item["profile"])
        except Exception:
            log.exception("Corrupt profile for %s", user_id)
            return None

    def put(self, profile: UserProfile) -> None:
        profile.last_updated = dt.datetime.utcnow().isoformat()
        self.table.put_item(
            Item={
                "user_id": profile.user_id,
                "profile": profile.model_dump_json(),
                "updated_at": profile.last_updated,
            }
        )


# ---------- Rewriter ----------

REWRITER_SYSTEM = """You update a durable User Profile for a real-estate search user.

You receive:
  - the CURRENT profile (JSON)
  - the user's LATEST query
  - the ExtractedQuery the system produced
  - OPTIONAL behavioural signals (favourited/hidden listing ids this turn)

Output a JSON object with the SAME UserProfile schema, representing the UPDATED profile.
Rules:
  1. Only move DURABLE preferences into the profile. A one-off "available from March"
     is NOT durable. "I always want 2+ bedrooms" IS.
  2. For numeric defaults (budget, rooms), update only if the user has stated the
     same range in at least 2 recent queries OR they said something like "my usual".
     Use `recent_query_summaries` to check history.
  3. For `semantic_preferences`, maintain a decayed moving average:
       new_weight = 0.7 * old_weight + 0.3 * current_query_weight
     Clip to [0, 1]. Drop keys below 0.1.
  4. For `preferred_features`, add features that appear in >= 2 recent queries.
     Remove any the user's latest query explicitly excludes.
  5. Append a <=12-word summary of this query to `recent_query_summaries`
     (keep the list to at most 20 entries; drop the oldest).
  6. Never remove the `user_id`.
  7. Output JSON only. No prose.
"""


class ClaudeProfileRewriter:
    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def rewrite(
        self,
        current: UserProfile,
        query: str,
        extracted: ExtractedQuery,
        favourited_ids: Optional[list[str]] = None,
        hidden_ids: Optional[list[str]] = None,
    ) -> UserProfile:
        payload = {
            "current_profile": current.model_dump(),
            "latest_query": query,
            "extracted_query": extracted.model_dump(),
            "favourited_ids": favourited_ids or [],
            "hidden_ids": hidden_ids or [],
        }
        resp = self.client.messages.create(
            model=settings.extractor_model,
            max_tokens=1500,
            system=REWRITER_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()

        try:
            data = json.loads(raw)
            updated = UserProfile.model_validate(data)
        except Exception:
            log.exception("Rewriter output invalid, keeping current profile. Raw: %s", raw[:400])
            return current

        # Merge favourites/hidden deterministically (don't trust the LLM here)
        fav = set(current.favourited_listing_ids or [])
        if favourited_ids:
            fav.update(favourited_ids)
        hid = set(current.hidden_listing_ids or [])
        if hidden_ids:
            hid.update(hidden_ids)
        updated.favourited_listing_ids = sorted(fav)
        updated.hidden_listing_ids = sorted(hid)
        updated.user_id = current.user_id

        return updated
