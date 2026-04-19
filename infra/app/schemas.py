"""Structured schemas for query understanding and user profiles.

Two key schemas:
  - ExtractedQuery: what we pull out of a natural-language query
  - UserProfile:    durable user-level preferences we merge in before search

Keep these strict: the LLM must produce JSON that validates against these.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- Query understanding ----------

class CommuteRequirement(BaseModel):
    """External computation: commute time to a named destination."""
    destination: str = Field(..., description="Landmark or address, e.g. 'ETH Zurich', 'Zurich HB'")
    max_minutes: int = Field(..., ge=1, le=180)
    mode: Literal["public_transport", "walking", "cycling", "driving"] = "public_transport"


class HardFilters(BaseModel):
    """Strict constraints. Violating these disqualifies a listing."""
    cities: list[str] = Field(default_factory=list, description="Any of these (OR)")
    districts: list[str] = Field(default_factory=list, description="e.g. 'Kreis 4', 'Oerlikon'")
    cantons: list[str] = Field(default_factory=list)
    price_chf_max: Optional[int] = None
    price_chf_min: Optional[int] = None
    rooms_min: Optional[float] = None
    rooms_max: Optional[float] = None
    area_sqm_min: Optional[int] = None
    area_sqm_max: Optional[int] = None
    # Features the user explicitly said "must have" vs "nice to have".
    # Don't guess — only fill this when the user is emphatic.
    required_features: list[str] = Field(default_factory=list)
    property_type: Optional[str] = None  # apartment, studio, house
    furnished: Optional[bool] = None
    available_from: Optional[str] = None  # ISO 8601


class SoftStructured(BaseModel):
    """Preferences that map directly to schema booleans/flags."""
    preferred_features: list[str] = Field(default_factory=list)
    avoid_features: list[str] = Field(default_factory=list)


class SoftSemantic(BaseModel):
    """Weights 0-1 for VLM-score-mapped concepts, plus free text for embeddings.

    The VLM score fields we can directly boost:
      brightness, modernity, condition, spaciousness, kitchen_appeal, bathroom_appeal
    """
    brightness: float = Field(0.0, ge=0.0, le=1.0)
    modernity: float = Field(0.0, ge=0.0, le=1.0)
    condition: float = Field(0.0, ge=0.0, le=1.0)
    spaciousness: float = Field(0.0, ge=0.0, le=1.0)
    kitchen_appeal: float = Field(0.0, ge=0.0, le=1.0)
    bathroom_appeal: float = Field(0.0, ge=0.0, le=1.0)
    # Non-visual "vibes" — go to the embedding API
    quietness: float = Field(0.0, ge=0.0, le=1.0)
    safety: float = Field(0.0, ge=0.0, le=1.0)
    family_friendly: float = Field(0.0, ge=0.0, le=1.0)
    near_lake_or_green: float = Field(0.0, ge=0.0, le=1.0)
    # Everything else that doesn't fit above goes here verbatim for semantic search.
    free_text: str = ""


class ExtractedQuery(BaseModel):
    """The full output of the NLU pass."""
    hard_filters: HardFilters = Field(default_factory=HardFilters)
    soft_structured: SoftStructured = Field(default_factory=SoftStructured)
    soft_semantic: SoftSemantic = Field(default_factory=SoftSemantic)
    commute: Optional[CommuteRequirement] = None
    # Ordered list of constraint keys the system may relax if the candidate
    # pool is too small. Lowest-impact first.
    relaxation_priority: list[str] = Field(default_factory=list)
    # Questions to ask the user if the query is severely underspecified.
    # Leave empty for confident extractions.
    clarifications_needed: list[str] = Field(default_factory=list)
    # Which profile fields (if any) we substituted defaults from.
    # The orchestrator fills this in; the LLM doesn't.
    profile_fields_used: list[str] = Field(default_factory=list)


# ---------- User profile ----------

class UserProfile(BaseModel):
    """Durable preferences carried across sessions.

    All fields optional — start empty and grow through interaction.
    """
    user_id: str
    # Default search context
    home_cities: list[str] = Field(default_factory=list)
    work_address: Optional[str] = None
    typical_budget_max_chf: Optional[int] = None
    typical_rooms_min: Optional[float] = None
    household_size: Optional[int] = None
    has_children: Optional[bool] = None

    # Recurring soft preferences (decayed average over sessions)
    preferred_features: list[str] = Field(default_factory=list)
    semantic_preferences: dict[str, float] = Field(default_factory=dict)
    # e.g. {"brightness": 0.7, "quietness": 0.8}

    # Behaviour signals (bonus task territory)
    favourited_listing_ids: list[str] = Field(default_factory=list)
    hidden_listing_ids: list[str] = Field(default_factory=list)
    # Listings already shown — kept as a rolling window, oldest dropped first
    seen_listing_ids: list[str] = Field(default_factory=list)
    recent_query_summaries: list[str] = Field(default_factory=list, max_length=20)

    # Freshness
    last_updated: Optional[str] = None
