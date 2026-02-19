# Layered Memory Access Control

## Problem

Nova can be reached by anyone who messages the owner's WhatsApp. When Nova messages Pallavi on behalf of Srinath, Pallavi's reply comes back through the same Twilio webhook — and Nova currently treats her as "the user." She could ask for bank details, contacts, or private information and Nova would answer.

## Design: Concentric Trust Rings

Memory is organized in layers. Each contact has an **access tier**. Each memory/fact has a **minimum tier** required to access it. Nova only reveals information the caller's tier permits.

```
┌──────────────────────────────────────────────────┐
│  TIER 5 — Public (anyone)                        │
│  Name, general interests, public preferences     │
│  ┌──────────────────────────────────────────┐    │
│  │  TIER 4 — Acquaintance (colleagues, etc) │    │
│  │  Work schedule, general availability      │    │
│  │  ┌──────────────────────────────────┐     │    │
│  │  │  TIER 3 — Close Friends          │     │    │
│  │  │  Personal preferences, opinions  │     │    │
│  │  │  ┌──────────────────────────┐    │     │    │
│  │  │  │  TIER 2 — Family         │    │     │    │
│  │  │  │  Health, schedules,      │    │     │    │
│  │  │  │  family matters          │    │     │    │
│  │  │  │  ┌──────────────────┐    │    │     │    │
│  │  │  │  │  TIER 1 — Owner  │    │    │     │    │
│  │  │  │  │  Full access:    │    │    │     │    │
│  │  │  │  │  bank, passwords │    │    │     │    │
│  │  │  │  │  all contacts    │    │    │     │    │
│  │  │  │  └──────────────────┘    │    │     │    │
│  │  │  └──────────────────────────┘    │     │    │
│  │  └──────────────────────────────────┘     │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

## Tier Definitions

| Tier | Label | Who | Can Access |
|------|-------|-----|------------|
| 1 | **Owner** | Srinath (OWNER_PHONE in .env) | Everything — bank, passwords, all contacts, all memory |
| 2 | **Family** | Wife, parents, siblings | Health, schedules, family events, personal preferences |
| 3 | **Close Friends** | Best friends | Opinions, preferences, casual personal info |
| 4 | **Acquaintance** | Colleagues, neighbors | Work schedule, general availability, public interests |
| 5 | **Public** | Anyone else / unknown numbers | Name, basic greetings only |

## Data Model Changes

### Contact Schema Addition

```python
# In contacts tool / ChromaDB metadata
{
    "name": "Pallavi",
    "phone": "+1937123456",
    "relationship": "wife",
    "access_tier": 2,          # ← NEW: auto-derived from relationship
}
```

### Relationship → Tier Auto-Mapping

```python
RELATIONSHIP_TIER_MAP = {
    # Tier 1: Owner (set via OWNER_PHONE env var, not relationship)
    # Tier 2: Family
    "wife": 2, "husband": 2, "spouse": 2,
    "mom": 2, "dad": 2, "mother": 2, "father": 2,
    "sister": 2, "brother": 2, "son": 2, "daughter": 2,
    # Tier 3: Close friends
    "best friend": 3, "close friend": 3, "friend": 3,
    # Tier 4: Acquaintance
    "colleague": 4, "coworker": 4, "neighbor": 4, "boss": 4,
    "doctor": 4, "accountant": 4,
    # Tier 5: Everyone else
    "default": 5,
}
```

### Memory Tagging

```python
# Each stored fact gets a min_access_tier
{
    "category": "personal_info",
    "fact": "Birthday is March 5th",
    "min_access_tier": 3,       # Friends and closer can know
}
{
    "category": "financial",
    "fact": "Bank account at Chase",
    "min_access_tier": 1,       # Owner only
}
```

### Category → Default Tier

```python
CATEGORY_TIER_MAP = {
    "financial": 1,        # Owner only
    "credential": 1,       # Owner only
    "health": 2,           # Family+
    "personal_info": 3,    # Friends+
    "preference": 3,       # Friends+
    "relationship": 2,     # Family+
    "nickname": 4,         # Acquaintance+
    "opinion": 3,          # Friends+
    "habit": 3,            # Friends+
    "schedule": 4,         # Acquaintance+
}
```

## Runtime Flow

```
Incoming message (WhatsApp/Telegram)
        │
        ▼
┌─ Identify Sender ──────────────┐
│  sender_phone → contacts DB    │
│  → resolve access_tier         │
│  Unknown phone → tier 5        │
│  OWNER_PHONE → tier 1          │
└────────────────────────────────┘
        │
        ▼
┌─ Build Context ────────────────┐
│  Query Brain for relevant      │
│  memory, but FILTER by:        │
│  memory.min_access_tier >=     │
│  sender.access_tier            │
│  (lower tier = more access)    │
└────────────────────────────────┘
        │
        ▼
┌─ Generate Response ────────────┐
│  System prompt includes:       │
│  "Caller is {name} (tier {n}). │
│  Do NOT reveal tier {n-1} or   │
│  deeper information."          │
└────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Owner Identification (Quick Win)
- [ ] Add `OWNER_PHONE` to `.env`
- [ ] In `_process_message_locked`, compare `user_id` against `OWNER_PHONE`
- [ ] Set `_current_access_tier = 1` if owner, else `5` (binary for now)
- [ ] Pass tier to system prompt: "This user has access tier {n}"

### Phase 2: Contact-Based Tier Resolution
- [ ] Add `access_tier` field to contacts schema
- [ ] Add `RELATIONSHIP_TIER_MAP` for auto-assignment
- [ ] On incoming message: lookup `sender_phone` → contact → `access_tier`
- [ ] Unknown numbers default to tier 5

### Phase 3: Memory-Level Access Control
- [ ] Add `min_access_tier` to all stored facts/preferences
- [ ] Apply `CATEGORY_TIER_MAP` defaults when storing
- [ ] Filter Brain query results by caller's tier before building context
- [ ] Audit existing stored data and backfill tiers

### Phase 4: Conversation Isolation
- [x] Per-user conversation buffers (already done)
- [x] Daily JSONL log with user_id (already done)
- [ ] Per-user daily log files (`data/conversations/{user_id}/YYYY-MM-DD.jsonl`)

## Security Considerations

- **Tier escalation**: Nova must never accept "I'm the owner" — verify by phone number only
- **LLM leakage**: Even with context filtering, add system prompt rules per tier
- **Conversation bleed**: Already fixed with per-user buffers
- **Override by owner**: Owner (tier 1) can explicitly grant temporary access: "Share my schedule with Pallavi"
