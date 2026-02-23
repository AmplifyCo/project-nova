"""LinkedIn OAuth 2.0 Setup — run once locally on your Mac.

Usage:
    python scripts/linkedin_auth.py

What it does:
    1. Prints the LinkedIn authorization URL
    2. You open it in your browser and approve
    3. LinkedIn redirects to the webhook URL with a code in the URL
    4. You paste that full redirect URL here
    5. Script exchanges the code, fetches your person URN, prints .env lines
    6. You add those lines to EC2 .env and restart Nova

Requirements:
    - LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET in .env (or enter below)
    - https://webhook.amplify-pixels.com/linkedin/callback registered as
      a redirect URI in your LinkedIn Developer App
      (Auth tab → OAuth 2.0 settings → Redirect URLs)

Run every ~60 days to refresh the token.
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

# ── Credentials ───────────────────────────────────────────────────────────────
CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID", "").strip() or input("LinkedIn Client ID: ").strip()
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET", "").strip() or input("LinkedIn Client Secret: ").strip()

BASE_URL = os.getenv("NOVA_BASE_URL", "https://webhook.amplify-pixels.com").rstrip("/")
REDIRECT_URI = f"{BASE_URL}/linkedin/callback"

# Set LINKEDIN_WITH_PROFILE=1 in .env after adding "Sign In with LinkedIn"
# product to your app. This fetches your person URN automatically.
_want_profile = os.getenv("LINKEDIN_WITH_PROFILE", "").strip() in ("1", "true", "yes")
SCOPE = "openid profile w_member_social" if _want_profile else "w_member_social"

# ── Step 1: Print auth URL ────────────────────────────────────────────────────
params = urllib.parse.urlencode({
    "response_type": "code",
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "state": "nova_linkedin_setup",
    "scope": SCOPE,
})
auth_url = f"https://www.linkedin.com/oauth/v2/authorization?{params}"

print()
print("=" * 65)
print("Step 1: Open this URL in your browser and approve access:")
print("=" * 65)
print(f"\n  {auth_url}\n")
print("After approving, your browser will redirect to:")
print(f"  {REDIRECT_URI}?code=...")
print("(The page may show an error or timeout — that's fine.)")
print()

# ── Step 2: User pastes the redirect URL ─────────────────────────────────────
print("=" * 65)
print("Step 2: Paste the full redirect URL from your browser address bar:")
print("=" * 65)
redirect_url = input("\nPaste URL here: ").strip()

parsed = urllib.parse.urlparse(redirect_url)
code = urllib.parse.parse_qs(parsed.query).get("code", [""])[0]

if not code:
    print(f"\nERROR: No 'code' found in URL: {redirect_url}")
    raise SystemExit(1)

print(f"\n  Code extracted: {code[:20]}...")

# ── Step 3: Exchange code for token ──────────────────────────────────────────
print("\nExchanging code for access token...")
token_data = urllib.parse.urlencode({
    "grant_type": "authorization_code",
    "code": code,
    "redirect_uri": REDIRECT_URI,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
}).encode()

req = urllib.request.Request(
    "https://www.linkedin.com/oauth/v2/accessToken",
    data=token_data,
    method="POST",
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)
with urllib.request.urlopen(req, timeout=15) as resp:
    token_resp = json.loads(resp.read())

access_token = token_resp.get("access_token", "")
if not access_token:
    print(f"\nERROR: Token exchange failed:\n{json.dumps(token_resp, indent=2)}")
    raise SystemExit(1)

expires_days = token_resp.get("expires_in", 0) // 86400
print(f"  Token received (expires in {expires_days} days)")

# ── Step 4: Fetch person URN ──────────────────────────────────────────────────
# w_member_social alone does not allow reading profile — try anyway
print("Fetching LinkedIn person ID...")
person_id = ""
try:
    req = urllib.request.Request(
        "https://api.linkedin.com/v2/me?projection=(id)",
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-Restli-Protocol-Version": "2.0.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        me_data = json.loads(resp.read())
    person_id = me_data.get("id", "")
    if person_id:
        print(f"  Person ID: {person_id}")
except Exception:
    pass  # Expected — handled below

# ── Step 5: Print results ─────────────────────────────────────────────────────
print()
print("=" * 65)
print("✅ Access token obtained:")
print("=" * 65)
print(f"\nLINKEDIN_CLIENT_ID={CLIENT_ID}")
print(f"LINKEDIN_CLIENT_SECRET={CLIENT_SECRET}")
print(f"LINKEDIN_ACCESS_TOKEN={access_token}")

if person_id:
    print(f"LINKEDIN_PERSON_URN=urn:li:person:{person_id}")
    print()
    print("Add all 4 lines to EC2 .env, then:")
    print("  sudo systemctl restart digital-twin")
else:
    print()
    print("-" * 65)
    print("⚠️  Could not auto-fetch person URN (w_member_social can't read profile)")
    print()
    print("ONE-TIME step to get your permanent LinkedIn Person URN:")
    print()
    print("  1. Go to: https://www.linkedin.com/developers/apps")
    print("     → Your app → Products tab")
    print('     → Add "Sign In with LinkedIn using OpenID Connect"')
    print()
    print("  2. Re-run this script — it will fetch your URN automatically")
    print("     using scope: openid profile w_member_social")
    print()
    print("  3. After getting your URN, you can remove that product from")
    print("     your app. The URN is permanent and never changes.")
    print()
    print("  Your URN looks like: urn:li:person:AbCdEfGhIj")
    print("  Add it to EC2 .env as: LINKEDIN_PERSON_URN=urn:li:person:...")
    print("-" * 65)

print()
print(f"Token expires in {expires_days} days — re-run this script to refresh.")
print("=" * 65)

# If openid scope now available, use it automatically
if not person_id and "openid" in os.getenv("LINKEDIN_SCOPES_AVAILABLE", ""):
    pass  # placeholder for future auto-upgrade
