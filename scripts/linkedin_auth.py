"""LinkedIn OAuth 2.0 Setup — run once to get your access token.

Usage:
    python scripts/linkedin_auth.py

What it does:
    1. Opens browser → LinkedIn authorization page
    2. You log in and approve Nova's access
    3. Exchanges the code for an access token
    4. Fetches your Person URN
    5. Prints the two lines to add to your .env

Prerequisites:
    - LinkedIn Developer App with "Share on LinkedIn" product enabled
    - LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET in .env (or entered below)
    - Redirect URI http://localhost:8765/callback added in the LinkedIn Developer Portal

Refresh:
    LinkedIn access tokens expire after 60 days. Re-run this script to refresh.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Try to load .env early so env vars are available ──────────────────────────
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

# ── Configuration ─────────────────────────────────────────────────────────────
CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID") or input("LinkedIn Client ID: ").strip()
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET") or input("LinkedIn Client Secret: ").strip()
REDIRECT_URI = "http://localhost:8765/callback"
SCOPE = "openid profile w_member_social"

_auth_code: str = ""


class _CallbackHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler to catch the OAuth redirect."""

    def do_GET(self):
        global _auth_code
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            _auth_code = params["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"<h2>Authorization successful!</h2>"
                b"<p>You can close this tab and return to your terminal.</p>"
            )
        else:
            error = params.get("error_description", ["Unknown error"])[0]
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"<h2>Authorization failed: {error}</h2>".encode())

    def log_message(self, *args):
        pass  # silence HTTP log noise


def _exchange_code(code: str) -> str:
    """Exchange authorization code for access token. Returns the token."""
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
        data = json.loads(resp.read())

    if "access_token" not in data:
        print(f"\nERROR: Token exchange failed:\n{json.dumps(data, indent=2)}")
        sys.exit(1)

    expires_in = data.get("expires_in", 0)
    expires_days = expires_in // 86400
    print(f"  Token expires in {expires_days} days")
    return data["access_token"]


def _get_person_urn(access_token: str) -> tuple[str, str]:
    """Return (person_urn, display_name) using LinkedIn's OpenID userinfo endpoint."""
    req = urllib.request.Request(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    person_id = data.get("sub", "")
    if not person_id:
        print(f"\nERROR: Could not retrieve person ID:\n{json.dumps(data, indent=2)}")
        sys.exit(1)

    person_urn = f"urn:li:person:{person_id}"
    display_name = data.get("name", "unknown")
    return person_urn, display_name


def main():
    # ── Step 1: Build authorization URL and open browser ─────────────────────
    params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "state": "nova_linkedin_setup",
        "scope": SCOPE,
    })
    auth_url = f"https://www.linkedin.com/oauth/v2/authorization?{params}"

    print("\n" + "="*60)
    print("LinkedIn OAuth Setup")
    print("="*60)
    print(f"\nOpening browser for LinkedIn authorization...")
    print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    # ── Step 2: Local server catches the redirect ─────────────────────────────
    print("Waiting for authorization on localhost:8765...")
    server = HTTPServer(("localhost", 8765), _CallbackHandler)
    server.handle_request()  # blocks until one request received

    if not _auth_code:
        print("\nERROR: No authorization code received. Aborting.")
        sys.exit(1)

    print("  Authorization code received.")

    # ── Step 3: Exchange code for token ───────────────────────────────────────
    print("  Exchanging code for access token...")
    access_token = _exchange_code(_auth_code)

    # ── Step 4: Fetch person URN ──────────────────────────────────────────────
    print("  Fetching LinkedIn profile info...")
    person_urn, display_name = _get_person_urn(access_token)

    # ── Step 5: Print results ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅  Authorized as: {display_name}")
    print(f"{'='*60}")
    print("\nAdd these lines to your .env file:\n")
    print(f"LINKEDIN_CLIENT_ID={CLIENT_ID}")
    print(f"LINKEDIN_CLIENT_SECRET={CLIENT_SECRET}")
    print(f"LINKEDIN_ACCESS_TOKEN={access_token}")
    print(f"LINKEDIN_PERSON_URN={person_urn}")
    print()
    print("Token expires in ~60 days. Re-run this script to refresh.")
    print("="*60)


if __name__ == "__main__":
    main()
