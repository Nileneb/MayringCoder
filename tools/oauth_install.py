#!/usr/bin/env python3
"""PKCE OAuth flow für MayringCoder Hook-Token-Setup.

Browser öffnet → User loggt sich auf app.linn.games ein → Token wird
automatisch gespeichert. Kein manuelles Copy-Paste nötig.

Usage:
    python3 tools/oauth_install.py
    python3 tools/oauth_install.py --jwt-file /custom/path/hook.jwt
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser

_MCP_URL = os.environ.get("MCP_OAUTH_BASE_URL", "https://mcp.linn.games").rstrip("/")
_DEFAULT_JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
_CLIENT_ID = "mayring-hook-cli"
_REDIRECT_PORT = 9876
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}/callback"
_TIMEOUT_SECONDS = 120


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _save_token(token: str, jwt_file: str) -> None:
    os.makedirs(os.path.dirname(jwt_file), exist_ok=True)
    with open(jwt_file, "w") as f:
        f.write(token)
    os.chmod(jwt_file, 0o600)


def _exchange_code(code: str, verifier: str) -> str:
    payload = urllib.parse.urlencode({
        "grant_type":   "authorization_code",
        "code":         code,
        "redirect_uri": _REDIRECT_URI,
        "client_id":    _CLIENT_ID,
        "code_verifier": verifier,
    }).encode()
    req = urllib.request.Request(
        f"{_MCP_URL}/token",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())
    token = data.get("access_token", "")
    if not token:
        raise ValueError(f"Kein access_token in Server-Antwort: {data}")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="MayringCoder OAuth-Setup")
    parser.add_argument("--jwt-file", default=_DEFAULT_JWT_FILE)
    args = parser.parse_args()

    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    callback_result: dict = {}
    server_done = threading.Event()

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if not self.path.startswith("/callback"):
                self.send_response(404)
                self.end_headers()
                return
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            callback_result["code"]  = parsed.get("code",  [None])[0]
            callback_result["state"] = parsed.get("state", [None])[0]
            callback_result["error"] = parsed.get("error", [None])[0]
            ok = bool(callback_result.get("code"))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if ok:
                self.wfile.write(
                    b"<h2>\xe2\x9c\x85 Login erfolgreich!</h2>"
                    b"<p>Du kannst dieses Fenster schlie\xc3\x9fen.</p>"
                )
            else:
                err = str(callback_result.get("error", "unbekannt")).encode()
                self.wfile.write(b"<h2>\xe2\x9d\x8c Fehler beim Login</h2><p>" + err + b"</p>")
            server_done.set()

        def log_message(self, *_):
            pass

    try:
        httpd = http.server.HTTPServer(("localhost", _REDIRECT_PORT), CallbackHandler)
    except OSError:
        print(f"Port {_REDIRECT_PORT} belegt — beende laufenden Prozess und versuche es erneut.")
        sys.exit(1)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    auth_url = f"{_MCP_URL}/authorize?" + urllib.parse.urlencode({
        "response_type":         "code",
        "client_id":             _CLIENT_ID,
        "redirect_uri":          _REDIRECT_URI,
        "scope":                 "mcp:memory",
        "state":                 state,
        "code_challenge":        challenge,
        "code_challenge_method": "S256",
    })

    print(f"Login-URL (falls Browser nicht öffnet): {auth_url}", file=sys.stderr)
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass
    print(f"Warte auf Login (max {_TIMEOUT_SECONDS}s) …")

    if not server_done.wait(timeout=_TIMEOUT_SECONDS):
        httpd.shutdown()
        print("Timeout — kein Login empfangen. Starte das Script erneut.")
        sys.exit(1)

    httpd.shutdown()

    if callback_result.get("error"):
        print(f"OAuth-Fehler: {callback_result['error']}")
        sys.exit(1)
    if not callback_result.get("code"):
        print("Kein Authorization-Code empfangen.")
        sys.exit(1)
    if callback_result.get("state") != state:
        print("Sicherheitsfehler: State-Parameter stimmt nicht überein.")
        sys.exit(1)

    try:
        token = _exchange_code(callback_result["code"], verifier)
    except Exception as e:
        print(f"Token-Austausch fehlgeschlagen: {e}")
        sys.exit(1)

    _save_token(token, args.jwt_file)
    print(f"Token gespeichert: {args.jwt_file}")
    print("Setup abgeschlossen.")


if __name__ == "__main__":
    main()
