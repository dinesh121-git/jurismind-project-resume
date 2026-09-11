"""
Lightweight token-based authentication for JurisMind.

This is intentionally simple (in-memory token store, not JWT) — appropriate for
a single-instance student/demo deployment. For a production system, swap this for
JWTs or Flask-Login with a proper session store; the seam is isolated here so
that swap only touches this one file.
"""
import secrets
from functools import wraps
from flask import request, jsonify

# token -> user_id. Lives only as long as the process does — logging in again
# after a backend restart is expected behavior for this simple version.
ACTIVE_TOKENS = {}


def issue_token(user_id):
    token = secrets.token_hex(32)
    ACTIVE_TOKENS[token] = user_id
    return token


def revoke_token(token):
    ACTIVE_TOKENS.pop(token, None)


def get_user_id_from_token(token):
    return ACTIVE_TOKENS.get(token)


def require_auth(f):
    """Decorator: rejects the request unless a valid 'Authorization: Bearer <token>' header is present."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ", 1)[1]
        user_id = get_user_id_from_token(token)
        if user_id is None:
            return jsonify({"error": "Invalid or expired session. Please log in again."}), 401

        request.user_id = user_id
        return f(*args, **kwargs)
    return wrapper