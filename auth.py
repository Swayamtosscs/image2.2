"""
auth.py — JWT Authentication module for AI Fashion Ad Studio.

Environment variables required:
  JWT_SECRET   : A long random secret string used to sign tokens.
  AUTH_USERS   : JSON string mapping username → password.
                 Example: '{"admin":"mypassword","designer":"pass2"}'
"""

import os
import json
import time
import hashlib
import streamlit as st

try:
    import jwt  # PyJWT
except ImportError:
    raise ImportError("PyJWT is required. Add 'PyJWT' to requirements.txt and reinstall.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-secret-in-production")
_JWT_ALGORITHM = "HS256"
_TOKEN_EXPIRY_HOURS = 8

# Load user credentials from environment variable.
# Format: JSON string → {"username": "password", ...}
def _load_users() -> dict:
    raw = os.environ.get("AUTH_USERS", '{"admin":"admin123"}')
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        st.error("Server Error: AUTH_USERS environment variable is not valid JSON.")
        return {}


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def create_token(username: str) -> str:
    """Create a signed JWT for the given username, valid for _TOKEN_EXPIRY_HOURS."""
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + _TOKEN_EXPIRY_HOURS * 3600,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def verify_token(token: str) -> str | None:
    """
    Verify a JWT.
    Returns the username (str) on success, None on failure/expiry.
    """
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ---------------------------------------------------------------------------
# Streamlit login form
# ---------------------------------------------------------------------------

def _render_login_form():
    """
    Render a centered, styled login page.
    Sets st.session_state["auth_token"] on success and reruns.
    """
    # --- Inject CSS ---
    st.markdown(
        """
        <style>
        /* Hide Streamlit header/footer/sidebar on the login page */
        #MainMenu, header, footer { visibility: hidden; }
        [data-testid="stSidebar"] { display: none; }

        /* Full-page gradient background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
        }

        /* Card container */
        .login-card {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 48px 40px 40px;
            max-width: 420px;
            margin: 0 auto;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5);
        }

        .login-logo {
            text-align: center;
            margin-bottom: 8px;
        }

        .login-title {
            color: #ffffff;
            font-size: 1.7rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 4px;
            letter-spacing: -0.3px;
        }

        .login-subtitle {
            color: rgba(255,255,255,0.45);
            font-size: 0.875rem;
            text-align: center;
            margin-bottom: 32px;
        }

        /* Streamlit input overrides — typed text is bright white */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.14) !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            border-radius: 10px !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            caret-color: #a78bfa !important;
            padding: 12px 16px !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #a78bfa !important;
            box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.2) !important;
            outline: none !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.35) !important;
            -webkit-text-fill-color: rgba(255, 255, 255, 0.35) !important;
        }
        .stTextInput label {
            color: rgba(255, 255, 255, 0.7) !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
        }

        /* Sign-in button */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 14px !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            width: 100% !important;
            letter-spacing: 0.5px !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.45) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 28px rgba(102, 126, 234, 0.6) !important;
        }
        .stButton > button:active {
            transform: translateY(0) !important;
        }

        /* Error alert */
        .stAlert {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Layout ---
    # Three columns: left spacer | form | right spacer
    _, center, _ = st.columns([1, 1.4, 1])

    with center:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        # Logo / icon
        st.markdown(
            '<div class="login-logo"><span style="font-size:2.8rem;">🎨</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="login-title">AI Fashion Ad Studio</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="login-subtitle">Sign in to access the studio</div>',
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password"
            )
            submitted = st.form_submit_button("Sign In →", use_container_width=True)

        if submitted:
            users = _load_users()
            if username in users and users[username] == password:
                token = create_token(username)
                st.session_state["auth_token"] = token
                st.session_state["auth_user"] = username
                st.rerun()
            else:
                st.error("⚠️ Invalid username or password. Please try again.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown(
            '<p style="text-align:center;color:rgba(255,255,255,0.2);'
            'font-size:0.75rem;margin-top:24px;">'
            "🔒 Authorized access only</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Public API — call this at the top of every Streamlit page
# ---------------------------------------------------------------------------

def require_auth() -> str:
    """
    Call at the top of every Streamlit page (after st.set_page_config).

    - If the user has a valid JWT in session_state → returns the username.
    - Otherwise → renders the login form and calls st.stop() to block the page.

    Usage:
        username = require_auth()
    """
    token = st.session_state.get("auth_token")

    if token:
        username = verify_token(token)
        if username:
            return username
        else:
            # Token expired — clear it and fall through to login
            st.session_state.pop("auth_token", None)
            st.session_state.pop("auth_user", None)
            st.warning("⏰ Your session has expired. Please sign in again.")

    _render_login_form()
    st.stop()


def render_logout_button():
    """
    Renders a logout button in the sidebar.
    Call this anywhere on authenticated pages (e.g. inside a sidebar block).
    """
    user = st.session_state.get("auth_user", "User")
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"<p style='font-size:0.8rem;color:rgba(255,255,255,0.5);margin-bottom:4px;'>"
            f"Signed in as</p>"
            f"<p style='font-size:0.9rem;font-weight:700;color:#a78bfa;margin-top:0'>"
            f"👤 {user}</p>",
            unsafe_allow_html=True,
        )
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            st.session_state.pop("auth_token", None)
            st.session_state.pop("auth_user", None)
            st.rerun()
