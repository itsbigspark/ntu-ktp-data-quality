"""
Authentication gate for the DQ Investigator.

Simple password-based login. Not full enterprise auth - just a gate
to prevent accidental access. For real enterprise deployment, swap
this for SSO/OAuth.

Credentials are loaded from the DQ_USERS environment variable:

    DQ_USERS=admin:s3cur3pass,analyst:an4lyst99,demo:d3m0pass

Each entry is username:password separated by commas. Passwords are
hashed with SHA-256 at startup — plain text is never stored in memory.

If DQ_USERS is not set the app refuses to start in production
(when DQ_ENV=production). In development it falls back to safe
defaults so local runs still work without any config.
"""

import hashlib
import os
import streamlit as st


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict[str, str]:
    """
    Build the username → hashed-password map from DQ_USERS env var.
    Falls back to dev defaults only when DQ_ENV != 'production'.
    """
    raw = os.environ.get("DQ_USERS", "").strip()
    if raw:
        users = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            username, _, password = entry.partition(":")
            username = username.strip()
            password = password.strip()
            if username and password:
                users[username] = _hash(password)
        if users:
            return users

    # In production, refuse to run with no credentials configured
    if os.environ.get("DQ_ENV", "").lower() == "production":
        raise RuntimeError(
            "DQ_USERS environment variable is not set. "
            "Set DQ_USERS=user1:pass1,user2:pass2 in your ECS task definition "
            "or AWS Secrets Manager before deploying."
        )

    # Dev fallback — only reachable locally when DQ_ENV is not 'production'
    return {
        "admin":   _hash("admin123"),
        "analyst": _hash("dq2026"),
        "demo":    _hash("demo"),
    }


_USERS = _load_users()


def check_auth() -> bool:
    """
    Display login form if not authenticated.
    Returns True if the user is authenticated, False otherwise.
    Call this at the top of every page.
    """
    if st.session_state.get("authenticated"):
        return True
    return False


def login_page():
    """Render the login page. Call this from Home.py when not authenticated."""
    from shared.theme import logo_strip_html
    st.markdown(
        "<div style='text-align:center;padding-top:48px;'>"
        "<h1 style='font-family:Orbitron,sans-serif;color:#00ff41;"
        "text-shadow:0 0 20px #00ff4180;letter-spacing:3px;'>"
        "AI POWERED DQ INVESTIGATOR</h1>"
        "<p style='color:#4a7a4f;font-family:monospace;letter-spacing:4px;"
        "font-size:0.85rem;'>ENTERPRISE DATA QUALITY PLATFORM</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(logo_strip_html(height=40), unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#4a7a4f;font-family:Share Tech Mono,monospace;"
        "font-size:0.65rem;letter-spacing:2px;margin-top:8px;'>"
        "BIGSPARK · NOTTINGHAM TRENT UNIVERSITY · INNOVATE UK KTP</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            st.markdown(
                "<p style='color:#00e5ff;font-family:monospace;font-size:0.8rem;"
                "letter-spacing:2px;text-align:center;'>AUTHENTICATE</p>",
                unsafe_allow_html=True,
            )
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("ACCESS SYSTEM", use_container_width=True)

            if submitted:
                if username in _USERS and _hash(password) == _USERS[username]:
                    st.session_state["authenticated"] = True
                    st.session_state["auth_user"] = username
                    st.rerun()
                else:
                    st.error("ACCESS DENIED - Invalid credentials")

        st.markdown(
            "<p style='color:#4a7a4f;font-family:monospace;font-size:0.65rem;"
            "text-align:center;margin-top:20px;letter-spacing:1px;'>"
            "Contact your administrator for access credentials.</p>",
            unsafe_allow_html=True,
        )


def logout_button():
    """Render a logout button in the sidebar."""
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state["authenticated"] = False
        st.session_state["auth_user"] = None
        st.rerun()


def require_auth():
    """
    Gate function. Call at top of every page.
    If not authenticated, shows a message and stops execution.
    """
    if not check_auth():
        st.warning("Please log in from the Home page to access this tool.")
        st.stop()
