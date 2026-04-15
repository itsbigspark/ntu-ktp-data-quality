"""
Authentication gate for the DQ Investigator.

Simple password-based login. Not full enterprise auth - just a gate
to prevent accidental access. For real enterprise deployment, swap
this for SSO/OAuth.
"""

import hashlib
import streamlit as st

# Default users. Override via environment or config file.
_USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "analyst": hashlib.sha256("dq2026".encode()).hexdigest(),
    "demo": hashlib.sha256("demo".encode()).hexdigest(),
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


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
    st.markdown(
        "<div style='text-align:center;padding-top:60px;'>"
        "<h1 style='font-family:Orbitron,sans-serif;color:#00ff41;"
        "text-shadow:0 0 20px #00ff4180;letter-spacing:3px;'>"
        "AI POWERED DQ INVESTIGATOR</h1>"
        "<p style='color:#4a7a4f;font-family:monospace;letter-spacing:4px;"
        "font-size:0.85rem;'>ENTERPRISE DATA QUALITY PLATFORM</p>"
        "</div>",
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
            "Default: admin / admin123</p>",
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
