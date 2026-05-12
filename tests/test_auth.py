"""Tests for app/shared/auth.py — credential loading and validation."""

import hashlib
import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _reload_auth(env: dict):
    """Reimport auth module with a clean environment."""
    # Provide a minimal streamlit stub so the module can be imported outside Streamlit
    st_stub = types.ModuleType("streamlit")
    st_stub.session_state = {}
    st_stub.markdown = lambda *a, **kw: None
    st_stub.columns = lambda *a, **kw: [MagicMock(), MagicMock(), MagicMock()]
    st_stub.form = MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: False))
    st_stub.text_input = MagicMock(return_value="")
    st_stub.form_submit_button = MagicMock(return_value=False)
    st_stub.error = MagicMock()
    st_stub.sidebar = MagicMock()
    st_stub.warning = MagicMock()
    st_stub.stop = MagicMock()
    st_stub.rerun = MagicMock()

    with patch.dict(sys.modules, {"streamlit": st_stub}):
        with patch.dict(os.environ, env, clear=False):
            # Remove cached module so _load_users() re-runs
            sys.modules.pop("app.shared.auth", None)
            sys.modules.pop("app.shared", None)

            # Ensure app.shared package exists in sys.modules
            pkg = sys.modules.get("app") or types.ModuleType("app")
            sys.modules.setdefault("app", pkg)
            shared_pkg = types.ModuleType("app.shared")
            sys.modules["app.shared"] = shared_pkg

            spec = importlib.util.spec_from_file_location(
                "app.shared.auth",
                os.path.join(os.path.dirname(__file__), "..", "app", "shared", "auth.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["app.shared.auth"] = mod
            spec.loader.exec_module(mod)
            return mod


class TestLoadUsers(unittest.TestCase):

    def _clean_env(self):
        """Return environment with DQ_USERS and DQ_ENV removed."""
        return {k: v for k, v in os.environ.items() if k not in ("DQ_USERS", "DQ_ENV")}

    def test_single_user_from_env(self):
        env = {**self._clean_env(), "DQ_USERS": "alice:secret123", "DQ_ENV": "development"}
        auth = _reload_auth(env)
        self.assertIn("alice", auth._USERS)
        self.assertEqual(auth._USERS["alice"], _hash("secret123"))

    def test_multiple_users_from_env(self):
        env = {**self._clean_env(), "DQ_USERS": "admin:pass1,analyst:pass2,demo:pass3", "DQ_ENV": "development"}
        auth = _reload_auth(env)
        self.assertEqual(len(auth._USERS), 3)
        self.assertEqual(auth._USERS["analyst"], _hash("pass2"))

    def test_passwords_are_hashed_not_plaintext(self):
        env = {**self._clean_env(), "DQ_USERS": "bob:mypassword", "DQ_ENV": "development"}
        auth = _reload_auth(env)
        self.assertNotEqual(auth._USERS["bob"], "mypassword")
        self.assertEqual(len(auth._USERS["bob"]), 64)  # SHA-256 hex length

    def test_malformed_entry_skipped(self):
        env = {**self._clean_env(), "DQ_USERS": "badentry,good:pass", "DQ_ENV": "development"}
        auth = _reload_auth(env)
        self.assertNotIn("badentry", auth._USERS)
        self.assertIn("good", auth._USERS)

    def test_whitespace_trimmed(self):
        env = {**self._clean_env(), "DQ_USERS": " user1 : pass1 , user2 : pass2 ", "DQ_ENV": "development"}
        auth = _reload_auth(env)
        self.assertIn("user1", auth._USERS)
        self.assertIn("user2", auth._USERS)

    def test_dev_fallback_when_no_dq_users(self):
        env = {k: v for k, v in os.environ.items() if k not in ("DQ_USERS", "DQ_ENV")}
        # DQ_ENV not set → dev mode → fallback credentials
        auth = _reload_auth(env)
        self.assertIn("admin", auth._USERS)
        self.assertIn("analyst", auth._USERS)
        self.assertIn("demo", auth._USERS)

    def test_production_mode_raises_without_dq_users(self):
        env = {k: v for k, v in os.environ.items() if k not in ("DQ_USERS", "DQ_ENV")}
        env["DQ_ENV"] = "production"
        with self.assertRaises(RuntimeError):
            _reload_auth(env)

    def test_production_mode_works_with_dq_users(self):
        env = {k: v for k, v in os.environ.items() if k not in ("DQ_USERS", "DQ_ENV")}
        env["DQ_ENV"] = "production"
        env["DQ_USERS"] = "admin:securepass"
        auth = _reload_auth(env)
        self.assertIn("admin", auth._USERS)

    def test_colon_in_password_parsed_correctly(self):
        # partition(":")  means only the first ":" splits username from password
        env = {**self._clean_env(), "DQ_USERS": "user:pass:with:colons", "DQ_ENV": "development"}
        auth = _reload_auth(env)
        self.assertIn("user", auth._USERS)
        self.assertEqual(auth._USERS["user"], _hash("pass:with:colons"))


if __name__ == "__main__":
    unittest.main()
