"""
app/pages/14_External_Validation.py
=====================================
Live external API validation -- check column values against
authoritative external sources in real time.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import time
import streamlit as st
import pandas as pd

from shared.auth import require_auth
from shared.state import get_df

require_auth()

st.title("External API Validation")
st.caption("Validate column values against live external APIs -- postcodes, emails, VAT, IBAN, company names, and more.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = get_df()
if df is None or df.empty:
    st.info("No data loaded. Go to Load Data and upload a file first.")
    st.stop()

st.success(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# Validator config
# ---------------------------------------------------------------------------
from core.external_validators import VALIDATOR_LABELS

VALIDATOR_DESCRIPTIONS = {
    "postcode": "Checks UK postcodes against postcodes.io — free, no API key. Returns region, district, lat/long.",
    "email": "Checks email format + MX record (does the domain actually receive email?). No API key needed.",
    "vat": "Validates UK VAT numbers against HMRC's live register. Free, no key required.",
    "iban": "Validates IBAN format and checksum (ISO 13616 algorithm). Works offline — no API call needed.",
    "phone": "Validates phone number format and structure. Uses phonenumbers library.",
    "sort_code": "Validates UK bank sort codes (6-digit format XX-XX-XX).",
    "company": "Searches Companies House for a company name. Free API key required.",
    "custom": "Validate against any REST API endpoint you configure.",
}

st.divider()
st.subheader("Configure Validation")

col1, col2 = st.columns(2)

with col1:
    selected_column = st.selectbox(
        "Column to validate",
        options=df.columns.tolist(),
        help="Select the column whose values you want to check against an external API"
    )

with col2:
    validator_type = st.selectbox(
        "Validator",
        options=list(VALIDATOR_LABELS.keys()),
        format_func=lambda x: VALIDATOR_LABELS[x],
    )

# Description
st.info(VALIDATOR_DESCRIPTIONS.get(validator_type, ""))

# Preview values
with st.expander(f"Preview: first 10 values in '{selected_column}'"):
    sample = df[selected_column].dropna().head(10).tolist()
    for v in sample:
        st.code(str(v))

# ---------------------------------------------------------------------------
# Validator-specific config
# ---------------------------------------------------------------------------
api_key = ""
custom_config = {}

if validator_type == "company":
    st.warning("Companies House requires a free API key. Get one at: https://developer.company-information.service.gov.uk/")
    api_key = st.text_input("Companies House API Key", type="password", placeholder="Your API key")

elif validator_type == "custom":
    st.subheader("Custom API Configuration")
    st.caption("Configure your own REST endpoint. The value from the selected column will be sent to this API.")

    custom_config["endpoint_url"] = st.text_input(
        "Endpoint URL",
        placeholder="https://api.example.com/validate/{value}",
        help="Use {value} as a placeholder for the column value in the URL, or leave it and use the param name below"
    )
    c1, c2 = st.columns(2)
    with c1:
        custom_config["method"] = st.selectbox("HTTP Method", ["GET", "POST"])
        custom_config["value_param"] = st.text_input("Value parameter name", value="value",
            help="For GET: query param name. For POST: JSON body key.")
    with c2:
        custom_config["valid_json_path"] = st.text_input(
            "JSON path to validity field", value="valid",
            help="Dot-separated path in response JSON. E.g. 'valid', 'result.is_valid', 'data.status'"
        )
        api_key = st.text_input("API Key (optional)", type="password")

    if custom_config["endpoint_url"]:
        st.caption(f"Example call: `{custom_config['method']} {custom_config['endpoint_url'].replace('{value}', '<column_value>')}`")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
with st.expander("Options"):
    max_rows = st.slider(
        "Max rows to validate",
        min_value=10, max_value=len(df), value=min(200, len(df)),
        help="Limit rows to keep within free API rate limits. Run on full dataset once you're satisfied."
    )
    rate_limit = st.slider("Delay between API calls (ms)", 0, 500, 50,
        help="Increase if you hit rate limit errors")

# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------
st.divider()

if st.button("Run External Validation", type="primary", use_container_width=True):

    if validator_type == "custom" and not custom_config.get("endpoint_url"):
        st.error("Please enter an endpoint URL for custom validation.")
        st.stop()

    sample_df = df.head(max_rows).copy()

    progress = st.progress(0, text="Starting external validation...")
    status = st.empty()
    t0 = time.time()

    try:
        from core.external_validators import validate_column

        status.info(f"Validating {len(sample_df):,} rows in '{selected_column}' via {VALIDATOR_LABELS[validator_type]}...")

        result_df = validate_column(
            df=sample_df,
            column=selected_column,
            validator_type=validator_type,
            api_key=api_key,
            custom_config=custom_config if validator_type == "custom" else None,
            rate_limit_delay=rate_limit / 1000,
        )

        elapsed = round(time.time() - t0, 1)
        progress.progress(100, text="Done")

        # ---------------------------------------------------------------------------
        # Results summary
        # ---------------------------------------------------------------------------
        valid_col = f"{selected_column}_ext_valid"
        error_col = f"{selected_column}_ext_error"
        detail_col = f"{selected_column}_ext_detail"

        total = len(result_df)
        valid = int(result_df[valid_col].sum())
        invalid = int((result_df[valid_col] == False).sum())

        status.empty()
        st.success(f"Validation complete in {elapsed}s")

        m1, m2, m3 = st.columns(3)
        m1.metric("Valid", f"{valid:,}", f"{round(valid/total*100, 1)}%")
        m2.metric("Invalid", f"{invalid:,}", f"-{round(invalid/total*100, 1)}%", delta_color="inverse")
        m3.metric("Rows checked", f"{total:,}")

        # ---------------------------------------------------------------------------
        # Invalid rows detail
        # ---------------------------------------------------------------------------
        if invalid > 0:
            st.subheader(f"Invalid Values ({invalid} rows)")
            invalid_rows = result_df[result_df[valid_col] == False][
                [selected_column, error_col, detail_col]
            ].rename(columns={
                selected_column: "Value",
                error_col: "Error",
                detail_col: "Detail"
            })
            st.dataframe(invalid_rows, use_container_width=True)

            # Top errors
            if error_col in result_df.columns:
                top_errors = result_df[result_df[valid_col] == False][error_col].value_counts().head(5)
                if not top_errors.empty:
                    st.subheader("Top Error Reasons")
                    for err, count in top_errors.items():
                        st.write(f"- **{err}**: {count} rows")

        # ---------------------------------------------------------------------------
        # Valid rows with enriched detail (e.g. normalised postcodes)
        # ---------------------------------------------------------------------------
        if valid > 0 and validator_type in ("postcode", "iban", "company", "vat"):
            with st.expander(f"Valid rows with enriched data ({valid} rows)"):
                valid_rows = result_df[result_df[valid_col] == True][
                    [selected_column, detail_col]
                ].rename(columns={
                    selected_column: "Original Value",
                    detail_col: "API Response"
                })
                st.dataframe(valid_rows.head(50), use_container_width=True)

        # ---------------------------------------------------------------------------
        # Save to session state for use elsewhere
        # ---------------------------------------------------------------------------
        st.session_state["ext_validation_result"] = result_df
        st.session_state["ext_validation_column"] = selected_column
        st.session_state["ext_validation_type"] = validator_type

        # Option to download
        st.divider()
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name=f"ext_validation_{selected_column}_{validator_type}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Validation failed: {str(e)}")
        with st.expander("Error details"):
            import traceback
            st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Existing results
# ---------------------------------------------------------------------------
if st.session_state.get("ext_validation_result") is not None:
    prev_col = st.session_state.get("ext_validation_column")
    prev_type = st.session_state.get("ext_validation_type")
    if prev_col and prev_type:
        st.divider()
        st.caption(f"Previous result: '{prev_col}' via {VALIDATOR_LABELS.get(prev_type, prev_type)}")

# ---------------------------------------------------------------------------
# Help / API reference
# ---------------------------------------------------------------------------
with st.expander("API Reference & Rate Limits"):
    st.markdown("""
| Validator | API | Free Tier | Key Required |
|-----------|-----|-----------|-------------|
| Postcode | postcodes.io | Unlimited (bulk endpoint) | No |
| Email | MX DNS check | Unlimited | No |
| VAT | HMRC VAT API | Unlimited | No |
| IBAN | ISO 13616 checksum | Unlimited (offline) | No |
| Phone | phonenumbers library | Unlimited (offline) | No |
| Sort Code | Format validation | Unlimited (offline) | No |
| Company | Companies House API | 600 req/min | Yes (free) |
| Custom | Your endpoint | Depends on your API | Configurable |

**Privacy:** Only the individual cell value is sent to the API -- never the full row or any other columns.

**Caching:** Each unique value is only checked once per session. Repeated values use the cached result.
    """)
