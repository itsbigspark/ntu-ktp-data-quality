"""
Page 2: Rules -- Discover, review, and manage validation rules.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header, terminal_block
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("RULES ENGINE", "Discover and manage validation rules")

df_raw = st.session_state.get("df_raw")
df_ref = st.session_state.get("df_ref")
rules_json = st.session_state.get("rules_json")

if df_raw is None:
    terminal_block("// NO DATA LOADED<br><span style='color:#4a7a4f;'>Go to Load Data first.</span>")
    st.stop()

# ---------------------------------------------------------------------------
# Rule type colour coding
# ---------------------------------------------------------------------------
_TYPE_COLOURS = {
    "required":    "#00e5ff",
    "categorical":  "#00ff41",
    "uniqueness":  "#ffd700",
    "range":       "#ff9500",
    "format":      "#b388ff",
    "regex":       "#ff6b9d",
    "cross_field": "#40c4ff",
}

def _rule_badge(rule_type: str) -> str:
    colour = _TYPE_COLOURS.get(rule_type, "#888")
    return (
        f'<span style="background:rgba(0,0,0,0.4);border:1px solid {colour};'
        f'color:{colour};font-family:Share Tech Mono;font-size:0.7rem;'
        f'padding:2px 8px;border-radius:3px;letter-spacing:1px;">'
        f'{rule_type.upper()}</span>'
    )

def _render_rule(rule: dict, key: str) -> bool:
    """Render a single rule row with checkbox. Returns checked state."""
    rule_type  = rule.get("rule_type", "unknown")
    source     = rule.get("source", "")
    coverage   = rule.get("coverage", None)
    quality    = rule.get("quality_score", None)
    ex_match   = rule.get("examples_match", [])
    rule_data  = rule.get("rule_data", {})
    selected   = rule.get("selected", True)

    # Build human-readable description from rule_data
    desc_parts = []
    if "required" in rule_data:
        desc_parts.append("must not be null")
    if "allowed_values" in rule_data:
        vals = rule_data["allowed_values"]
        preview = ", ".join(str(v) for v in vals[:4])
        if len(vals) > 4:
            preview += f" … +{len(vals)-4} more"
        desc_parts.append(f"allowed: [{preview}]")
    if "min" in rule_data or "max" in rule_data:
        desc_parts.append(f"range: {rule_data.get('min','?')} – {rule_data.get('max','?')}")
    if "pattern" in rule_data:
        desc_parts.append(f"pattern: {rule_data['pattern']}")
    if "unique" in rule_data:
        desc_parts.append("must be unique")
    if not desc_parts:
        desc_parts.append(json.dumps(rule_data, default=str)[:80])

    description = " | ".join(desc_parts)

    col_check, col_info = st.columns([0.05, 0.95])
    with col_check:
        checked = st.checkbox("", value=selected, key=key, label_visibility="collapsed")
    with col_info:
        meta = ""
        if coverage is not None:
            meta += f'<span style="color:#4a7a4f;">cov {coverage*100:.0f}%</span>'
        if quality is not None:
            qcol = "#00ff41" if quality >= 90 else "#ffd700" if quality >= 70 else "#ff4444"
            meta += f'  <span style="color:{qcol};">q{quality}</span>'
        if source:
            meta += f'  <span style="color:#555;font-size:0.7rem;">[{source}]</span>'
        if ex_match:
            examples = ", ".join(str(v) for v in ex_match[:3])
            meta += f'  <span style="color:#444;font-size:0.7rem;">e.g. {examples}</span>'

        st.markdown(
            f'{_rule_badge(rule_type)}&nbsp;&nbsp;'
            f'<span style="color:#b0ffb8;font-family:Share Tech Mono;font-size:0.78rem;">{description}</span>'
            f'&nbsp;&nbsp;{meta}',
            unsafe_allow_html=True,
        )
    return checked


# ---------------------------------------------------------------------------
# Statistical rule generation
# ---------------------------------------------------------------------------
from dq_engine import RulesWorkflow
workflow = RulesWorkflow()

section_header("// Statistical Rule Discovery")

col_gen, col_status = st.columns([2, 1], gap="large")
with col_gen:
    gen_from_data = st.checkbox("Infer rules from uploaded data", value=True, key="gen_from_data")
    gen_from_ref  = st.checkbox(
        "Generate rules from reference dataset",
        value=df_ref is not None,
        key="gen_from_ref",
        disabled=df_ref is None,
    )
    gen_from_json = st.checkbox(
        "Include imported JSON rules",
        value=rules_json is not None,
        key="gen_from_json",
        disabled=rules_json is None,
    )

with col_status:
    st.markdown(
        f'<div class="glass-card">'
        f'<p style="color:#00e5ff;font-family:Orbitron;font-size:0.75rem;letter-spacing:2px;">SOURCES</p>'
        f'<p style="color:#b0ffb8;font-size:0.78rem;">'
        f'Data: {len(df_raw)} rows, {len(df_raw.columns)} cols<br>'
        f'Reference: {"Loaded" if df_ref is not None else "None"}<br>'
        f'JSON rules: {"Loaded" if rules_json else "None"}'
        f'</p></div>',
        unsafe_allow_html=True,
    )

if st.button("GENERATE STATISTICAL RULES", key="generate_rules", use_container_width=True):
    with st.spinner("Discovering rules from data patterns..."):
        try:
            generated = workflow.generate(
                df_raw=df_raw,
                df_ref=df_ref if gen_from_ref else None,
                rules_json=rules_json if gen_from_json else None,
                use_inferred=gen_from_data,
                use_reference=gen_from_ref,
                use_json=gen_from_json,
            )
            st.session_state["generated_rules"] = generated
            st.session_state["rules_merged"] = generated
            total = sum(len(v) for v in generated.values() if isinstance(v, list))
            st.success(f"Rules generated: {len(generated)} columns, {total} rules")
        except Exception as e:
            st.error(f"Rule generation failed: {e}")

# ---------------------------------------------------------------------------
# Display statistical rules per column
# ---------------------------------------------------------------------------
generated = st.session_state.get("generated_rules")

if generated:
    section_header("// Rules by Column")

    rule_selections = st.session_state.setdefault("rule_selections", {})

    if isinstance(generated, dict) and "columns" in generated:
        # Standard format {"columns": {col: rules_dict}}
        columns_dict = generated["columns"]
        for col_name, col_rules in columns_dict.items():
            with st.expander(f"{col_name}", expanded=False):
                if isinstance(col_rules, dict):
                    for key, val in col_rules.items():
                        st.markdown(
                            f'<div style="font-family:Share Tech Mono;font-size:0.78rem;'
                            f'color:#b0ffb8;padding:4px 0;">'
                            f'<span style="color:#00e5ff;">{key}:</span> {val}</div>',
                            unsafe_allow_html=True,
                        )

    elif isinstance(generated, dict):
        # Enhanced format {col: [rule_dicts]}
        for col_name, rules_list in generated.items():
            if not isinstance(rules_list, list) or len(rules_list) == 0:
                continue

            selected_count = sum(1 for r in rules_list if r.get("selected", True))
            with st.expander(
                f"{col_name}  —  {len(rules_list)} rules  ({selected_count} active)",
                expanded=False,
            ):
                for i, rule in enumerate(rules_list):
                    if not isinstance(rule, dict):
                        continue
                    key = f"rule_{col_name}_{i}"
                    checked = _render_rule(rule, key)
                    rule_selections[f"{col_name}_{i}"] = checked
                    st.markdown("<hr style='border-color:rgba(0,255,65,0.08);margin:4px 0;'>", unsafe_allow_html=True)

    # Export
    section_header("// Export Rules")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            "DOWNLOAD RULES [JSON]",
            data=workflow.to_json(generated),
            file_name="validation_rules.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_exp2:
        if st.button("SAVE TO SESSION", key="save_rules", use_container_width=True):
            st.session_state["rules_merged"] = generated
            st.success("Rules saved to session — ready for validation")

else:
    terminal_block(
        "// NO RULES GENERATED YET<br>"
        "<span style='color:#4a7a4f;'>Click GENERATE STATISTICAL RULES above to discover validation rules from your data.</span>"
    )

# ---------------------------------------------------------------------------
# AI Rule Suggestions
# ---------------------------------------------------------------------------
section_header("// AI Rule Suggestions")

st.markdown(
    '<p style="color:#b0ffb8;font-family:Share Tech Mono;font-size:0.8rem;">'
    'Claude analyses your column metadata to suggest semantic rules that statistics alone cannot infer — '
    'e.g. email format, non-negative balances, valid country codes, postcode patterns.</p>',
    unsafe_allow_html=True,
)

provider_type = st.session_state.get("ai_provider_type", "ollama")
_api_key      = st.session_state.get("anthropic_api_key", "")
_model        = st.session_state.get("anthropic_model", "claude-opus-4-6")

if provider_type != "anthropic" or not _api_key:
    st.warning("Configure Anthropic API in Settings to use AI Rule Suggestions.")
else:
    # ── Mode selector ────────────────────────────────────────────────────────
    data_aware = st.toggle(
        "Data-Aware Mode — send sample values to AI",
        value=False,
        key="ai_rules_data_aware",
    )

    if data_aware:
        st.warning(
            "Data-Aware Mode is ON. Up to 5 sample values per column from your dataset "
            "(and reference dataset if loaded) will be sent to the Anthropic API. "
            "Disable this if your data contains PII or sensitive information."
        )
    else:
        st.info(
            "Safe Mode: only column names, data types, and statistics are sent to the API. "
            "No actual data values leave your environment."
        )

    def _build_col_stats(df: pd.DataFrame, label: str) -> list[str]:
        """Build statistics-only summary for each column — no actual values."""
        import re as _re
        lines = []
        for col in df.columns:
            series   = df[col]
            dtype    = str(series.dtype)
            null_pct = round(series.isna().mean() * 100, 1)
            n_unique = series.nunique()
            n_total  = len(series)
            parts    = [f"dtype={dtype}", f"null={null_pct}%", f"unique={n_unique}/{n_total}"]

            if pd.api.types.is_numeric_dtype(series):
                s = series.dropna()
                if len(s):
                    parts += [
                        f"min={s.min():.4g}", f"max={s.max():.4g}",
                        f"mean={s.mean():.4g}", f"neg_pct={round((s < 0).mean()*100,1)}%",
                    ]
            else:
                s = series.dropna().astype(str)
                if len(s):
                    lengths = s.str.len()
                    parts += [f"avg_len={lengths.mean():.1f}", f"max_len={int(lengths.max())}"]
                    # Pattern detection
                    email_pct   = round(s.str.match(r'^[^@]+@[^@]+\.[^@]+$').mean()*100, 1)
                    date_pct    = round(s.str.match(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}').mean()*100, 1)
                    phone_pct   = round(s.str.match(r'^[\+\d\s\-\(\)]{7,15}$').mean()*100, 1)
                    post_pct    = round(s.str.match(r'^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$').mean()*100, 1)
                    if email_pct > 50:  parts.append(f"looks_like=email({email_pct}%)")
                    if date_pct  > 50:  parts.append(f"looks_like=date({date_pct}%)")
                    if phone_pct > 50:  parts.append(f"looks_like=phone({phone_pct}%)")
                    if post_pct  > 50:  parts.append(f"looks_like=uk_postcode({post_pct}%)")

            lines.append(f"[{label}] {col}: {', '.join(parts)}")
        return lines

    def _build_col_with_samples(df: pd.DataFrame, label: str, n: int = 5) -> list[str]:
        """Statistics + actual sample values."""
        stat_lines = _build_col_stats(df, label)
        sample_lines = []
        for i, col in enumerate(df.columns):
            samples = df[col].dropna().head(n).tolist()
            sample_lines.append(f"[{label} samples] {col}: {samples}")
        return stat_lines + sample_lines

    if st.button("GET AI RULE SUGGESTIONS", key="ai_rules_btn", use_container_width=True):

        # Build column info based on mode
        if data_aware:
            col_lines = _build_col_with_samples(df_raw, "main_data")
            if df_ref is not None:
                col_lines += _build_col_with_samples(df_ref, "reference_data")
        else:
            col_lines = _build_col_stats(df_raw, "main_data")
            if df_ref is not None:
                col_lines += _build_col_stats(df_ref, "reference_data")

        # Summarise existing JSON rules so AI doesn't duplicate
        json_rules_summary = ""
        if rules_json:
            try:
                covered = list(rules_json.get("columns", rules_json).keys())
                json_rules_summary = (
                    f"\n\nExisting JSON rules already cover these columns: {covered}. "
                    "Do not suggest rules that duplicate what is already defined."
                )
            except Exception:
                pass

        ref_note = ""
        if df_ref is not None:
            ref_note = (
                "\n\nReference dataset (label: reference_data) represents clean, valid data. "
                "Use its statistics/samples to understand what valid values look like for each column."
            )

        prompt = (
            "You are a data quality expert. Given the following column metadata from a dataset, "
            "suggest specific validation rules for each column. Focus on semantic meaning that "
            "statistics alone cannot infer — e.g. email format validation, non-negative constraint "
            "for financial balances, ISO date format, valid country codes, UK postcode patterns, "
            "phone number formats, etc.\n\n"
            "For each column that needs a rule, respond in this EXACT JSON format (array only):\n"
            "[\n"
            "  {\"column\": \"col_name\", \"rule_type\": \"format|range|required|categorical|regex\", "
            "\"description\": \"human readable rule description\", \"rule_data\": {\"key\": \"value\"}}\n"
            "]\n\n"
            "Rules:\n"
            "- Only include columns where you can infer a meaningful SEMANTIC rule\n"
            "- Do NOT duplicate rules already covered by the JSON rules summary below\n"
            "- Return ONLY the JSON array — no markdown, no explanation, no code fences\n"
            + ref_note
            + json_rules_summary
            + "\n\nColumn metadata:\n"
            + "\n".join(col_lines)
        )

        with st.spinner("Claude is analysing your columns..."):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=_api_key)
                response = client.messages.create(
                    model=_model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = "\n".join(raw.split("\n")[1:])
                if raw.endswith("```"):
                    raw = "\n".join(raw.split("\n")[:-1])
                ai_rules = json.loads(raw)
                st.session_state["ai_suggested_rules"] = ai_rules
                st.success(f"Claude suggested {len(ai_rules)} rules across your columns.")
            except json.JSONDecodeError:
                st.session_state["ai_suggested_rules_raw"] = raw
                st.error("Claude response was not valid JSON. Raw response shown below.")
                st.code(raw)
            except Exception as e:
                st.error(f"AI rule generation failed: {e}")

# Display AI suggested rules
ai_rules = st.session_state.get("ai_suggested_rules")
if ai_rules:
    st.markdown(
        '<p style="color:#00e5ff;font-family:Orbitron;font-size:0.75rem;'
        'letter-spacing:2px;margin-top:12px;">AI SUGGESTED RULES</p>',
        unsafe_allow_html=True,
    )

    ai_selections = st.session_state.setdefault("ai_rule_selections", {})

    # Group by column
    from collections import defaultdict
    by_col = defaultdict(list)
    for r in ai_rules:
        by_col[r.get("column", "unknown")].append(r)

    for col_name, col_ai_rules in by_col.items():
        with st.expander(f"{col_name}  —  {len(col_ai_rules)} AI suggestions", expanded=True):
            for i, rule in enumerate(col_ai_rules):
                key = f"ai_rule_{col_name}_{i}"
                col_check, col_info = st.columns([0.05, 0.95])
                with col_check:
                    checked = st.checkbox("", value=True, key=key, label_visibility="collapsed")
                    ai_selections[key] = checked
                with col_info:
                    rule_type   = rule.get("rule_type", "format")
                    description = rule.get("description", "")
                    rule_data   = rule.get("rule_data", {})
                    st.markdown(
                        f'{_rule_badge(rule_type)}&nbsp;&nbsp;'
                        f'<span style="color:#b0ffb8;font-family:Share Tech Mono;font-size:0.78rem;">{description}</span>'
                        f'<br><span style="color:#4a7a4f;font-size:0.72rem;">{json.dumps(rule_data)}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown("<hr style='border-color:rgba(0,229,255,0.08);margin:4px 0;'>", unsafe_allow_html=True)

    if st.button("MERGE AI RULES INTO SESSION", key="merge_ai_rules", use_container_width=True):
        # Merge selected AI rules into the existing rules structure
        base = st.session_state.get("generated_rules") or {}
        for rule in ai_rules:
            col   = rule.get("column")
            key_check = f"ai_rule_{col}_{ai_rules.index(rule)}"
            if not ai_selections.get(key_check, True):
                continue
            if col not in base:
                base[col] = []
            base[col].append({
                "rule_type":    rule.get("rule_type", "format"),
                "source":       "ai",
                "description":  rule.get("description", ""),
                "rule_data":    rule.get("rule_data", {}),
                "coverage":     None,
                "quality_score": None,
                "selected":     True,
            })
        st.session_state["generated_rules"] = base
        st.session_state["rules_merged"]    = base
        st.success("AI rules merged — ready for validation.")
        st.rerun()
