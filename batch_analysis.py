"""
batch_analysis.py — Batch prediction page for StudentPredict.

In app.py:
    from batch_analysis import render_batch_page, apply_theme_fixes
    ...
    apply_theme_fixes(NAV_ICONS)            # right after the main <style> block
    ...
    elif page == "Batch Analysis":
        render_batch_page(model, scaler, FEATURE_NAMES, page_header, stat_card, render_table, themed_chart)

Behaviour
  * Uploaded CSV / Excel  -> columns are detected automatically and the model runs immediately.
  * "Use sample data"     -> unchanged: match columns, then press "Run Batch Prediction".
Prediction logic matches the single-student page: the seven features are scaled with the saved
scaler in FEATURE_NAMES order, class 1 = On-Track (passing), class 0 = At-Risk.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

TEAL = "#14b8a6"
RED = "#ef4444"
AMBER = "#f59e0b"
NONE_OPTION = "— none —"

# Used only to WARN about values that look like a different scale than the training data.
RANGES = {
    "Study_Hours_Weekly": (2, 35),
    "Attendance_Pct": (40, 100),
    "Past_GPA": (3, 10),
    "Sleep_Hours_Daily": (4, 10),
    "Extracurricular": (0, 1),
    "Family_Relations": (1, 5),
    "Social_Engagement": (1, 5),
}

# Typical values used when an uploaded file has no column for a feature.
DEFAULTS = {
    "Study_Hours_Weekly": 14.0,
    "Attendance_Pct": 80.0,
    "Past_GPA": 7.0,
    "Sleep_Hours_Daily": 7.0,
    "Extracurricular": 0,
    "Family_Relations": 3,
    "Social_Engagement": 3,
}

LABELS = {
    "Study_Hours_Weekly": "Weekly Study Hours",
    "Attendance_Pct": "Attendance (%)",
    "Past_GPA": "Past GPA / Score",
    "Sleep_Hours_Daily": "Daily Sleep Hours",
    "Extracurricular": "Extracurricular (Yes/No)",
    "Family_Relations": "Family Relations (1–5)",
    "Social_Engagement": "Social Engagement (1–5)",
}

# Exact (normalised) column names that are recognised for each feature.
ALIASES = {
    "Study_Hours_Weekly": ["study_hours_weekly", "study_hours_per_week", "study_hours", "weekly_study_hours", "study_hrs"],
    "Attendance_Pct": ["attendance_pct", "attendance_rate", "attendance", "attendance_percentage"],
    "Past_GPA": ["past_gpa", "gpa", "cgpa", "past_score", "past_exam_scores", "past_exam_score", "previous_score", "previous_gpa"],
    "Sleep_Hours_Daily": ["sleep_hours_daily", "sleep_hours", "daily_sleep_hours", "sleep"],
    "Extracurricular": ["extracurricular", "extracurricular_activities", "activities"],
    "Family_Relations": ["family_relations", "family_relationship", "family_relations_quality"],
    "Social_Engagement": ["social_engagement", "social_engagement_level", "social"],
}
# Fallback: any unused column whose name contains one of these words.
TOKENS = {
    "Study_Hours_Weekly": ["study"],
    "Attendance_Pct": ["attend"],
    "Past_GPA": ["gpa", "past", "previous"],
    "Sleep_Hours_Daily": ["sleep"],
    "Extracurricular": ["extra", "activit"],
    "Family_Relations": ["family"],
    "Social_Engagement": ["social"],
}
ID_ALIASES = ["student_id", "id", "roll_no", "roll_number", "student_name", "name"]


# ------------------------------------------------------------------ theme / widget CSS
THEME_FIX_CSS = """
@keyframes navSlide { from {opacity:0; transform:translateX(-22px);} to {opacity:1; transform:translateX(0);} }

/* ===== Sidebar navigation (Streamlit's newer radio component) ===== */
[data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap:6px; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div { width:100%; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    position:relative; overflow:hidden; width:100%; cursor:pointer;
    padding:11px 14px 11px 16px; border-radius:12px; background:rgba(255,255,255,0);
    transition: background .35s cubic-bezier(.4,0,.2,1), transform .35s cubic-bezier(.4,0,.2,1),
                box-shadow .35s cubic-bezier(.4,0,.2,1), padding-left .35s cubic-bezier(.4,0,.2,1);
    animation: navSlide .55s cubic-bezier(.22,1,.36,1) both;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display:none; }  /* radio circle */
[data-testid="stSidebar"] [data-testid="stRadioOption"]::before {
    content:""; position:absolute; left:0; top:20%; bottom:20%; width:4px; border-radius:0 4px 4px 0;
    background:#5eead4; transform:scaleY(0); transition:transform .35s cubic-bezier(.4,0,.2,1);
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
    background:var(--side_hover); transform:translateX(5px); padding-left:20px;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover::before { transform:scaleY(.6); }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background:linear-gradient(90deg,#0f766e 0%,#14b8a6 100%);
    box-shadow:0 6px 18px rgba(20,184,166,.35); transform:translateX(5px); padding-left:20px;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before { transform:scaleY(1); background:#fff; }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { font-weight:700; }

/* ===== Dropdowns (selectbox) — follow the app theme, not the OS/browser theme ===== */
[data-testid="stSelectbox"] [role="group"] {
    background-color:var(--card) !important; border:1px solid var(--border) !important;
    border-radius:12px !important; box-shadow:0 2px 8px var(--shadow);
    transition:border-color .25s ease, box-shadow .25s ease;
}
[data-testid="stSelectbox"] [role="group"]:hover { border-color:#14b8a6 !important; box-shadow:0 8px 20px var(--shadow_hover); }
[data-testid="stSelectbox"] [role="group"]:focus-within {
    border-color:#14b8a6 !important; box-shadow:0 0 0 3px rgba(20,184,166,.22);
}
[data-testid="stSelectbox"] input[role="combobox"] {
    background:transparent !important; color:var(--text) !important;
    -webkit-text-fill-color:var(--text) !important; caret-color:#14b8a6;
}
[data-testid="stSelectbox"] button { background:transparent !important; color:var(--muted) !important; }
[data-testid="stSelectbox"] button svg { fill:var(--muted) !important; }

/* the opened list */
[data-testid="stSelectboxVirtualDropdown"] {
    background-color:var(--card) !important; border:1px solid var(--border) !important;
    border-radius:12px !important; box-shadow:0 18px 44px var(--shadow_hover) !important; overflow:hidden;
}
[data-testid="stSelectboxVirtualDropdown"] * { background-color:transparent !important; color:var(--text) !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"] { transition:background-color .2s ease, padding-left .2s ease; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {
    background-color:rgba(20,184,166,.20) !important; padding-left:18px;
}

/* ===== Alerts / tabs / expanders ===== */
[data-testid="stAlertContainer"] {
    background:var(--alert) !important; border:1px solid var(--border) !important;
    border-left:4px solid #14b8a6 !important; border-radius:12px !important;
}
[data-testid="stAlert"] *, [data-testid="stAlertContainer"] * { color:var(--text) !important; }
[role="tablist"] { border-bottom:1px solid var(--border); }
[role="tab"] { background:transparent !important; transition:background .25s ease; }
[role="tab"] * { color:var(--muted) !important; font-weight:600; }
[role="tab"][aria-selected="true"] * { color:var(--text) !important; }
[role="tab"]:hover { background:var(--track) !important; }
[data-testid="stExpander"] { background:var(--card); border:1px solid var(--border) !important; border-radius:14px; }
[data-testid="stExpander"] * { color:var(--text); }

/* ===== File uploader ===== */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section {
    background:var(--card) !important; border:2px dashed var(--border) !important; border-radius:16px !important;
    transition:border-color .3s ease, box-shadow .3s ease, transform .3s ease;
}
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploader"] section:hover {
    border-color:#14b8a6 !important; box-shadow:0 10px 26px var(--shadow_hover); transform:translateY(-2px);
}
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] section * { color:var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] section button {
    background:var(--card) !important; border:1px solid var(--border) !important;
}
[data-testid="stFileChip"] { background:var(--track) !important; border-radius:10px !important; }
[data-testid="stFileChip"] *, [data-testid="stFileChipName"] { color:var(--text) !important; }
[data-testid="stFileChip"] button { background:transparent !important; }
"""

BATCH_CSS = """
.batch-step {
    display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px;
    border-radius:50%; background:linear-gradient(135deg,#0f766e,#14b8a6);
    color:#fff !important; font-size:.8rem; font-weight:800; margin-right:9px;
}
.chip-row { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 4px; }
.chip {
    padding:5px 11px; border-radius:999px; font-size:.8rem; border:1px solid var(--border);
    background:var(--track); color:var(--text) !important;
}
.chip.ok { border-color:rgba(20,184,166,.55); background:rgba(20,184,166,.12); }
.chip.warn { border-color:rgba(245,158,11,.6); background:rgba(245,158,11,.12); }
"""


def apply_theme_fixes(nav_icons=None):
    """Inject CSS that makes newer Streamlit widgets follow the app's light/dark toggle."""
    icon_css = ""
    for i, icon in enumerate(nav_icons or [], start=1):
        icon_css += (
            f'[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-of-type({i}) [data-testid="stRadioOption"]'
            f'{{animation-delay:{0.1 + i * 0.06:.2f}s;}}'
            f'[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-of-type({i}) [data-testid="stMarkdownContainer"] p::before'
            f'{{content:"{icon}";margin-right:10px;display:inline-block;transition:transform .3s ease;}}'
            f'[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-of-type({i}) [data-testid="stRadioOption"]:hover '
            f'[data-testid="stMarkdownContainer"] p::before{{transform:scale(1.25) rotate(-8deg);}}'
        )
    st.markdown(f"<style>{THEME_FIX_CSS}{icon_css}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def _norm(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _read_upload(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)  # .xlsx needs: pip install openpyxl


def _to_number(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _to_binary(series):
    mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1, "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


def _normalise_scale(feat, num):
    """Convert common alternative scales to the scale the model was trained on."""
    valid = num.dropna()
    if valid.empty:
        return num, None
    top = valid.max()
    if feat == "Attendance_Pct" and top <= 1.0:
        return num * 100, "Attendance looked like a fraction (0–1), so it was converted to a percentage."
    if feat == "Past_GPA" and 10 < top <= 100:
        return num / 10, "Past score looked like a 0–100 mark, so it was converted to a 0–10 scale."
    return num, None


def _auto_map(columns, feature_names):
    lookup = {_norm(c): c for c in columns}
    used, mapping = set(), {}
    for feat in feature_names:
        hit = next((lookup[a] for a in ALIASES.get(feat, []) + [_norm(feat)] if a in lookup and lookup[a] not in used), None)
        mapping[feat] = hit
        if hit:
            used.add(hit)
    for feat in feature_names:
        if mapping[feat] is None:
            hit = next((c for c in columns if c not in used and any(t in _norm(c) for t in TOKENS.get(feat, []))), None)
            mapping[feat] = hit
            if hit:
                used.add(hit)
    id_hit = next((lookup[a] for a in ID_ALIASES if a in lookup and lookup[a] not in used), None)
    return mapping, id_hit


def _template(feature_names):
    return pd.DataFrame({
        "Student_ID": ["S001", "S002", "S003"],
        feature_names[0]: [18, 6, 25], feature_names[1]: [88, 58, 95], feature_names[2]: [7.8, 4.9, 8.9],
        feature_names[3]: [7, 5, 8], feature_names[4]: ["Yes", "No", "Yes"],
        feature_names[5]: [4, 2, 5], feature_names[6]: [3, 1, 4],
    })


def _sample_data(feature_names, n=40, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, n + 1)],
        feature_names[0]: rng.uniform(3, 34, n).round(1), feature_names[1]: rng.uniform(45, 100, n).round(1),
        feature_names[2]: rng.uniform(3.5, 9.8, n).round(1), feature_names[3]: rng.uniform(4.5, 9.5, n).round(1),
        feature_names[4]: rng.choice(["Yes", "No"], n),
        feature_names[5]: rng.integers(1, 6, n), feature_names[6]: rng.integers(1, 6, n),
    })


def _predict(model, scaler, features_df):
    scaled = scaler.transform(features_df.to_numpy(dtype=float))
    prediction = model.predict(scaled)
    is_pass = np.array([str(p).strip().lower() in ["1", "pass", "passed", "true"] for p in prediction])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)
        classes = list(getattr(model, "classes_", [0, 1]))
        pass_p = proba[:, classes.index(1)] * 100 if 1 in classes else np.where(is_pass, 100.0, 0.0)
        risk_p = proba[:, classes.index(0)] * 100 if 0 in classes else 100.0 - pass_p
    else:
        pass_p = np.where(is_pass, 100.0, 0.0)
        risk_p = 100.0 - pass_p
    return is_pass, pass_p, risk_p


def _risk_level(risk_prob):
    return "High" if risk_prob >= 70 else "Medium" if risk_prob >= 40 else "Low"


def _compute(model, scaler, feature_names, data, mapping, id_col, source):
    """Build the feature table from the chosen columns, run the model, return a result dict."""
    data = data.dropna(how="all")
    feats = pd.DataFrame(index=data.index)
    notes, defaulted = [], []

    for feat in feature_names:
        col = mapping.get(feat)
        if col in (None, NONE_OPTION):
            feats[feat] = DEFAULTS[feat]
            defaulted.append(feat)
            continue
        if feat == "Extracurricular":
            num = _to_binary(data[col])
        else:
            num, note = _normalise_scale(feat, _to_number(data[col]))
            if note:
                notes.append(note)
        missing = int(num.isna().sum())
        if missing:
            if num.notna().any():
                fill = num.mode().iloc[0] if feat == "Extracurricular" else float(num.median())
            else:
                fill = DEFAULTS[feat]
            num = num.fillna(fill)
            notes.append(f"{missing} missing value(s) in “{col}” were filled with {fill:g}.")
        feats[feat] = num

    is_pass, pass_p, risk_p = _predict(model, scaler, feats)

    out = pd.DataFrame(index=feats.index)
    out["Row"] = out.index + 1
    if id_col and id_col != NONE_OPTION:
        out["Student"] = data[id_col].astype(str)
    for feat in feature_names:
        out[feat] = feats[feat]
    out["Status"] = np.where(is_pass, "On-Track", "At-Risk")
    out["Passing Probability (%)"] = np.round(pass_p, 1)
    out["At-Risk Probability (%)"] = np.round(risk_p, 1)
    out["Risk Level"] = [_risk_level(r) for r in risk_p]

    odd = {}
    for feat, (lo, hi) in RANGES.items():
        if feat in feats and feat not in defaulted:
            count = int(((feats[feat] < lo) | (feats[feat] > hi)).sum())
            if count:
                odd[LABELS.get(feat, feat)] = count

    return {
        "df": out.reset_index(drop=True), "notes": notes, "odd": odd, "source": source,
        "defaulted": [LABELS.get(f, f) + f" = {DEFAULTS[f]:g}" for f in defaulted],
        "matched": len(feature_names) - len(defaulted),
    }


def _chart(fig, themed_chart, height=300):
    fig = themed_chart(fig, height)
    fig.update_xaxes(automargin=True)   # keep axis titles from overlapping tick labels
    fig.update_yaxes(automargin=True)
    colour = fig.layout.font.color
    if colour:
        fig.update_xaxes(tickfont_color=colour, title_font_color=colour)
        fig.update_yaxes(tickfont_color=colour, title_font_color=colour)
    st.plotly_chart(fig, theme=None, use_container_width=True)


# ------------------------------------------------------------------ results display
def _show_results(result, feature_names, stat_card, render_table, themed_chart):
    res = result["df"]
    total = len(res)
    n_pass = int((res["Status"] == "On-Track").sum())
    n_risk = total - n_pass

    st.write("")
    st.markdown("<div class='card-title'><span class='batch-step'>3</span>Batch results</div>", unsafe_allow_html=True)

    if result["defaulted"]:
        st.warning(
            f"{len(result['defaulted'])} of {len(feature_names)} features were not found in your file, so typical "
            "values were used for them (" + "; ".join(result["defaulted"]) + "). Results are approximate — "
            "add those columns for a more accurate analysis."
        )
    for note in result["notes"]:
        st.info(note)
    if result["odd"]:
        details = ", ".join(f"{k} ({v} value(s))" for k, v in result["odd"].items())
        st.info(
            "Some values are outside the ranges used on the prediction page: "
            f"{details}. This is fine if they are genuine, but check the scale if it looks wrong."
        )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        stat_card("👥", "Students Analysed", f"{total:,}", f"Source: {result['source']}")
    with k2:
        stat_card("✅", "On-Track", f"{n_pass:,}", f"{n_pass / total * 100:.1f}% of students")
    with k3:
        stat_card("⚠️", "At-Risk", f"{n_risk:,}", f"{n_risk / total * 100:.1f}% need support")
    with k4:
        stat_card("🎯", "Avg Passing Probability", f"{res['Passing Probability (%)'].mean():.1f}%", "Across all students")

    st.write("")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Status Split</div>", unsafe_allow_html=True)
            fig = px.pie(res, names="Status", hole=0.58, color="Status",
                         color_discrete_map={"On-Track": TEAL, "At-Risk": RED})
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=False)
            _chart(fig, themed_chart)
    with c2:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Risk Level</div>", unsafe_allow_html=True)
            levels = res["Risk Level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).reset_index()
            levels.columns = ["Risk Level", "Students"]
            fig = px.bar(levels, x="Risk Level", y="Students", color="Risk Level",
                         color_discrete_map={"Low": TEAL, "Medium": AMBER, "High": RED})
            fig.update_layout(showlegend=False)
            _chart(fig, themed_chart)
    with c3:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Passing Probability</div>", unsafe_allow_html=True)
            fig = px.histogram(res, x="Passing Probability (%)", nbins=10, color_discrete_sequence=[TEAL])
            fig.update_layout(yaxis_title="Students")
            _chart(fig, themed_chart)

    with st.container(border=True):
        st.markdown("<div class='card-title'>Student-level results</div>", unsafe_allow_html=True)
        f1, f2 = st.columns([1, 1.4])
        with f1:
            view = st.radio("Show", ["All", "At-Risk", "On-Track"], horizontal=True, key="batch_view")
        with f2:
            order = st.radio("Sort by", ["Highest risk first", "Lowest risk first", "Original order"],
                             horizontal=True, key="batch_sort")
        shown = res if view == "All" else res[res["Status"] == view]
        if order == "Highest risk first":
            shown = shown.sort_values("At-Risk Probability (%)", ascending=False)
        elif order == "Lowest risk first":
            shown = shown.sort_values("At-Risk Probability (%)", ascending=True)

        st.caption(f"Showing {min(len(shown), 200):,} of {len(shown):,} matching students.")
        render_table(shown.head(200))
        st.write("")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇️ Download all results (CSV)", data=res.to_csv(index=False).encode("utf-8"),
                               file_name="batch_predictions.csv", mime="text/csv", use_container_width=True)
        with d2:
            st.download_button("⬇️ Download at-risk students (CSV)",
                               data=res[res["Status"] == "At-Risk"].to_csv(index=False).encode("utf-8"),
                               file_name="batch_at_risk_students.csv", mime="text/csv", use_container_width=True)


# ------------------------------------------------------------------ page
def render_batch_page(model, scaler, feature_names, page_header, stat_card, render_table, themed_chart):
    apply_theme_fixes()
    st.markdown(f"<style>{BATCH_CSS}</style>", unsafe_allow_html=True)
    page_header(
        "Batch Analysis",
        "Upload a file with many students and predict On-Track / At-Risk status for all of them at once.",
    )

    if model is None or scaler is None:
        st.error("Batch prediction is unavailable until the model and scaler load successfully.")
        return

    st.session_state.setdefault("batch_sample", None)
    st.session_state.setdefault("batch_results", None)

    # ---------------- Step 1: upload ----------------
    with st.container(border=True):
        st.markdown("<div class='card-title'><span class='batch-step'>1</span>Upload student data</div>",
                    unsafe_allow_html=True)
        up_col, help_col = st.columns([1.6, 1], gap="large")
        with up_col:
            uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"],
                                        label_visibility="collapsed", key="batch_upload")
            st.caption("Upload any student CSV — the model will find the matching columns and analyse it automatically.")
        with help_col:
            st.markdown("<div class='small-muted'>Want a ready-made format? Download the template, or try the "
                        "sample data first.</div>", unsafe_allow_html=True)
            st.write("")
            st.download_button("⬇️ Download template (CSV)", data=_template(feature_names).to_csv(index=False).encode("utf-8"),
                               file_name="batch_template.csv", mime="text/csv", use_container_width=True)
            if st.button("🧪 Use sample data", use_container_width=True, disabled=uploaded is not None,
                         help="Remove the uploaded file to use the sample data." if uploaded is not None else None):
                st.session_state.batch_sample = _sample_data(feature_names)
                st.session_state.batch_results = None

    # =====================================================================
    # A) UPLOADED FILE  ->  automatic detection + automatic prediction
    # =====================================================================
    if uploaded is not None:
        try:
            data = _read_upload(uploaded)
        except Exception as exc:
            st.error("The file could not be read. Please upload a valid CSV or Excel file.")
            with st.expander("Technical details"):
                st.code(str(exc))
            return
        if data.empty:
            st.warning("The uploaded file has no rows.")
            return

        columns = list(data.columns)
        signature = f"{uploaded.name}_{len(data)}_{len(columns)}"
        auto_map, auto_id = _auto_map(columns, feature_names)
        found = [f for f in feature_names if auto_map[f]]

        with st.container(border=True):
            st.markdown("<div class='card-title'>🔎 Columns detected in your file</div>", unsafe_allow_html=True)
            st.caption(f"Read **{len(data):,}** rows and **{len(columns)}** columns from {uploaded.name}.")
            chips = "".join(
                f"<span class='chip ok'>{LABELS[f]} ← {auto_map[f]}</span>" if auto_map[f]
                else f"<span class='chip warn'>{LABELS[f]} — not found (typical value used)</span>"
                for f in feature_names
            )
            st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)

            mapping, id_col = dict(auto_map), auto_id

        if not any(mapping.values()):
            st.error("No matching columns were found in this file. Rename your columns to match the template "
                     "(for example Study_Hours_Weekly, Attendance_Pct, Past_GPA) and upload it again.")
            return

        try:
            with st.spinner("Analysing students..."):
                result = _compute(model, scaler, feature_names, data, mapping, id_col, uploaded.name)
        except Exception as exc:
            st.error("Prediction failed. Confirm that the scaler and model were trained with these seven "
                     "features, in the same order.")
            with st.expander("Technical details"):
                st.code(str(exc))
            return

        _show_results(result, feature_names, stat_card, render_table, themed_chart)
        return

    # =====================================================================
    # B) SAMPLE DATA  ->  unchanged flow: match columns, then press Run
    # =====================================================================
    data = st.session_state.batch_sample
    if data is None or data.empty:
        st.info("Upload a file or use the sample data to begin.")
        return

    source_name = "sample data"
    columns = list(data.columns)
    signature = f"{source_name}_{len(data)}_{len(columns)}"
    auto_map, auto_id = _auto_map(columns, feature_names)

    with st.container(border=True):
        st.markdown("<div class='card-title'><span class='batch-step'>2</span>Match your columns to the model features</div>",
                    unsafe_allow_html=True)
        st.caption(f"Loaded **{len(data):,}** rows and **{len(columns)}** columns from {source_name}. "
                   "Columns are matched automatically when the names are recognised.")
        options = [NONE_OPTION] + columns
        mapping = {}
        grid = st.columns(3, gap="large")
        for i, feat in enumerate(feature_names):
            default = options.index(auto_map[feat]) if auto_map[feat] in options else 0
            with grid[i % 3]:
                mapping[feat] = st.selectbox(LABELS[feat], options, index=default, key=f"bmap_{signature}_{feat}")
        with grid[len(feature_names) % 3]:
            id_default = options.index(auto_id) if auto_id in options else 0
            id_col = st.selectbox("Student ID / Name (optional)", options, index=id_default, key=f"bmap_{signature}_id")

        missing = [LABELS[f] for f, c in mapping.items() if c == NONE_OPTION]
        chosen = [c for c in mapping.values() if c != NONE_OPTION]
        if missing:
            st.warning("Select a column for: " + ", ".join(missing))
        elif len(set(chosen)) != len(chosen):
            st.warning("The same column is selected for more than one feature. Please check the mapping.")
        ready = not missing and len(set(chosen)) == len(chosen)
        run = st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True, disabled=not ready)

    if run and ready:
        try:
            with st.spinner("Analysing students..."):
                st.session_state.batch_results = _compute(model, scaler, feature_names, data, mapping, id_col, source_name)
        except Exception as exc:
            st.error("Prediction failed. Confirm that the scaler and model were trained with these seven "
                     "features, in the same order.")
            with st.expander("Technical details"):
                st.code(str(exc))
            return

    if st.session_state.batch_results:
        _show_results(st.session_state.batch_results, feature_names, stat_card, render_table, themed_chart)
        if st.button("🗑️ Clear results"):
            st.session_state.batch_results = None
            st.rerun()