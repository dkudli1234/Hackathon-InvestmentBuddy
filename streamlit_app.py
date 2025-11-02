# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ───────────────────────────────  PAGE & THEME  ───────────────────────────────
st.set_page_config(
    page_title="Investment Buddy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS (typography, cards, buttons) ---
st.markdown(
    """
<style>
:root {
  --ink-1:#0b1020; --ink-2:#0f172a; --ink-3:#111827;
  --panel:#0c1325; --panel-2:#101a33; --line:#253044;
  --accent:#06b6d4; --accent-2:#22c55e;
}
.block-container { max-width: 1100px; padding-top: 46px !important; }
.hero { font-size: clamp(2.0rem, 2.1vw + 1rem, 2.6rem); font-weight: 850; line-height: 1.15; margin: 0 0 6px 0; }
.herosub { color:#98a2b3; font-size: 1.02rem; margin-bottom: 10px; }
.card { border:1px solid var(--line); border-radius: 16px;
        background: radial-gradient(1200px 200px at 0% 0%, rgba(34,197,94,.06), transparent 40%),
                   linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.00));
        padding: 1.0rem 1.15rem; }
.section-title { font-weight: 850; font-size: 1.22rem; margin: 6px 0 8px; }
.q { font-weight: 750; margin: 4px 0 4px; }
.help { color:#94a3b8; font-size:.92rem; margin:-2px 0 10px; }
.badge { display:inline-block; padding:.22rem .55rem; border-radius:999px; background:#0b1323; border:1px solid #273244; font-size:.82rem; margin:.18rem .3rem .18rem 0; }
div.stButton > button[kind="primary"] { border-radius: 12px; padding:.6rem 1rem; font-weight:800; }
.navbar { display:flex; gap:10px; justify-content:space-between; padding-top:8px; }
.small { font-size: .92rem; color:#9ca3af; }
footer { color:#9ca3af; font-size:.86rem; margin-top:22px; }
</style>
""",
    unsafe_allow_html=True
)

# ───────────────────────────────  CONSTANTS  ───────────────────────────────
DISPLAY_LABELS = [
    "💸 Make more money",
    "💼 Companies already making good profits",
    "🛡️ Not too risky",
    "💧 Has enough cash",
    "⚙️ Runs smoothly",
    "🚀 Spends on future/new ideas",
]
DISPLAY_TO_BASE = {
    "💸 Make more money": "Make more money",
    "💼 Companies already making good profits": "Companies already making good profits",
    "🛡️ Not too risky": "Not too risky",
    "💧 Has enough cash": "Has enough cash",
    "⚙️ Runs smoothly": "Runs smoothly",
    "🚀 Spends on future/new ideas": "Spends on future/new ideas",
}
BASE_TO_BUCKET = {
    "Make more money": "Return",
    "Companies already making good profits": "Profit",
    "Not too risky": "Risk",
    "Has enough cash": "Liquidity",
    "Runs smoothly": "Efficiency",
    "Spends on future/new ideas": "Future",
}
BUCKET_ORDER = ["Return", "Profit", "Risk", "Liquidity", "Efficiency", "Future"]
BUCKET_DESCRIPTIONS = {
    "Return": "potential short-term upside",
    "Profit": "companies already producing healthy earnings",
    "Risk": "how fragile/debt-heavy the sector is",
    "Liquidity": "cash/near-term financing comfort",
    "Efficiency": "how smoothly assets are used",
    "Future": "investment in innovation & pipelines",
}
# Harmonious colors used across charts
CRITERIA_COLORS = {
    "Return": "#19a974",     # teal/green
    "Profit": "#f6c431",     # gold
    "Risk": "#ef4444",       # red
    "Liquidity": "#3b82f6",  # blue
    "Efficiency": "#8b5cf6", # purple
    "Future": "#06b6d4",     # cyan
}
DEFAULT_DATA_PATH = "data/industry_ratings_latest.csv"

# ───────────────────────────────  SIDEBAR  ───────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Investment Buddy")
    st.write(
        "A beginner-friendly way to figure out **which industries to invest in** next month. "
        "Tell us your goals; we translate them into simple weights and blend them with our predictive model to rank industries."
    )
    st.markdown("**How it works**")
    st.write("1) Set preferences\n2) Tune weights (sensitivity)\n3) See market snapshot\n4) Get ranked picks")
    st.markdown("**The six drivers**")
    emoji_map = {"Return":"💸","Profit":"💼","Risk":"🛡️","Liquidity":"💧","Efficiency":"⚙️","Future":"🚀"}
    for b in BUCKET_ORDER:
        st.markdown(f"- {emoji_map[b]} **{b}** — {BUCKET_DESCRIPTIONS[b]}")
    st.markdown("---")
    st.caption("Plain English up front; smarter math under the hood.")

# ───────────────────────────────  STEP NAV STATE  ──────────────────────────
qp = st.query_params
if "step" not in qp:
    qp["step"] = "1"
active_step = int(qp.get("step", "1"))

def go_to(step: int):
    st.query_params["step"] = str(step)
    st.rerun()  # single-click navigation

# ───────────────────────────────  HEADER  ──────────────────────────────────
st.markdown('<div class="hero">Welcome to your Personal Investment Buddy</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="herosub">Answer a few simple questions. We’ll turn them into clear 1–5 priorities and then blend them with a predictive model to suggest industries to consider.</div>',
    unsafe_allow_html=True
)

# Single navigation row (numbered). No duplicate stepper.
cols = st.columns(4)
labels = ["Preferences", "Weights", "Market View", "Top Picks"]
for i, c in enumerate(cols, start=1):
    with c:
        if st.button(f"{i}. {labels[i-1]}", use_container_width=True):
            go_to(i)

# ───────────────────────────────  HELPERS  ────────────────────────────────
def init_weights_from_persona(p):
    """Create initial weights (percent) from answers + small nudges."""
    decay12 = 0.62 if p["gap_1_vs_2"] == "A lot" else 0.78
    decay23 = 0.68 if p["gap_2_vs_3"] == "A lot" else 0.82
    decay_rest = 0.85
    ranks = p["rank_order_buckets"]
    w = [0]*6
    w[0] = 1.0
    w[1] = w[0]*decay12
    w[2] = w[1]*decay23
    for i in range(3,6):
        w[i] = w[i-1]*decay_rest
    raw = {ranks[i]: w[i] for i in range(6)}

    # nudges: risk/return and liquidity attitude
    if p["risk_vs_return"] == "Be careful":
        raw["Risk"] = raw.get("Risk",0)*1.10; raw["Return"] = raw.get("Return",0)*0.90
    else:
        raw["Return"] = raw.get("Return",0)*1.10; raw["Risk"] = raw.get("Risk",0)*0.90
    if p["keep_cash_buffer"] == "Yes, keep some":
        raw["Liquidity"] = raw.get("Liquidity",0)*1.07; raw["Return"] = raw.get("Return",0)*0.96; raw["Future"] = raw.get("Future",0)*0.97
    else:
        raw["Liquidity"] = raw.get("Liquidity",0)*0.93; raw["Return"] = raw.get("Return",0)*1.04; raw["Future"] = raw.get("Future",0)*1.03

    vec = np.array([raw.get(b,0.0) for b in BUCKET_ORDER], float)
    if vec.sum() == 0: vec[:] = 1/6
    vec /= vec.sum()
    pct = (vec*100).round(2)
    return {BUCKET_ORDER[i]: float(pct[i]) for i in range(6)}

def fix_rounding_to_100(values_dict, pivot_key):
    total = round(sum(values_dict.values()), 2)
    diff = round(100.0 - total, 2)
    if abs(diff) < 0.01:
        return values_dict
    for k in BUCKET_ORDER[::-1]:
        if k != pivot_key:
            values_dict[k] = round(values_dict[k] + diff, 2)
            break
    return values_dict

def rebalance_on_change(changed_bucket):
    wc = st.session_state["weights_pct"]
    new_v = float(st.session_state[f"w_{changed_bucket}"])
    wc[changed_bucket] = new_v

    others = [b for b in BUCKET_ORDER if b != changed_bucket]
    s = sum(float(st.session_state.get(f"w_{b}", wc[b])) for b in others)
    target = max(0.0, 100.0 - new_v)

    if s <= 0:
        even = round(target/len(others), 2)
        for ob in others:
            st.session_state[f"w_{ob}"] = even; wc[ob] = even
    else:
        scale = target / s
        for ob in others:
            new_ob = round(float(st.session_state.get(f"w_{ob}", wc[ob])) * scale, 2)
            st.session_state[f"w_{ob}"] = new_ob; wc[ob] = new_ob

    fix_rounding_to_100(wc, changed_bucket)
    for b in BUCKET_ORDER:
        st.session_state[f"w_{b}"] = wc[b]

def load_industry_table_from_default():
    # Supports CSV or XLSX at DEFAULT_DATA_PATH
    try:
        df = pd.read_csv(DEFAULT_DATA_PATH)
    except Exception:
        try:
            df = pd.read_excel(DEFAULT_DATA_PATH)
        except Exception:
            return None
    df.columns = [c.strip() for c in df.columns]
    if "Profitability" in df.columns and "Profit" not in df.columns:
        df = df.rename(columns={"Profitability": "Profit"})
    if "Industry " in df.columns and "Industry" not in df.columns:
        df = df.rename(columns={"Industry ": "Industry"})
    needed = ["Industry","Return","Profit","Risk","Liquidity","Efficiency","Future"]
    for c in needed[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed[1:])
    return df[needed]

def equal_weight_chart(df, top_n=10):
    """
    Stacked 1–5 per-criterion bars; axis is total out of 30; sorted high→low.
    (Numbers inside segments removed to avoid cramped labels.)
    """
    crits = BUCKET_ORDER
    view = df.copy()
    view["Total_30"] = view[crits].sum(axis=1)
    top = view.sort_values("Total_30", ascending=False).head(top_n)

    long = top.melt(id_vars=["Industry"], value_vars=crits, var_name="Criterion", value_name="Score")
    order = top.sort_values("Total_30", ascending=False)["Industry"].tolist()

    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y("Industry:N", sort=order, title=""),
            x=alt.X("sum(Score):Q",
                    scale=alt.Scale(domain=[0,30]),
                    title="1–30 total (sum of six 1–5 scores)"),
            color=alt.Color("Criterion:N",
                            legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=list(CRITERIA_COLORS.keys()),
                                            range=list(CRITERIA_COLORS.values()))),
            tooltip=["Industry","Criterion","Score"]
        )
        .properties(height=max(36*len(order), 220))
    )
    st.altair_chart(chart, use_container_width=True)

def ahp_rank(df, weights_vec):
    crits = BUCKET_ORDER
    contrib = df.copy()
    for c in crits:
        contrib[c] = contrib[c] * float(weights_vec[c])  # contribution
    ranked = contrib[["Industry"] + crits].copy()
    ranked["Priority score"] = ranked[crits].sum(axis=1)
    ranked = ranked.sort_values("Priority score", ascending=False).reset_index(drop=True)
    contrib_sorted = contrib.loc[ranked.index, ["Industry"] + crits].reset_index(drop=True)
    return ranked, contrib_sorted

def implied_consistency(weights_vec):
    crits = BUCKET_ORDER
    w = np.array([weights_vec[c] for c in crits], dtype=float)
    n = len(w)
    pcm = np.outer(w, 1.0 / w)   # perfectly consistent from ratios
    vals, _ = np.linalg.eig(pcm)
    lam_max = np.max(np.real(vals))
    CI = (lam_max - n) / (n - 1)
    RI = 1.24  # Saaty RI for n=6
    CR = CI / RI if RI > 0 else 0.0
    return float(lam_max), float(CI), float(CR)

# ───────────────────────────────  STEP 1: PREFERENCES  ─────────────────────
if active_step == 1:
    st.markdown('<div class="section-title">Tell us what you care about</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form("persona_form", clear_on_submit=False):
        st.markdown('<div class="q">Q1. What’s the primary goal of this portfolio right now?</div>', unsafe_allow_html=True)
        q1 = st.radio(
            "", ["Making more money", "Staying safe", "Having cash available", "Investing in the future"],
            index=None, help="We’ll keep this in mind while translating your answers into weights."
        )

        st.markdown('<div class="q" style="margin-top:10px;">Q2. Rank these six drivers (1 → 6)</div>', unsafe_allow_html=True)
        st.markdown('<div class="help">Click in the order you care about. Once all six are selected, we’ll lock the list. You can edit later.</div>', unsafe_allow_html=True)

        locked = st.session_state.get("q2_locked", False)
        current_order = st.session_state.get("q2_order", [])

        if not locked:
            q2_order = st.multiselect(
                "Pick all six (1 = first clicked):", options=DISPLAY_LABELS,
                default=current_order, max_selections=6,
                help="To change the order, clear and re-select in your desired order."
            )
            # Auto-lock as soon as all six are chosen
            if len(q2_order) == 6 and not st.session_state.get("q2_locked", False):
                st.session_state["q2_order"] = q2_order
                st.session_state["q2_locked"] = True
                st.rerun()
        else:
            order_text = " ".join([f"{i+1}. {lab}" for i, lab in enumerate(current_order)])
            st.markdown(f"**Your ranking:**  \n{order_text}")

        c1, c2 = st.columns(2)
        with c1:
            q3 = st.radio(
                "Q3. How much stronger is your #1 vs #2?",
                ["A little", "A lot"], index=None, horizontal=True
            )
        with c2:
            q4 = st.radio(
                "Q4. How much stronger is your #2 vs #3?",
                ["A little", "A lot"], index=None, horizontal=True
            )

        c3, c4 = st.columns(2)
        with c3:
            q5 = st.radio(
                "Q5. If good money requires taking more risk, which way do you lean?",
                ["Be careful", "Go for it"], index=None, horizontal=True
            )
        with c4:
            q6 = st.radio(
                "Q6. Keep a safety cash buffer?",
                ["Yes, keep some", "No, not needed"], index=None, horizontal=True
            )

        left, right = st.columns([1,1])
        with left:
            submitted = st.form_submit_button("Save & go to Weights →", use_container_width=True)
        with right:
            reset = st.form_submit_button("Reset form", use_container_width=True)

    if reset:
        for k in ("persona","weights_pct","weights_vec","industry_table","q2_locked","q2_order"):
            if k in st.session_state: del st.session_state[k]
        st.rerun()

    if submitted:
        errs = []
        if q1 is None: errs.append("• Please answer Q1.")
        if not st.session_state.get("q2_locked", False): errs.append("• Please select **all six** items in your desired order.")
        if q3 is None: errs.append("• Please answer Q3.")
        if q4 is None: errs.append("• Please answer Q4.")
        if q5 is None: errs.append("• Please answer Q5.")
        if q6 is None: errs.append("• Please answer Q6.")

        if errs:
            st.error("Almost there—fix these:\n\n" + "\n".join(errs))
        else:
            base_order = [DISPLAY_TO_BASE[d] for d in st.session_state["q2_order"]]
            bucket_order = [BASE_TO_BUCKET[b] for b in base_order]
            st.session_state["persona"] = {
                "objective_now": q1,
                "rank_order_labels": base_order,
                "rank_order_buckets": bucket_order,
                "gap_1_vs_2": q3,
                "gap_2_vs_3": q4,
                "risk_vs_return": q5,
                "keep_cash_buffer": q6
            }
            st.success("Saved! Your preferences are locked in.")
            go_to(2)

    # outside the form: unlock control
    if st.session_state.get("q2_locked", False):
        if st.button("Change ranking", help="Unlock and edit your ranking"):
            st.session_state["q2_locked"] = False
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<footer>Created by <b>Dhruvaraj Kudli</b>; curated with thanks to <b>Aaryan Kharambale</b> & <b>Naveen Akoju</b> for contributions to the regression and AHP logic.</footer>', unsafe_allow_html=True)

# ───────────────────────────────  STEP 2: WEIGHTS  ─────────────────────────
if active_step == 2:
    if "persona" not in st.session_state:
        st.warning("Please complete Preferences first.")
    else:
        st.markdown('<div class="section-title">Tune your weights</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Think of this as a **sensitivity** step: these sliders represent how much each driver matters. They always add up to **100%**. Move any one—others will rebalance automatically.")

        if "weights_pct" not in st.session_state:
            seed = init_weights_from_persona(st.session_state["persona"])
            st.session_state["weights_pct"] = seed.copy()
            for b in BUCKET_ORDER:
                st.session_state[f"w_{b}"] = seed[b]

        cL, cR = st.columns(2)
        for i, b in enumerate(BUCKET_ORDER):
            label = f"{b} — {BUCKET_DESCRIPTIONS[b]}"
            col = cL if i % 2 == 0 else cR
            with col:
                st.slider(
                    label, min_value=0.0, max_value=100.0, step=0.1,
                    key=f"w_{b}", on_change=rebalance_on_change, args=(b,)
                )

        st.session_state["weights_pct"] = {b: float(st.session_state[f"w_{b}"]) for b in BUCKET_ORDER}
        total_now = sum(st.session_state["weights_pct"].values())
        st.caption(f"Total: **{total_now:.2f}%**")
        st.session_state["weights_vec"] = {b: round(st.session_state["weights_pct"][b]/100.0, 6) for b in BUCKET_ORDER}

        left, right = st.columns([1,1])
        with left:
            if st.button("← Back to Preferences", use_container_width=True):
                go_to(1)
        with right:
            if st.button("Apply & see Market View →", type="primary", use_container_width=True):
                go_to(3)

        st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────  STEP 3: MARKET VIEW  ─────────────────────
if active_step == 3:
    if "weights_vec" not in st.session_state:
        st.warning("Please complete Weights first.")
    else:
        st.markdown('<div class="section-title">Market snapshot (model output, no preferences)</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        df_ind = load_industry_table_from_default()
        if df_ind is None:
            st.error(
                "Couldn’t find your model output at "
                f"`{DEFAULT_DATA_PATH}`.\n\n"
                "Please place a CSV/XLSX there (columns: Industry, Return, Profit, Risk, Liquidity, Efficiency, Future) "
                "or change `DEFAULT_DATA_PATH` at the top of this file."
            )
        else:
            st.caption("We first show an equal-weight view to explain the **current market state** before blending your preferences.")
            st.caption("**Method note**: This snapshot is a **Random-Forest regression** trained on **Wharton Research Data Services (WRDS) Financial Ratios Suite** features (valuation, profitability, liquidity, leverage, efficiency), normalized to a 1–5 scale per driver and summed to a **1–30 total**.")
            max_show = min(30, len(df_ind))
            top_n = st.slider("How many industries to display?", 5, max_show, min(12, max_show), step=1)
            equal_weight_chart(df_ind, top_n=top_n)
            st.session_state["industry_table"] = df_ind.copy()

            navL, navR = st.columns([1,1])
            with navL:
                if st.button("← Back to Weights", use_container_width=True):
                    go_to(2)
            with navR:
                if st.button("Blend with my weights →", type="primary", use_container_width=True):
                    go_to(4)

        st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────  STEP 4: TOP PICKS  ───────────────────────
if active_step == 4:
    missing = [k for k in ("weights_vec","industry_table") if k not in st.session_state]
    if missing:
        st.warning("Finish the earlier steps first.")
    else:
        st.markdown('<div class="section-title">Your top picks for next month</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        df_ind = st.session_state["industry_table"]
        wvec = st.session_state["weights_vec"]

        ranked, contrib = ahp_rank(df_ind, wvec)

        st.markdown("**Top 5 industries (blending your preferences with model scores)**")
        show_cols = ["Industry","Priority score"] + BUCKET_ORDER
        st.dataframe(ranked.loc[:4, show_cols], use_container_width=True, hide_index=True)

        # WHY THESE? — Pie chart per selected industry (labels removed; tooltips only)
        st.markdown("**Why these? (what drove the score)**")
        st.caption("Pick an industry to see how each driver contributed to its Priority score. (Your driver weights × our model scores.)")
        choices = ranked["Industry"].tolist()
        pick = st.selectbox("Industry:", choices[:15] if len(choices) > 15 else choices, index=0)

        contrib_row = contrib[contrib["Industry"] == pick][BUCKET_ORDER].iloc[0]
        pie_df = pd.DataFrame({
            "Criterion": BUCKET_ORDER,
            "Contribution": [float(contrib_row[c]) for c in BUCKET_ORDER]
        })
        pie_df["Pct"] = pie_df["Contribution"] / pie_df["Contribution"].sum()

        pie_chart = (
            alt.Chart(pie_df)
            .mark_arc(outerRadius=130)
            .encode(
                theta="Pct:Q",
                color=alt.Color("Criterion:N",
                                legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=list(CRITERIA_COLORS.keys()),
                                                range=list(CRITERIA_COLORS.values()))),
                tooltip=[alt.Tooltip("Criterion:N"),
                         alt.Tooltip("Contribution:Q", title="Weighted points", format=".3f"),
                         alt.Tooltip("Pct:Q", title="Share", format=".1%")]
            )
            .properties(height=360)
        )
        st.altair_chart(pie_chart, use_container_width=False)

        with st.expander("View all industries"):
            st.dataframe(ranked[show_cols], use_container_width=True, hide_index=True)
            csv = ranked[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button("Download full ranking (CSV)", data=csv, file_name="industry_rankings_ahp.csv")

        # AHP consistency in plain English
        lam_max, CI, CR = implied_consistency(wvec)
        st.markdown(
            f"<div class='small'>AHP check: λ<sub>max</sub>={lam_max:.2f}, CI={CI:.4f}, CR={CR:.4f}. "
            "Because your sliders directly translate to consistent pairwise ratios (each weight compared to the others), "
            "the implied matrix is **perfectly consistent** (CR≈0). If CR were > 0.10, it would suggest conflicting preferences "
            "and we’d recommend nudging the sliders until CR drops.</div>",
            unsafe_allow_html=True
        )

        navL, navR = st.columns([1,1])
        with navL:
            if st.button("← Back to Market View", use_container_width=True):
                go_to(3)
        with navR:
            if st.button("Restart", use_container_width=True):
                for k in list(st.session_state.keys()):
                    if k.startswith("w_") or k in ("persona","weights_pct","weights_vec","industry_table","q2_locked","q2_order"):
                        del st.session_state[k]
                go_to(1)

        st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────  FOOTER CREDIT  ───────────────────────────
if active_step != 1:
    st.markdown('<footer>Created by <b>Dhruvaraj Kudli</b>; <b>Aaryan Kharambale</b> & <b>Naveen Akoju</b> </footer>', unsafe_allow_html=True)
