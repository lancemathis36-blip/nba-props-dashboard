import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import re
import warnings

# Suppress BigQuery Storage warnings
warnings.filterwarnings('ignore', message='BigQuery Storage module not found')

# ============================================================
# CONFIG
# ============================================================
PROJECT_ID = "prop-v4"
BQ_DATASET = "NBA_Data"
BQ_TZ = "America/New_York"

st.set_page_config(
    page_title="NBA Props Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# BIGQUERY CLIENT (STREAMLIT SECRETS)
# ============================================================
@st.cache_resource
def get_bq_client():
    """
    Initialize BigQuery client using Streamlit secrets.
    Secrets should be configured in .streamlit/secrets.toml locally
    or in Streamlit Cloud dashboard for production.
    """
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return bigquery.Client(credentials=credentials, project=PROJECT_ID)
    except Exception as e:
        st.error(f"Failed to initialize BigQuery client: {e}")
        st.info("Please configure your GCP service account in Streamlit secrets.")
        st.stop()

bq_client = get_bq_client()

# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data(ttl=900)  # 15 min - longer cache to reduce rerun query load
def load_todays_picks_with_explanations():
    """
    Single query that loads today's picks WITH explanations joined.
    Deduplicates to handle any duplicate data in the table.
    """
    query = f"""
    WITH latest_runs AS (
        SELECT
            run_type,
            ARRAY_AGG(STRUCT(run_id, as_of_ts) ORDER BY as_of_ts DESC LIMIT 1)[OFFSET(0)] AS r
        FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
        WHERE DATE(game_date) = CURRENT_DATE('{BQ_TZ}')
          AND run_type IN ('AM', 'PM')
        GROUP BY run_type
    ),
    deduped_picks AS (
        SELECT 
            p.*,
            ROW_NUMBER() OVER (
                PARTITION BY 
                    p.run_type,
                    CAST(p.run_id AS STRING),
                    p.game_id,
                    p.player_id,
                    p.market,
                    p.side
                ORDER BY COALESCE(p.as_of_ts, TIMESTAMP('1970-01-01')) ASC
            ) as rn
        FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded` p
        INNER JOIN latest_runs lr
            ON p.run_type = lr.run_type
            AND CAST(p.run_id AS STRING) = CAST(lr.r.run_id AS STRING)
        WHERE p.confidence IN ('elite', 'high')
    )
    SELECT
        p.run_id, p.run_type, p.as_of_ts,
        p.player_name, p.team_name, p.market, p.side,
        p.line_use, p.pred_use, p.odds_snapshot_time,
        ROUND(p.pred_use / NULLIF(p.line_use, 0), 3) AS ratio,
        p.PRED_minutes, p.confidence, p.bet_units, p.grade,
        p.actual_outcome, p.hit_flag, p.roi, p.game_id, p.player_id,
        e.summary,
        e.factor_1_explanation, e.factor_1_impact,
        e.factor_2_explanation, e.factor_2_impact,
        e.factor_3_explanation, e.factor_3_impact,
        e.full_explanation_json
    FROM deduped_picks p
    LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.pick_explanations` e
        ON e.game_date = CURRENT_DATE('{BQ_TZ}')
        AND CAST(e.run_id AS STRING) = CAST(p.run_id AS STRING)
        AND e.game_id = p.game_id
        AND e.player_id = p.player_id
        AND e.market = p.market
    WHERE p.rn = 1
    ORDER BY p.run_type DESC, p.bet_units DESC, ratio ASC
    """
    df = bq_client.query(query).to_dataframe()
    df = df.drop_duplicates(subset=['run_id', 'game_id', 'player_id', 'market', 'side']).reset_index(drop=True)
    return df

@st.cache_data(ttl=3600)
def load_daily_performance(days):
    query = f"""
    SELECT
        DATE(game_date) AS date,
        COUNT(*) AS picks,
        SUM(hit_flag) AS wins,
        ROUND(AVG(hit_flag), 3) AS win_rate,
        SUM(
            CASE confidence
                WHEN 'elite' THEN 3 * roi
                WHEN 'high' THEN 2 * roi
            END
        ) AS profit_units
    FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
    WHERE DATE(game_date) >= DATE_SUB(CURRENT_DATE('{BQ_TZ}'), INTERVAL {days} DAY)
      AND actual_outcome IS NOT NULL
      AND confidence IN ('elite','high')
    GROUP BY date
    ORDER BY date DESC
    """
    return bq_client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_picks_for_date(selected_date):
    """Load all picks for a specific date, deduplicated"""
    query = f"""
    WITH deduped AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY 
                    DATE(game_date),
                    game_id,
                    player_id,
                    market,
                    side
                ORDER BY COALESCE(as_of_ts, TIMESTAMP('1970-01-01')) ASC
            ) as rn
        FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
        WHERE DATE(game_date) = DATE('{selected_date}')
          AND confidence IN ('elite', 'high')
    )
    SELECT 
        player_name,
        team_name,
        market,
        side,
        line_use,
        pred_use,
        ROUND(pred_use / NULLIF(line_use, 0), 3) AS ratio,
        PRED_minutes,
        confidence,
        bet_units,
        actual_outcome,
        hit_flag,
        roi,
        run_type,
        as_of_ts
    FROM deduped
    WHERE rn = 1
    ORDER BY bet_units DESC, ratio ASC
    """
    return bq_client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_market_performance(days):
    query = f"""
    SELECT
        market,
        confidence,
        COUNT(*) AS picks,
        SUM(hit_flag) AS wins,
        ROUND(AVG(hit_flag), 3) AS win_rate,
        ROUND(AVG(roi), 3) AS avg_roi,
        SUM(
            CASE confidence
                WHEN 'elite' THEN 3 * roi
                WHEN 'high' THEN 2 * roi
            END
        ) AS profit_units
    FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
    WHERE DATE(game_date) >= DATE_SUB(CURRENT_DATE('{BQ_TZ}'), INTERVAL {days} DAY)
      AND actual_outcome IS NOT NULL
      AND confidence IN ('elite','high')
    GROUP BY market, confidence
    """
    return bq_client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_run_type_performance(days):
    query = f"""
    SELECT
        run_type,
        COUNT(*) AS picks,
        SUM(hit_flag) AS wins,
        ROUND(AVG(hit_flag), 3) AS win_rate,
        SUM(
            CASE confidence
                WHEN 'elite' THEN 3 * roi
                WHEN 'high' THEN 2 * roi
            END
        ) AS profit_units
    FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
    WHERE DATE(game_date) >= DATE_SUB(CURRENT_DATE('{BQ_TZ}'), INTERVAL {days} DAY)
      AND actual_outcome IS NOT NULL
      AND confidence IN ('elite','high')
      AND run_type IS NOT NULL
    GROUP BY run_type
    ORDER BY run_type
    """
    return bq_client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_summary_stats(days):
    query = f"""
    SELECT
        COUNT(*) AS total_picks,
        SUM(hit_flag) AS total_wins,
        ROUND(AVG(hit_flag), 3) AS overall_win_rate,
        SUM(
            CASE confidence
                WHEN 'elite' THEN 3 * roi
                WHEN 'high' THEN 2 * roi
            END
        ) AS total_profit
    FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
    WHERE DATE(game_date) >= DATE_SUB(CURRENT_DATE('{BQ_TZ}'), INTERVAL {days} DAY)
      AND actual_outcome IS NOT NULL
      AND confidence IN ('elite','high')
    """
    df = bq_client.query(query).to_dataframe()
    return df.iloc[0] if not df.empty else None

# ============================================================
# SHAP EXPLANATION HELPERS
# ============================================================
FEATURE_LABELS = {
    "num__usage_pm_l10": "Recent usage",
    "num__pts_pm_l10": "Recent scoring rate",
    "num__ast_pm_l10": "Recent assist rate",
    "num__reb_pm_l10": "Recent rebound rate",
    "num__mins_avg_l10": "Recent minutes",
    "num__PRED_minutes": "Projected minutes tonight",
    "num__PRED_team_total": "Projected team score",
    "num__Opp_L10_Pace": "Opponent pace",
    "num__Opp_L10_Off_Eff": "Opponent offensive efficiency",
    "num__Opp_L10_Def_Eff": "Opponent defensive efficiency",
    "num__Opp_Def_Reb_ZScore": "Opponent rebounding strength",
    "num__days_of_rest": "Days of rest",
    "num__vacated_mins_l10": "Vacated minutes (injuries)",
    "num__vacated_usage_l10": "Vacated usage (injuries)",
    "num__vacated_pts_l10": "Vacated scoring (injuries)",
    "cat__archetype_Connector / Point Wing": "Playmaker archetype",
}

def parse_factor(exp: str):
    if not exp or not isinstance(exp, str):
        return None
    left = exp.split("→")[0].strip() if "→" in exp else exp
    m = re.match(r"(.+?)\s*=\s*([-\d\.]+)", left)
    feature = m.group(1).strip() if m else None
    value = float(m.group(2)) if (m and m.group(2) is not None) else None
    direction_word = None
    if "→" in exp:
        right = exp.split("→", 1)[1].strip().lower()
        if right.startswith("increases"):
            direction_word = "up"
        elif right.startswith("decreases"):
            direction_word = "down"
    m2 = re.search(r"\(([-+]\d+(\.\d+)?)\)", exp)
    impact = float(m2.group(1)) if m2 else None
    return feature, value, direction_word, impact

def format_feature_value(feature: str, value: float):
    if value is None:
        return None
    if feature in ("num__mins_avg_l10", "num__PRED_minutes"):
        return f"{value:.1f} min"
    if feature.endswith("_pm_l10"):
        return f"{value:.3f}/min"
    if feature == "num__days_of_rest":
        return f"{int(round(value))} days"
    if feature.startswith("cat__"):
        return "Yes" if float(value) == 1.0 else "No"
    if "team_total" in feature.lower():
        return f"{value:.0f} pts"
    if "pace" in feature.lower():
        return f"{value:.1f}"
    return f"{value:.2f}"

def _safe_float_from_disp(value_disp: str):
    if value_disp is None:
        return None
    m = re.search(r"[-+]?\d+(\.\d+)?", str(value_disp))
    return float(m.group(0)) if m else None

def human_sentence(market: str, feature: str, value_disp: str, impact: float):
    label = FEATURE_LABELS.get(feature, feature.replace("num__", "").replace("_", " "))
    market_word = market.upper()
    push = "boosting" if impact > 0 else "pulling"
    direction = "up" if impact > 0 else "down"
    if feature in ("num__mins_avg_l10", "num__PRED_minutes"):
        return f"**{label}** at **{value_disp}** is {push} the **{market_word}** projection {direction} (~{abs(impact):.2f})."
    if feature == "num__usage_pm_l10":
        if market.lower() in ("ast", "pra"):
            return f"**{label}** (**{value_disp}**) suggests fewer playmaking chances, {push} **{market_word}** {direction} (~{abs(impact):.2f})."
        return f"**{label}** (**{value_disp}**) signals shot volume, {push} **{market_word}** {direction} (~{abs(impact):.2f})."
    if feature == "num__pts_pm_l10" and market.lower() == "ast":
        return f"**{label}** (**{value_disp}**) shifts role/shot profile, {push} assists {direction} (~{abs(impact):.2f})."
    if "team_total" in feature.lower():
        tt = _safe_float_from_disp(value_disp)
        tempo = "high-scoring" if (tt is not None and tt > 110) else "lower-scoring"
        return f"**{label}** (**{value_disp}**, {tempo} environment) is {push} **{market_word}** {direction} (~{abs(impact):.2f})."
    if "pace" in feature.lower():
        pace = _safe_float_from_disp(value_disp)
        speed = "fast" if (pace is not None and pace > 100) else "slow"
        return f"**{label}** (**{value_disp}**, {speed} opponent) is {push} **{market_word}** {direction} (~{abs(impact):.2f})."
    if feature.startswith("cat__archetype"):
        on_off = "present" if value_disp == "Yes" else "not present"
        return f"**{label}** is **{on_off}**, {push} **{market_word}** {direction} (~{abs(impact):.2f})."
    return f"**{label}** (**{value_disp}**) is {push} **{market_word}** {direction} (~{abs(impact):.2f})."

def display_shap_explanation(pick_row):
    if pd.isna(pick_row.get("summary")):
        st.info("💡 No explanation available for this pick yet")
        st.caption("SHAP explanations are generated when you run the prediction script with SHAP enabled.")
        return
    player = pick_row["player_name"]
    market = str(pick_row["market"]).upper()
    side = str(pick_row["side"]).upper()
    line = float(pick_row["line_use"])
    pred = float(pick_row["pred_use"])
    diff = abs(pred - line)
    below_above = "below" if pred < line else "above"
    st.markdown("### 🔍 Why this pick?")
    conf = str(pick_row.get("confidence", "")).lower()
    conf_color = "#FF6B6B" if conf == "elite" else "#4ECDC4"
    st.markdown(
        f"""
        **{player}** is projected at **{pred:.1f} {market}**, about **{diff:.1f} {market} {below_above}** the **{line:.1f}** line → leaning **{side}**.
        <span style='background-color:{conf_color}; color:white; padding:2px 8px; border-radius:4px; font-size:0.85em; margin-left:8px;'>
            {conf.upper()}
        </span>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### 🧠 What's driving the projection")
    raw_factors = [
        pick_row.get("factor_1_explanation"),
        pick_row.get("factor_2_explanation"),
        pick_row.get("factor_3_explanation"),
    ]
    impacts = []
    for exp in raw_factors:
        parsed = parse_factor(exp)
        if parsed and parsed[3] is not None:
            impacts.append(abs(parsed[3]))
    scale = max(impacts) if impacts else 1.0
    any_shown = False
    for i, exp in enumerate(raw_factors, 1):
        parsed = parse_factor(exp)
        if not parsed:
            continue
        feature, value, _, impact = parsed
        if feature is None or impact is None:
            continue
        value_disp = format_feature_value(feature, value)
        sentence = human_sentence(market, feature, value_disp, impact)
        emoji = "📈" if impact > 0 else "📉"
        bar_width = (abs(impact) / scale) * 100
        color = "green" if impact > 0 else "red"
        st.markdown(f"{emoji} **{i}.** {sentence}")
        st.markdown(
            f"<div style='background-color:{color}; width:{bar_width:.0f}%; height:14px; border-radius:6px; margin:4px 0;'></div>",
            unsafe_allow_html=True
        )
        st.caption(f"Impact on {market}: {impact:+.2f}")
        st.markdown("")
        any_shown = True
    if not any_shown:
        st.warning("Factor details not available for this pick.")
    st.markdown("---")
    st.caption("💡 **Green** pushes the prediction up • **Red** pushes it down • Wider bars = stronger effect")

# ============================================================
# DISPLAY HELPERS
# ============================================================
def format_picks_table(df):
    display = df.copy()
    display["Player"] = display["player_name"]
    display["Team"] = display["team_name"]
    display["Market"] = display["market"].str.upper()
    display["Side"] = display["side"].str.upper()
    display["Line"] = display["line_use"].round(1)
    display["Pred"] = display["pred_use"].round(1)
    display["Ratio"] = display["ratio"].round(3)
    display["Mins"] = display["PRED_minutes"].round(0).astype(int)
    display["Conf"] = display["confidence"].str.upper()
    display["Units"] = display["bet_units"]
    def get_status(row):
        if pd.notna(row.get('hit_flag')):
            if row['hit_flag'] == 1:
                return f"✅ WIN ({row.get('actual_outcome', 'N/A')})"
            else:
                return f"❌ LOSS ({row.get('actual_outcome', 'N/A')})"
        return "⏳ Pending"
    display["Status"] = display.apply(get_status, axis=1)
    return display

def highlight_confidence(row, original_df):
    idx = row.name
    conf = original_df.loc[idx, 'confidence']
    if conf == 'elite':
        return ['background-color: rgba(255, 107, 107, 0.2)'] * len(row)
    elif conf == 'high':
        return ['background-color: rgba(78, 205, 196, 0.2)'] * len(row)
    return [''] * len(row)

# ============================================================
# APP
# ============================================================
def main():
    st.title("🏀 NBA Props Dashboard")
    st.caption("Elite & High Confidence Picks Only | Powered by XGBoost + SHAP")

    with st.sidebar:
        st.header("⚙️ Settings")
        lookback = st.slider("Lookback Days", 7, 90, 30)
        st.markdown("---")
        st.header("📊 Summary Stats")
        summary = load_summary_stats(lookback)
        if summary is not None:
            st.metric("Total Picks", f"{int(summary['total_picks']):,}")
            st.metric("Win Rate", f"{summary['overall_win_rate']:.1%}",
                     delta=f"{(summary['overall_win_rate'] - 0.524):.1%}")
            st.metric("Total Profit", f"{summary['total_profit']:+.1f}u")
            roi_pct = (summary['total_profit'] / summary['total_picks']) * 100 if summary['total_picks'] > 0 else 0
            st.metric("ROI", f"{roi_pct:+.1f}%")
        st.markdown("---")
        st.header("🔄 Run Type Performance")
        run_perf = load_run_type_performance(lookback)
        if not run_perf.empty:
            for _, row in run_perf.iterrows():
                with st.expander(f"{row['run_type']} Run"):
                    st.metric("Picks", int(row['picks']))
                    st.metric("Win Rate", f"{row['win_rate']:.1%}")
                    st.metric("Profit", f"{row['profit_units']:+.1f}u")

    tab1, tab2, tab3 = st.tabs(["📋 Today's Picks", "📈 Performance Trends", "🎯 Market Breakdown"])

    # ================= TAB 1 =================
    with tab1:
        st.header("Today's Picks")
        today = load_todays_picks_with_explanations()
        if today.empty:
            st.info("🕐 No picks available yet. Check back closer to game time!")
        else:
            run_types = today['run_type'].unique()
            if len(run_types) > 1:
                selected_run = st.radio(
                    "Select Run:",
                    options=sorted(run_types, reverse=True),
                    horizontal=True
                )
                today_filtered = today[today['run_type'] == selected_run].copy()
            else:
                today_filtered = today.copy()

            # Run metadata (timezone-aware conversion only)
            if 'run_type' in today_filtered.columns and pd.notna(today_filtered['run_type'].iloc[0]):
                run_type = today_filtered['run_type'].iloc[0]
                run_time = pd.to_datetime(today_filtered['as_of_ts'].iloc[0])
                run_time = run_time.tz_convert('America/New_York')
                st.info(f"📊 **{run_type} Run** | Generated at {run_time.strftime('%I:%M %p ET')}")

            col1, col2, col3, col4 = st.columns(4)
            elite_count = len(today_filtered[today_filtered['confidence'] == 'elite'])
            high_count = len(today_filtered[today_filtered['confidence'] == 'high'])
            total_units = today_filtered['bet_units'].sum()
            graded = today_filtered[today_filtered['actual_outcome'].notna()]
            col1.metric("Elite (3u)", elite_count)
            col2.metric("High (2u)", high_count)
            col3.metric("Total Units", f"{total_units:.0f}u")
            if not graded.empty:
                live_wr = graded['hit_flag'].mean()
                col4.metric("Live Win Rate", f"{live_wr:.1%}")
            else:
                col4.metric("Status", "⏳ Pending")

            st.markdown("---")
            display = format_picks_table(today_filtered)
            show_cols = ["Player", "Team", "Market", "Side", "Line", "Pred",
                        "Ratio", "Mins", "Conf", "Units", "Status"]
            st.dataframe(
                display[show_cols].style.apply(lambda row: highlight_confidence(row, today_filtered), axis=1),
                width="stretch",
                hide_index=True,
                height=400
            )

            st.markdown("---")
            st.subheader("🧠 Pick Explanations (SHAP Analysis)")
            st.caption("Select a pick to see why the model made this prediction")
            has_explanation = today_filtered['summary'].notna().sum()
            st.caption(f"✅ {has_explanation}/{len(today_filtered)} picks have explanations")
            pick_options = []
            for idx, row in today_filtered.iterrows():
                label = f"{row['player_name']} - {row['market'].upper()} {row['side'].upper()} {row['line_use']}"
                pick_options.append((label, idx))
            if pick_options:
                selected_label = st.selectbox(
                    "Choose a pick to explain:",
                    options=[label for label, _ in pick_options],
                    index=0,
                    key="pick_selector"
                )
                selected_idx = [idx for label, idx in pick_options if label == selected_label][0]
                selected_pick = today_filtered.loc[selected_idx]
                col1, col2 = st.columns([2, 1])
                with col1:
                    display_shap_explanation(selected_pick)
                with col2:
                    st.markdown("### 📊 Pick Details")
                    st.metric("Player", selected_pick['player_name'])
                    st.metric("Team", selected_pick['team_name'])
                    st.metric("Market", selected_pick['market'].upper())
                    st.metric("Line", f"{selected_pick['side'].upper()} {selected_pick['line_use']}")
                    st.metric("Prediction", f"{selected_pick['pred_use']:.1f}")
                    st.metric("Minutes", f"{selected_pick['PRED_minutes']:.0f}")
                    st.metric("Confidence", selected_pick['confidence'].upper())
                    if pd.notna(selected_pick.get('odds_snapshot_time')):
                        snap_time = pd.to_datetime(selected_pick['odds_snapshot_time'])
                        st.caption(f"Odds from: {snap_time.strftime('%I:%M %p ET')}")
            st.markdown("---")
            csv = today_filtered.to_csv(index=False)
            st.download_button(
                "📥 Download Today's Picks (CSV)",
                csv,
                f"nba_picks_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

    # ================= TAB 2 =================
    with tab2:
        st.header(f"Performance Trends (Last {lookback} Days)")
        daily = load_daily_performance(lookback)
        if daily.empty:
            st.warning("No historical data available for selected period")
        else:
            daily = daily.sort_values('date')
            daily["cumulative_profit"] = daily["profit_units"].cumsum()
            total_picks = daily["picks"].sum()
            total_wins = daily["wins"].sum()
            overall_wr = total_wins / total_picks if total_picks > 0 else 0
            total_profit = daily["profit_units"].sum()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Picks", f"{total_picks:,}")
            col2.metric("Win Rate", f"{overall_wr:.1%}")
            col3.metric("Total Profit", f"{total_profit:+.1f}u")
            col4.metric("ROI", f"{(total_profit/total_picks*100):+.1f}%")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                fig_wr = px.line(
                    daily,
                    x="date",
                    y="win_rate",
                    markers=True,
                    title="Daily Win Rate Trend"
                )
                fig_wr.add_hline(y=0.524, line_dash="dash", line_color="red",
                               annotation_text="Break-even (52.4%)")
                fig_wr.add_hline(y=overall_wr, line_dash="dot", line_color="green",
                               annotation_text=f"Average ({overall_wr:.1%})")
                fig_wr.update_yaxes(tickformat=".0%")
                fig_wr.update_layout(height=400)
                st.plotly_chart(fig_wr, width="stretch")
            with col2:
                fig_cp = px.line(
                    daily,
                    x="date",
                    y="cumulative_profit",
                    markers=True,
                    title="Cumulative Profit"
                )
                fig_cp.add_hline(y=0, line_dash="dash", line_color="gray")
                colors = ['green' if x > 0 else 'red' for x in daily['cumulative_profit']]
                fig_cp.update_traces(line_color='#4ECDC4', marker=dict(color=colors))
                fig_cp.update_layout(height=400)
                st.plotly_chart(fig_cp, width="stretch")
            st.markdown("---")
            st.subheader("📅 Daily Breakdown - Click to View Picks")
            daily_sorted = daily.sort_values('date', ascending=False)
            if not pd.api.types.is_datetime64_any_dtype(daily_sorted['date']):
                daily_sorted['date'] = pd.to_datetime(daily_sorted['date'])
            selected_date = st.selectbox(
                "Select a date to view picks:",
                options=daily_sorted['date'].dt.strftime('%Y-%m-%d').tolist(),
                format_func=lambda x: pd.to_datetime(x).strftime('%a, %b %d, %Y')
            )
            if selected_date:
                date_picks = load_picks_for_date(selected_date)
                if not date_picks.empty:
                    date_stats = daily_sorted[daily_sorted['date'] == pd.to_datetime(selected_date)].iloc[0]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Picks", int(date_stats['picks']))
                    col2.metric("Wins", int(date_stats['wins']))
                    col3.metric("Win Rate", f"{date_stats['win_rate']:.1%}")
                    col4.metric("Profit", f"{date_stats['profit_units']:+.1f}u")
                    if 'run_type' in date_picks.columns and pd.notna(date_picks['run_type'].iloc[0]):
                        run_types = date_picks['run_type'].value_counts()
                        st.info(f"📊 Runs: {', '.join([f'{rt} ({cnt})' for rt, cnt in run_types.items()])}")
                    st.markdown("---")
                    display = format_picks_table(date_picks)
                    show_cols = ["Player", "Team", "Market", "Side", "Line", "Pred",
                                "Ratio", "Mins", "Conf", "Units", "Status"]
                    st.dataframe(
                        display[show_cols].style.apply(
                            lambda row: highlight_confidence(row, date_picks), axis=1
                        ),
                        width="stretch",
                        hide_index=True,
                        height=400
                    )
                    csv = date_picks.to_csv(index=False)
                    st.download_button(
                        f"📥 Download {selected_date} Picks",
                        csv,
                        f"nba_picks_{selected_date}.csv",
                        "text/csv",
                        key=f"download_{selected_date}"
                    )
                else:
                    st.warning(f"No picks found for {selected_date}")
            st.markdown("---")
            st.subheader("Summary Table")
            display_daily = daily_sorted.copy()
            display_daily['date'] = display_daily['date'].dt.strftime('%Y-%m-%d')
            display_daily['win_rate'] = display_daily['win_rate'].apply(lambda x: f"{x:.1%}")
            display_daily['profit_units'] = display_daily['profit_units'].round(1)
            display_daily['cumulative_profit'] = display_daily['cumulative_profit'].round(1)
            st.dataframe(
                display_daily,
                width="stretch",
                hide_index=True
            )

    # ================= TAB 3 =================
    with tab3:
        st.header("Market Performance Analysis")
        market = load_market_performance(lookback)
        if market.empty:
            st.warning("No market data available")
        else:
            market_agg = market.groupby('market').agg({
                'picks': 'sum',
                'wins': 'sum',
                'profit_units': 'sum'
            }).reset_index()
            market_agg['win_rate'] = market_agg['wins'] / market_agg['picks']
            market_agg = market_agg.sort_values('profit_units', ascending=False)
            fig_mkt = go.Figure()
            colors = ['#4ECDC4' if x > 0 else '#FF6B6B' for x in market_agg['profit_units']]
            fig_mkt.add_trace(go.Bar(
                x=market_agg["market"].str.upper(),
                y=market_agg["profit_units"],
                text=market_agg["profit_units"].round(1),
                textposition="auto",
                marker_color=colors
            ))
            fig_mkt.update_layout(
                title="Profit by Market",
                xaxis_title="Market",
                yaxis_title="Profit (Units)",
                height=400
            )
            st.plotly_chart(fig_mkt, width="stretch")
            st.markdown("---")
            st.subheader("Detailed Market Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Elite Confidence (3u)**")
                elite = market[market['confidence'] == 'elite'].copy()
                if not elite.empty:
                    elite_display = elite[['market', 'picks', 'wins', 'win_rate', 'profit_units']].copy()
                    elite_display['market'] = elite_display['market'].str.upper()
                    elite_display['win_rate'] = elite_display['win_rate'].apply(lambda x: f"{x:.1%}")
                    elite_display['profit_units'] = elite_display['profit_units'].round(1)
                    st.dataframe(elite_display, width="stretch", hide_index=True)
                else:
                    st.info("No elite picks in this period")
            with col2:
                st.markdown("**High Confidence (2u)**")
                high = market[market['confidence'] == 'high'].copy()
                if not high.empty:
                    high_display = high[['market', 'picks', 'wins', 'win_rate', 'profit_units']].copy()
                    high_display['market'] = high_display['market'].str.upper()
                    high_display['win_rate'] = high_display['win_rate'].apply(lambda x: f"{x:.1%}")
                    high_display['profit_units'] = high_display['profit_units'].round(1)
                    st.dataframe(high_display, width="stretch", hide_index=True)
                else:
                    st.info("No high picks in this period")

if __name__ == "__main__":
    main()