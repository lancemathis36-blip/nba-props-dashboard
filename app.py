import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

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
        # Load credentials from Streamlit secrets
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

@st.cache_data(ttl=300)  # Refresh every 5 min
def load_todays_picks():
    """Load today's elite/high picks with run metadata"""
    query = f"""
    WITH latest_run AS (
        SELECT run_id, run_type, as_of_ts
        FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
        WHERE DATE(game_date) = CURRENT_DATE('{BQ_TZ}')
        ORDER BY as_of_ts DESC
        LIMIT 1
    )
    SELECT 
        p.run_id,
        p.run_type,
        p.as_of_ts,
        p.player_name,
        p.team_name,
        p.market,
        p.side,
        p.line_use,
        p.pred_use,
        p.odds_snapshot_time,
        ROUND(p.pred_use / NULLIF(p.line_use, 0), 3) AS ratio,
        p.PRED_minutes,
        p.confidence,
        p.bet_units,
        p.grade,
        p.actual_outcome,
        p.hit_flag,
        p.roi,
        p.game_id,
        p.player_id
    FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded` p
    INNER JOIN latest_run lr ON p.run_id = lr.run_id
    WHERE p.confidence IN ('elite', 'high')
    ORDER BY p.bet_units DESC, ratio ASC
    """
    return bq_client.query(query).to_dataframe()

@st.cache_data(ttl=300)
def load_pick_explanations(game_id, player_id, market, run_id):
    """
    Load SHAP explanations for a specific pick.
    Include run_id to handle multiple runs per day (AM/PM).
    """
    query = f"""
    SELECT 
        summary,
        factor_1_explanation,
        factor_1_impact,
        factor_2_explanation,
        factor_2_impact,
        factor_3_explanation,
        factor_3_impact,
        full_explanation_json
    FROM `{PROJECT_ID}.{BQ_DATASET}.pick_explanations`
    WHERE game_date = CURRENT_DATE('{BQ_TZ}')
      AND game_id = {game_id}
      AND player_id = {player_id}
      AND market = '{market}'
      AND run_id = '{run_id}'
    LIMIT 1
    """
    try:
        df = bq_client.query(query).to_dataframe()
        return df.iloc[0] if not df.empty else None
    except Exception as e:
        st.error(f"Error loading explanation: {e}")
        return None

@st.cache_data(ttl=3600)
def load_daily_performance(days):
    """Load daily performance summary"""
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
    """Load all picks for a specific date"""
    query = f"""
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
    FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
    WHERE DATE(game_date) = DATE('{selected_date}')
      AND confidence IN ('elite', 'high')
    ORDER BY bet_units DESC, ratio ASC
    """
    return bq_client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_market_performance(days):
    """Load performance by market and confidence"""
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
    """Load performance by run type (AM vs PM)"""
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
    """Load overall summary statistics"""
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
# HELPER FUNCTIONS
# ============================================================

def display_shap_explanation(explanation):
    """Display SHAP explanation in a nice format"""
    if explanation is None:
        st.info("💡 No explanation available for this pick yet")
        st.caption("SHAP explanations are generated when you run the prediction script. If you haven't run the updated script yet, older picks won't have explanations.")
        return
    
    st.markdown("### 🔍 Why This Pick?")
    
    # Summary with error handling
    if pd.notna(explanation.get('summary')):
        st.markdown(f"**{explanation['summary']}**")
    else:
        st.warning("Summary not available")
    
    st.markdown("---")
    
    # Top 3 factors with impact visualization
    factors = [
        (explanation.get('factor_1_explanation'), explanation.get('factor_1_impact')),
        (explanation.get('factor_2_explanation'), explanation.get('factor_2_impact')),
        (explanation.get('factor_3_explanation'), explanation.get('factor_3_impact')),
    ]
    
    has_factors = False
    for i, (exp, impact) in enumerate(factors, 1):
        if pd.notna(exp) and pd.notna(impact):
            has_factors = True
            # Color code by impact direction
            color = "green" if impact > 0 else "red"
            impact_text = f"{impact:+.2f}"
            
            # Impact bar visualization
            abs_impact = abs(impact)
            max_width = 100
            bar_width = min(abs_impact / 2 * max_width, max_width)  # Scale for visualization
            
            st.markdown(f"**Factor {i}:** {exp}")
            st.markdown(
                f'<div style="background-color: {color}; width: {bar_width}%; '
                f'height: 20px; border-radius: 5px; display: inline-block;"></div> '
                f'<span style="margin-left: 10px; font-weight: bold;">{impact_text}</span>',
                unsafe_allow_html=True
            )
            st.markdown("")  # Spacing
    
    if not has_factors:
        st.warning("Factor details not available for this pick")

def format_picks_table(df):
    """Format picks dataframe for display"""
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
    
    # Status with actual outcome if available
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
    """Color code rows by confidence level"""
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

    # ---------------- SIDEBAR ----------------
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
            
            # ROI calculation
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

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📋 Today's Picks", "📈 Performance Trends", "🎯 Market Breakdown"])

    # ================= TAB 1: TODAY'S PICKS =================
    with tab1:
        st.header("Today's Picks")
        
        today = load_todays_picks()

        if today.empty:
            st.info("🕐 No picks available yet. Check back closer to game time!")
        else:
            # Run metadata
            if 'run_type' in today.columns and pd.notna(today['run_type'].iloc[0]):
                run_type = today['run_type'].iloc[0]
                run_time = pd.to_datetime(today['as_of_ts'].iloc[0])
                st.info(f"📊 **{run_type} Run** | Generated at {run_time.strftime('%I:%M %p ET')}")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            
            elite_count = len(today[today['confidence'] == 'elite'])
            high_count = len(today[today['confidence'] == 'high'])
            total_units = today['bet_units'].sum()
            graded = today[today['actual_outcome'].notna()]
            
            col1.metric("Elite (3u)", elite_count)
            col2.metric("High (2u)", high_count)
            col3.metric("Total Units", f"{total_units:.0f}u")
            
            if not graded.empty:
                live_wr = graded['hit_flag'].mean()
                col4.metric("Live Win Rate", f"{live_wr:.1%}")
            else:
                col4.metric("Status", "⏳ Pending")
            
            st.markdown("---")
            
            # Format and display picks table
            display = format_picks_table(today)
            
            show_cols = ["Player", "Team", "Market", "Side", "Line", "Pred", 
                        "Ratio", "Mins", "Conf", "Units", "Status"]
            
            st.dataframe(
                display[show_cols].style.apply(lambda row: highlight_confidence(row, today), axis=1),
                width="stretch",
                hide_index=True,
                height=400
            )
            
            # SHAP Explanations Section
            st.markdown("---")
            st.subheader("🧠 Pick Explanations (SHAP Analysis)")
            st.caption("Select a pick to see why the model made this prediction")
            
            # Create selection dropdown
            pick_options = []
            for idx, row in today.iterrows():
                label = f"{row['player_name']} - {row['market'].upper()} {row['side'].upper()} {row['line_use']}"
                pick_options.append((label, idx))
            
            if pick_options:
                selected_label = st.selectbox(
                    "Choose a pick to explain:",
                    options=[label for label, _ in pick_options],
                    index=0,
                    key="pick_selector"
                )
                
                # Get the selected pick's data
                selected_idx = [idx for label, idx in pick_options if label == selected_label][0]
                selected_pick = today.loc[selected_idx]
                
                # Load and display explanation with proper error handling
                try:
                    with st.spinner("Loading explanation..."):
                        explanation = load_pick_explanations(
                            int(selected_pick['game_id']),
                            int(selected_pick['player_id']),
                            str(selected_pick['market']),
                            str(selected_pick['run_id'])
                        )
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            display_shap_explanation(explanation)
                        
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
                except Exception as e:
                    st.error(f"⚠️ Error loading explanation: {str(e)}")
                    st.info("This pick may not have SHAP data yet. Run the prediction script with SHAP enabled.")
            
            # Download button
            st.markdown("---")
            csv = today.to_csv(index=False)
            st.download_button(
                "📥 Download Today's Picks (CSV)",
                csv,
                f"nba_picks_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

    # ================= TAB 2: PERFORMANCE TRENDS =================
    with tab2:
        st.header(f"Performance Trends (Last {lookback} Days)")
        
        daily = load_daily_performance(lookback)

        if daily.empty:
            st.warning("No historical data available for selected period")
        else:
            # Calculate cumulative
            daily = daily.sort_values('date')
            daily["cumulative_profit"] = daily["profit_units"].cumsum()
            
            # Overall metrics
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

            # Win Rate Trend
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

            # Cumulative Profit
            with col2:
                fig_cp = px.line(
                    daily,
                    x="date",
                    y="cumulative_profit",
                    markers=True,
                    title="Cumulative Profit"
                )
                fig_cp.add_hline(y=0, line_dash="dash", line_color="gray")
                
                # Color based on profit/loss
                colors = ['green' if x > 0 else 'red' for x in daily['cumulative_profit']]
                fig_cp.update_traces(line_color='#4ECDC4', marker=dict(color=colors))
                fig_cp.update_layout(height=400)
                st.plotly_chart(fig_cp, width="stretch")
            
            # Interactive Daily Breakdown
            st.markdown("---")
            st.subheader("📅 Daily Breakdown - Click to View Picks")
            
            # Sort by date descending for the dropdown
            daily_sorted = daily.sort_values('date', ascending=False)
            
            # Ensure date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(daily_sorted['date']):
                daily_sorted['date'] = pd.to_datetime(daily_sorted['date'])
            
            # Create date selection
            selected_date = st.selectbox(
                "Select a date to view picks:",
                options=daily_sorted['date'].dt.strftime('%Y-%m-%d').tolist(),
                format_func=lambda x: pd.to_datetime(x).strftime('%a, %b %d, %Y')
            )
            
            if selected_date:
                # Load picks for selected date
                date_picks = load_picks_for_date(selected_date)
                
                if not date_picks.empty:
                    # Show daily summary
                    date_stats = daily_sorted[daily_sorted['date'] == pd.to_datetime(selected_date)].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Picks", int(date_stats['picks']))
                    col2.metric("Wins", int(date_stats['wins']))
                    col3.metric("Win Rate", f"{date_stats['win_rate']:.1%}")
                    col4.metric("Profit", f"{date_stats['profit_units']:+.1f}u")
                    
                    # Show run type info if available
                    if 'run_type' in date_picks.columns and pd.notna(date_picks['run_type'].iloc[0]):
                        run_types = date_picks['run_type'].value_counts()
                        st.info(f"📊 Runs: {', '.join([f'{rt} ({cnt})' for rt, cnt in run_types.items()])}")
                    
                    st.markdown("---")
                    
                    # Display picks table
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
                    
                    # Download button for this date
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
            
            # Summary table
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

    # ================= TAB 3: MARKET BREAKDOWN =================
    with tab3:
        st.header("Market Performance Analysis")
        
        market = load_market_performance(lookback)

        if market.empty:
            st.warning("No market data available")
        else:
            # Aggregate by market
            market_agg = market.groupby('market').agg({
                'picks': 'sum',
                'wins': 'sum',
                'profit_units': 'sum'
            }).reset_index()
            market_agg['win_rate'] = market_agg['wins'] / market_agg['picks']
            market_agg = market_agg.sort_values('profit_units', ascending=False)

            # Market profit chart
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
            
            # Detailed breakdown by confidence
            st.markdown("---")
            st.subheader("Detailed Market Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Elite picks
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
                # High picks
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
