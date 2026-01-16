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
        grade,
        actual_outcome,
        hit_flag,
        roi
    FROM `{PROJECT_ID}.{BQ_DATASET}.picks_fact_over_under_graded`
    WHERE DATE(game_date) = CURRENT_DATE('{BQ_TZ}')
      AND confidence IN ('elite', 'high')
    ORDER BY bet_units DESC, ratio ASC
    """
    return bq_client.query(query).to_dataframe()

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
    ORDER BY date
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
# APP
# ============================================================

def main():
    st.title("🏀 NBA Props Dashboard")
    st.caption("Elite & High Confidence Picks Only")

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

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📋 Today's Picks", "📈 Performance Trends", "🎯 Market Breakdown"])

    # ================= TAB 1: TODAY'S PICKS =================
    with tab1:
        st.header("Today's Picks")
        
        today = load_todays_picks()

        if today.empty:
            st.info("🕐 No picks available yet. Check back closer to game time!")
        else:
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
            
            # Format and display
            display = today.copy()
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
            display["Status"] = display.apply(
                lambda r: "✅ WIN" if r.hit_flag == 1 
                         else "❌ LOSS" if r.hit_flag == 0 
                         else "⏳ Pending",
                axis=1
            )
            
            # Color code by confidence
            def highlight_confidence(row):
                if row['confidence'] == 'elite':
                    return ['background-color: rgba(255, 107, 107, 0.2)'] * len(row)
                elif row['confidence'] == 'high':
                    return ['background-color: rgba(78, 205, 196, 0.2)'] * len(row)
                return [''] * len(row)
            
            show_cols = ["Player", "Team", "Market", "Side", "Line", "Pred", 
                        "Ratio", "Mins", "Conf", "Units", "Status"]
            
            st.dataframe(
                display[show_cols].style.apply(highlight_confidence, axis=1),
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Download button
            csv = today.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
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
                st.plotly_chart(fig_wr, use_container_width=True)

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
                st.plotly_chart(fig_cp, use_container_width=True)
            
            # Data table
            st.markdown("---")
            st.subheader("Daily Breakdown")
            
            display_daily = daily.copy()
            display_daily['win_rate'] = display_daily['win_rate'].apply(lambda x: f"{x:.1%}")
            display_daily['profit_units'] = display_daily['profit_units'].round(1)
            display_daily['cumulative_profit'] = display_daily['cumulative_profit'].round(1)
            
            st.dataframe(
                display_daily.sort_values('date', ascending=False),
                use_container_width=True,
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
            st.plotly_chart(fig_mkt, use_container_width=True)
            
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
                    st.dataframe(elite_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No elite picks in this period")
            
            with col2:
                # High picks
                st.markdown("**High Confidence (2u)**")
                high = market[market['confidence'] == 'high'].copy()
                if not high.empty:
                    high_display = high[['market', 'picks', 'wins', 'win_rate', 'profit_units']].copy()
                    high_display['market'] = high_display['market'].str.upper()
                    st.dataframe(high_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No high picks in this period")

if __name__ == "__main__":
    main()
