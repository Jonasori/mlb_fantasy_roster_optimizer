"""
Streamlit Dashboard for MLB Fantasy Roster Optimizer.

Main entry point for the in-season dashboard.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="MLB Fantasy Roster Optimizer",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.projections = None
        st.session_state.my_roster = None
        st.session_state.opponent_rosters = None
        st.session_state.standings = None
        st.session_state.last_refresh = None
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "🏠 Overview"
    if "trade_results" not in st.session_state:
        st.session_state.trade_results = None
    if "player_values" not in st.session_state:
        st.session_state.player_values = None
    if "situation" not in st.session_state:
        st.session_state.situation = None
    if "trade_targets" not in st.session_state:
        st.session_state.trade_targets = None
    if "trade_pieces" not in st.session_state:
        st.session_state.trade_pieces = None


def load_data():
    """Load all data from database."""
    from optimizer.database import refresh_all_data

    with st.spinner("Loading data..."):
        data = refresh_all_data(skip_fantrax=True)

        st.session_state.projections = data["projections"]
        st.session_state.my_roster = data["my_roster"]
        st.session_state.opponent_rosters = data["opponent_rosters"]
        st.session_state.standings = data["standings"]
        st.session_state.data_loaded = True
        st.session_state.last_refresh = datetime.now()
        # Clear cached results when data refreshes
        st.session_state.trade_results = None
        st.session_state.player_values = None
        st.session_state.situation = None
        st.session_state.trade_targets = None
        st.session_state.trade_pieces = None

    st.success("Data loaded successfully!")


PAGES = [
    ("🏠 Overview", "Overview"),
    ("📊 My Team", "My Team"),
    ("🔄 Trades", "Trades"),
    ("🔍 Free Agents", "Free Agents"),
    ("📋 All Players", "All Players"),
]


def navigate_to(page: str):
    """Navigate to a specific page by setting session state."""
    st.session_state.nav_page = page


def main():
    """Main dashboard entry point."""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("⚾ Roster Optimizer")

        if st.button("🔄 Refresh Data", width="stretch"):
            load_data()

        if st.session_state.last_refresh:
            st.caption(
                f"Last refresh: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M')}"
            )

        st.markdown("---")
        st.markdown("#### Navigation")

        # Navigation buttons - clean and simple
        current_page = st.session_state.nav_page

        for page_key, page_label in PAGES:
            # Highlight current page
            is_current = current_page == page_key
            button_type = "primary" if is_current else "secondary"

            if st.button(
                page_key,
                key=f"nav_{page_key}",
                width="stretch",
                type=button_type,
            ):
                navigate_to(page_key)
                st.rerun()

    # Main content
    if not st.session_state.data_loaded:
        st.title("MLB Fantasy Roster Optimizer")
        st.info("Click 'Refresh Data' in the sidebar to load projections and rosters.")

        st.markdown("""
        ## Welcome!
        
        This dashboard helps you:
        - **Optimize your roster** using MILP optimization
        - **Identify trade opportunities** with probabilistic win modeling
        - **Simulate roster changes** and see their impact
        - **Track your standings** across all categories
        
        ### Getting Started
        
        1. Click **Refresh Data** to load FanGraphs projections
        2. If you have Fantrax cookies configured, roster data will also load
        3. Navigate using the sidebar to explore different features
        """)
        return

    # Route to appropriate page
    current_page = st.session_state.nav_page
    if current_page == "🏠 Overview":
        show_overview()
    elif current_page == "📊 My Team":
        show_my_team()
    elif current_page == "🔄 Trades":
        show_trades()
    elif current_page == "🔍 Free Agents":
        show_simulator()
    elif current_page == "📋 All Players":
        show_players()
    else:
        # Handle old nav state pointing to removed page
        show_overview()


def show_overview():
    """Show dashboard overview with league standings."""
    st.title("League Overview")

    from optimizer.data_loader import (
        ALL_CATEGORIES,
        NEGATIVE_CATEGORIES,
        compute_all_opponent_totals,
        compute_team_totals,
    )

    projections = st.session_state.projections
    my_roster = st.session_state.my_roster
    opponent_rosters = st.session_state.opponent_rosters

    # Quick actions at the top
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 View My Team", width="stretch"):
            navigate_to("📊 My Team")
            st.rerun()

    with col2:
        if st.button("🔍 Free Agents", width="stretch"):
            navigate_to("🔍 Free Agents")
            st.rerun()

    with col3:
        if st.button("🔄 Analyze Trades", width="stretch"):
            navigate_to("🔄 Trades")
            st.rerun()

    # Standings section
    st.divider()

    # Check if we have actual standings from Fantrax
    standings = st.session_state.standings
    has_actual_standings = standings is not None and len(standings) > 0

    if has_actual_standings:
        st.subheader("League Standings")
        as_of_date = (
            standings["as_of_date"].iloc[0]
            if "as_of_date" in standings.columns
            else "Unknown"
        )
        st.caption(f"Actual standings from Fantrax (as of {as_of_date})")

        # Format the standings for display
        display_cols = ["team_name", "overall_rank", "total_points"]
        standings_display = standings[display_cols].copy()
        standings_display = standings_display.rename(
            columns={
                "team_name": "Team",
                "overall_rank": "Rank",
                "total_points": "Total Points",
            }
        )
        standings_display = standings_display.sort_values("Rank")

        # Add indicator for my team
        from optimizer.data_loader import MY_TEAM_NAME

        standings_display[""] = standings_display["Team"].apply(
            lambda x: "👉" if x == MY_TEAM_NAME else ""
        )
        cols = standings_display.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        standings_display = standings_display[cols]

        st.dataframe(standings_display, hide_index=True, width="stretch")

        # Show my position
        my_row = standings_display[standings_display["Team"] == MY_TEAM_NAME]
        if len(my_row) > 0:
            my_rank = int(my_row["Rank"].iloc[0])
            my_points = my_row["Total Points"].iloc[0]
            if pd.isna(my_points):
                st.info(
                    "Season hasn't started yet - standings will update once games begin."
                )
            elif my_rank == 1:
                st.success(f"🏆 You're in **1st place** with {my_points} roto points!")
            elif my_rank <= 3:
                ordinal = {1: "st", 2: "nd", 3: "rd"}.get(my_rank, "th")
                st.info(
                    f"📈 You're in **{my_rank}{ordinal} place** with {my_points} roto points"
                )
            else:
                st.warning(
                    f"📊 You're in **{my_rank}th place** with {my_points} roto points"
                )
    else:
        st.subheader("League Standings")
        st.info(
            "No standings data available. Click 'Refresh Data' to fetch from Fantrax."
        )

    # Projected standings (based on projections)
    if my_roster and opponent_rosters and len(opponent_rosters) > 0:
        with st.expander("📊 Projected Standings (based on FanGraphs projections)"):
            st.caption("How teams would rank if projections played out perfectly.")

            # Compute all team totals
            my_totals = compute_team_totals(my_roster, projections)
            opponent_rosters_indexed = {
                i + 1: names for i, (team, names) in enumerate(opponent_rosters.items())
            }
            opponent_totals = compute_all_opponent_totals(
                opponent_rosters_indexed, projections
            )

            # Build team names map
            team_names_map = {"My Team": my_totals}
            for i, (team_name, _) in enumerate(opponent_rosters.items(), 1):
                team_names_map[team_name] = opponent_totals[i]

            # Compute rankings for each category
            standings_data = []
            for team_name, totals in team_names_map.items():
                team_row = {"Team": team_name, "Total Points": 0}

                for cat in ALL_CATEGORIES:
                    all_values = [(t, d[cat]) for t, d in team_names_map.items()]
                    if cat in NEGATIVE_CATEGORIES:
                        all_values.sort(key=lambda x: x[1])
                    else:
                        all_values.sort(key=lambda x: x[1], reverse=True)

                    rank = next(
                        i + 1 for i, (t, _) in enumerate(all_values) if t == team_name
                    )
                    points = len(team_names_map) - rank + 1
                    team_row[cat] = rank
                    team_row["Total Points"] += points

                standings_data.append(team_row)

            proj_standings_df = pd.DataFrame(standings_data)
            proj_standings_df = proj_standings_df.sort_values(
                "Total Points", ascending=False
            )
            proj_standings_df.insert(0, "Rank", range(1, len(proj_standings_df) + 1))

            proj_standings_df[""] = proj_standings_df["Team"].apply(
                lambda x: "👉" if x == "My Team" else ""
            )
            cols = proj_standings_df.columns.tolist()
            cols = [cols[-1]] + cols[:-1]
            proj_standings_df = proj_standings_df[cols]

            st.dataframe(proj_standings_df, hide_index=True, width="stretch")

    # Show roster summary
    if my_roster and len(my_roster) > 0:
        st.divider()
        st.subheader("My Roster Summary")

        roster_df = projections[projections["Name"].isin(my_roster)]
        hitters = roster_df[roster_df["player_type"] == "hitter"]
        pitchers = roster_df[roster_df["player_type"] == "pitcher"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hitters", len(hitters))
        with col2:
            st.metric("Pitchers", len(pitchers))
        with col3:
            total_sgp = roster_df["SGP"].sum()
            st.metric("Total SGP", f"{total_sgp:.1f}")
        with col4:
            avg_sgp = roster_df["SGP"].mean()
            st.metric("Avg SGP", f"{avg_sgp:.1f}")
    else:
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Players", len(projections))
        with col2:
            st.metric("My Roster", len(my_roster) if my_roster else 0)
        with col3:
            st.metric(
                "Opponent Teams", len(opponent_rosters) if opponent_rosters else 0
            )


def show_my_team():
    """Show current team roster and analysis."""
    st.title("My Team")

    my_roster = st.session_state.my_roster
    projections = st.session_state.projections

    if not my_roster:
        st.warning("No roster data available. Make sure Fantrax data is loaded.")
        return

    # Filter to my roster
    roster_df = projections[projections["Name"].isin(my_roster)]

    # Split by player type
    hitters = roster_df[roster_df["player_type"] == "hitter"].copy()
    pitchers = roster_df[roster_df["player_type"] == "pitcher"].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Hitters ({len(hitters)})")
        if len(hitters) > 0:
            # Select columns that exist
            hitter_cols = [
                "Name",
                "Position",
                "Team",
                "PA",
                "R",
                "HR",
                "RBI",
                "SB",
                "OPS",
                "SGP",
            ]
            available_cols = [c for c in hitter_cols if c in hitters.columns]
            st.dataframe(
                hitters[available_cols].sort_values("SGP", ascending=False),
                hide_index=True,
            )
        else:
            st.info("No hitters on roster")

    with col2:
        st.subheader(f"Pitchers ({len(pitchers)})")
        if len(pitchers) > 0:
            pitcher_cols = [
                "Name",
                "Position",
                "Team",
                "IP",
                "W",
                "SV",
                "K",
                "ERA",
                "WHIP",
                "SGP",
            ]
            available_cols = [c for c in pitcher_cols if c in pitchers.columns]
            st.dataframe(
                pitchers[available_cols].sort_values("SGP", ascending=False),
                hide_index=True,
            )
        else:
            st.info("No pitchers on roster")

    # Team totals
    st.divider()
    st.subheader("Team Totals")

    from optimizer.data_loader import (
        compute_all_opponent_totals,
        compute_team_totals,
        strip_name_suffix,
    )

    totals = compute_team_totals(my_roster, projections)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Hitting**")
        hit_data = {
            "Category": ["R", "HR", "RBI", "SB", "OPS"],
            "Value": [
                f"{int(totals['R'])}",
                f"{int(totals['HR'])}",
                f"{int(totals['RBI'])}",
                f"{int(totals['SB'])}",
                f"{totals['OPS']:.3f}",
            ],
        }
        st.dataframe(pd.DataFrame(hit_data), hide_index=True, width="stretch")

    with col2:
        st.markdown("**Pitching**")
        pitch_data = {
            "Category": ["W", "SV", "K", "ERA", "WHIP"],
            "Value": [
                f"{int(totals['W'])}",
                f"{int(totals['SV'])}",
                f"{int(totals['K'])}",
                f"{totals['ERA']:.2f}",
                f"{totals['WHIP']:.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(pitch_data), hide_index=True, width="stretch")

    # Visualizations section
    opponent_rosters = st.session_state.opponent_rosters
    if opponent_rosters and len(opponent_rosters) > 0:
        st.divider()
        st.subheader("Team Performance Visualizations")

        from dashboard.components import display_figure
        from optimizer.visualizations import plot_team_dashboard

        # Compute opponent totals and build team name map
        opponent_rosters_indexed = {
            i + 1: names for i, (team, names) in enumerate(opponent_rosters.items())
        }
        team_names_map = {
            i + 1: team for i, (team, names) in enumerate(opponent_rosters.items())
        }
        opponent_totals = compute_all_opponent_totals(
            opponent_rosters_indexed, projections
        )

        # Combined 3-panel visualization
        fig_dashboard = plot_team_dashboard(
            my_totals=totals,
            opponent_totals=opponent_totals,
            optimal_roster_names=list(my_roster),
            projections=projections,
            team_names=team_names_map,
        )
        display_figure(fig_dashboard, width=2200)

        # ==========================================================================
        # POSITION SENSITIVITY ANALYSIS
        # ==========================================================================
        st.divider()
        st.subheader("Position Sensitivity Analysis")
        st.caption(
            "Analyze which positions offer the most upgrade opportunity "
            "and where your roster is strongest/weakest."
        )

        # Initialize session state for sensitivity results
        if "position_sensitivity" not in st.session_state:
            st.session_state.position_sensitivity = None

        if st.button(
            "📊 Analyze Positions"
            if st.session_state.position_sensitivity is None
            else "🔄 Refresh Analysis",
            key="btn_position_sensitivity",
        ):
            _compute_position_sensitivity(
                my_roster, opponent_rosters, projections, totals, opponent_totals
            )

        # Display results if computed
        if st.session_state.position_sensitivity is not None:
            _display_position_sensitivity_plots()


def show_trades():
    """Show trade analysis with targets, pieces, and recommendations."""
    st.title("Trade Analysis")

    from optimizer.data_loader import strip_name_suffix

    my_roster = st.session_state.my_roster
    opponent_rosters = st.session_state.opponent_rosters
    projections = st.session_state.projections

    if not my_roster or len(my_roster) == 0:
        st.warning("No roster data available. Load data first.")
        return

    if not opponent_rosters or len(opponent_rosters) == 0:
        st.warning("No opponent roster data available.")
        return

    # Auto-compute situation when entering the Trades tab
    if st.session_state.situation is None:
        _compute_roster_situation()
        st.rerun()

    # Show current situation
    situation = st.session_state.situation
    st.success(
        f"**Current Position:** Win probability {situation['win_probability']:.1%} | "
        f"Expected wins {situation['expected_wins']:.1f}/60"
    )

    # Trade Builder section - available immediately after situation is computed
    if st.session_state.player_values is not None:
        st.divider()
        _show_trade_builder(projections, opponent_rosters)

    st.divider()

    # Find recommended trades section
    st.subheader("🔍 Find Recommended Trades")
    st.caption("Automatically search for favorable trade opportunities")

    if st.button(
        "🔍 Find Trade Targets"
        if st.session_state.trade_results is None
        else "🔄 Refresh Trade Targets",
        width="stretch",
        type="primary" if st.session_state.trade_results is None else "secondary",
    ):
        _compute_trade_targets(
            fairness_threshold=st.session_state.get("trade_fairness", 0.30),
            min_improvement=st.session_state.get("trade_min_improvement", 0.003),
        )

    # Display situation analysis
    if st.session_state.situation:
        situation = st.session_state.situation

        st.divider()
        st.subheader("Category Analysis")

        cat_data = []
        for cat, analysis in situation["category_analysis"].items():
            cat_data.append(
                {
                    "Category": cat,
                    "My Value": f"{analysis['my_value']:.3f}"
                    if cat in ["OPS", "ERA", "WHIP"]
                    else f"{int(analysis['my_value'])}",
                    "Avg Opponent": f"{analysis['avg_opponent']:.3f}"
                    if cat in ["OPS", "ERA", "WHIP"]
                    else f"{int(analysis['avg_opponent'])}",
                    "Win Prob": f"{analysis['avg_win_prob']:.0%}",
                    "Status": analysis["status"],
                }
            )

        st.dataframe(pd.DataFrame(cat_data), hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Strengths:** {', '.join(situation['strengths']) or 'None'}")
        with col2:
            st.markdown(
                f"**Weaknesses:** {', '.join(situation['weaknesses']) or 'None'}"
            )

    # Display trade targets and pieces side by side
    if (
        st.session_state.trade_targets is not None
        and st.session_state.trade_pieces is not None
    ):
        st.divider()

        # Build opponent name map for display
        opponent_name_map = {}
        for i, team_name in enumerate(opponent_rosters.keys(), 1):
            opponent_name_map[i] = team_name

        targets_col, pieces_col = st.columns(2)

        with targets_col:
            st.subheader("🎯 Players to Acquire")
            st.caption("Ranked by value to your team")

            targets = st.session_state.trade_targets
            if not targets.empty:
                targets_display = []
                for _, row in targets.iterrows():
                    owner_id = row.get("owner_id")
                    owner_name = opponent_name_map.get(owner_id, f"Team {owner_id}")
                    targets_display.append(
                        {
                            "Player": strip_name_suffix(row["Name"]),
                            "Pos": row.get("Position", ""),
                            "Owner": owner_name,
                            "Value": f"+{row['delta_V_acquire']:.2%}",
                            "SGP": f"{row['generic_value']:.1f}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(targets_display),
                    hide_index=True,
                    column_config={
                        "Player": st.column_config.TextColumn("Player", width="medium"),
                        "Pos": st.column_config.TextColumn("Pos", width="small"),
                        "Owner": st.column_config.TextColumn("Owner", width="medium"),
                        "Value": st.column_config.TextColumn(
                            "Value to Me", width="small"
                        ),
                        "SGP": st.column_config.TextColumn("SGP", width="small"),
                    },
                )
            else:
                st.info("No trade targets identified.")

        with pieces_col:
            st.subheader("📤 Players to Offer")
            st.caption("Your most expendable players")

            pieces = st.session_state.trade_pieces
            if not pieces.empty:
                pieces_display = []
                for _, row in pieces.iterrows():
                    lose_cost = row.get("delta_V_lose", 0)
                    pieces_display.append(
                        {
                            "Player": strip_name_suffix(row["Name"]),
                            "Pos": row.get("Position", ""),
                            "Lose Cost": f"{lose_cost:.2%}" if lose_cost else "N/A",
                            "SGP": f"{row['generic_value']:.1f}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(pieces_display),
                    hide_index=True,
                    column_config={
                        "Player": st.column_config.TextColumn("Player", width="medium"),
                        "Pos": st.column_config.TextColumn("Pos", width="small"),
                        "Lose Cost": st.column_config.TextColumn(
                            "Lose Cost", width="small"
                        ),
                        "SGP": st.column_config.TextColumn("SGP", width="small"),
                    },
                )
            else:
                st.info("No trade pieces identified.")

        # Trade search parameters - shown after candidates, before recommendations
        st.divider()
        st.subheader("⚙️ Trade Search Settings")
        st.caption("Adjust these to find more or fewer trades")

        param_col1, param_col2 = st.columns(2)

        with param_col1:
            fairness_threshold = st.slider(
                "Fairness Threshold (SGP Balance)",
                min_value=0.05,
                max_value=0.50,
                value=st.session_state.get("trade_fairness", 0.30),
                step=0.05,
                key="trade_fairness",
                help="How much SGP imbalance to allow. 10% = fair trades only. 50% = show trades where you send much more/less SGP than you receive.",
            )

        with param_col2:
            min_improvement = st.slider(
                "Min Win Probability Improvement",
                min_value=-0.01,
                max_value=0.02,
                value=st.session_state.get("trade_min_improvement", 0.003),
                step=0.001,
                format="%.3f",
                key="trade_min_improvement",
                help="Minimum win% gain. Set to 0 or negative to see trades that hurt your win probability.",
            )

        # Brief interpretation
        hint_parts = []
        if fairness_threshold >= 0.30:
            hint_parts.append("lenient fairness (showing unbalanced trades)")
        if min_improvement <= 0:
            hint_parts.append("including trades that hurt you")

        if hint_parts:
            st.caption(f"💡 Current: {', '.join(hint_parts)}")

        if st.button("🔄 Re-run with New Settings", width="stretch"):
            _compute_trade_targets(
                fairness_threshold=fairness_threshold,
                min_improvement=min_improvement,
            )

    # Display trade recommendations
    if st.session_state.trade_results is not None:
        st.divider()
        st.subheader("💱 Recommended Trades")

        if len(st.session_state.trade_results) == 0:
            st.info(
                "No favorable fair trades found. Consider adjusting your targets or being more flexible."
            )
        else:
            # Build opponent name map
            opponent_name_map = {}
            for i, team_name in enumerate(opponent_rosters.keys(), 1):
                opponent_name_map[i] = team_name

            # Create summary table
            trade_table_data = []
            for i, trade in enumerate(st.session_state.trade_results, 1):
                send_str = ", ".join(
                    strip_name_suffix(p) for p in trade["send_players"]
                )
                receive_str = ", ".join(
                    strip_name_suffix(p) for p in trade["receive_players"]
                )
                partner_id = trade.get("partner_id")
                partner_name = opponent_name_map.get(partner_id, f"Team {partner_id}")

                trade_table_data.append(
                    {
                        "#": i,
                        "Partner": partner_name,
                        "You Send": send_str,
                        "You Get": receive_str,
                        "ΔWin%": f"{trade['delta_V']:+.2%}",
                        "Fairness": "Fair" if trade["is_fair"] else "Unfair",
                        "Rec": trade["recommendation"],
                    }
                )

            st.dataframe(
                pd.DataFrame(trade_table_data),
                hide_index=True,
                column_config={
                    "#": st.column_config.NumberColumn("#", width="small"),
                    "Partner": st.column_config.TextColumn("Partner", width="medium"),
                    "You Send": st.column_config.TextColumn("You Send", width="large"),
                    "You Get": st.column_config.TextColumn("You Get", width="large"),
                    "ΔWin%": st.column_config.TextColumn("ΔWin%", width="small"),
                    "Fairness": st.column_config.TextColumn("Fair?", width="small"),
                    "Rec": st.column_config.TextColumn("Rec", width="small"),
                },
            )

            # Expandable details for each trade
            st.markdown("**Trade Details:**")
            for i, trade in enumerate(st.session_state.trade_results[:10], 1):
                send_str = ", ".join(
                    strip_name_suffix(p) for p in trade["send_players"]
                )
                receive_str = ", ".join(
                    strip_name_suffix(p) for p in trade["receive_players"]
                )

                with st.expander(f"Trade #{i}: Send {send_str} → Get {receive_str}"):
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.markdown("**You Send:**")
                        for p in trade["send_players"]:
                            player_row = projections[projections["Name"] == p]
                            if not player_row.empty:
                                sgp = player_row.iloc[0].get("SGP", 0)
                                st.markdown(
                                    f"- {strip_name_suffix(p)} (SGP: {sgp:.1f})"
                                )

                        st.markdown("**You Receive:**")
                        for p in trade["receive_players"]:
                            player_row = projections[projections["Name"] == p]
                            if not player_row.empty:
                                sgp = player_row.iloc[0].get("SGP", 0)
                                st.markdown(
                                    f"- {strip_name_suffix(p)} (SGP: {sgp:.1f})"
                                )

                    with detail_col2:
                        st.metric("Win Probability Change", f"{trade['delta_V']:+.2%}")
                        st.markdown(
                            f"**Dynasty SGP Change:** {trade['delta_generic']:+.1f}"
                        )
                        st.markdown(
                            f"**Fairness:** {'✅ Fair' if trade['is_fair'] else '⚠️ Unfair'}"
                        )
                        st.markdown(
                            f"**Recommendation:** **{trade['recommendation']}**"
                        )


def _show_trade_builder(projections, opponent_rosters):
    """Show the Trade Builder section for custom trade evaluation."""
    from optimizer.data_loader import strip_name_suffix

    st.subheader("🔧 Trade Builder")
    st.caption("Build and evaluate custom trades")

    player_values = st.session_state.player_values

    # Get options for dropdowns
    my_players = player_values[player_values["on_my_roster"]]["Name"].tolist()
    opponent_players = player_values[~player_values["on_my_roster"]]["Name"].tolist()

    my_player_options = [
        f"{strip_name_suffix(n)} (SGP: {player_values[player_values['Name'] == n].iloc[0]['generic_value']:.1f})"
        for n in my_players
    ]
    opp_player_options = [
        f"{strip_name_suffix(n)} (SGP: {player_values[player_values['Name'] == n].iloc[0]['generic_value']:.1f})"
        for n in opponent_players
    ]

    my_name_map = dict(zip(my_player_options, my_players))
    opp_name_map = dict(zip(opp_player_options, opponent_players))

    builder_col1, builder_col2 = st.columns(2)

    with builder_col1:
        send_selections = st.multiselect(
            "Players you SEND",
            options=my_player_options,
            key="trade_builder_send",
            help="Select players from your roster to trade away",
        )

    with builder_col2:
        receive_selections = st.multiselect(
            "Players you RECEIVE",
            options=opp_player_options,
            key="trade_builder_receive",
            help="Select players from opponent rosters to acquire",
        )

    if send_selections and receive_selections:
        # Evaluate the trade
        send_names = [my_name_map[s] for s in send_selections]
        receive_names = [opp_name_map[r] for r in receive_selections]

        if st.button("📊 Evaluate Trade", type="primary"):
            _evaluate_custom_trade(send_names, receive_names)


def _compute_roster_situation():
    """Compute and cache roster situation and player values for Trade Builder."""
    from optimizer.data_loader import compute_all_opponent_totals, compute_team_totals
    from optimizer.trade_engine import compute_player_values, compute_roster_situation

    my_roster = st.session_state.my_roster
    opponent_rosters = st.session_state.opponent_rosters
    projections = st.session_state.projections

    with st.spinner("Computing roster situation..."):
        # Convert opponent rosters to indexed format
        opponent_rosters_indexed = {
            i + 1: names for i, (team, names) in enumerate(opponent_rosters.items())
        }
        opponent_totals = compute_all_opponent_totals(
            opponent_rosters_indexed, projections
        )

        situation = compute_roster_situation(my_roster, projections, opponent_totals)

        st.session_state.situation = situation
        st.session_state.opponent_totals = opponent_totals

        # Also compute player values for Trade Builder (separate from recommendations)
        my_totals = compute_team_totals(my_roster, projections)
        category_sigmas = situation["category_sigmas"]

        # Get all rostered player names
        opponent_roster_names = set().union(*opponent_rosters.values())
        all_roster_names = my_roster | opponent_roster_names

        player_values = compute_player_values(
            player_names=all_roster_names,
            my_roster_names=my_roster,
            projections=projections,
            my_totals=my_totals,
            opponent_totals=opponent_totals,
            category_sigmas=category_sigmas,
        )
        st.session_state.player_values = player_values

    st.rerun()


def _compute_trade_targets(
    fairness_threshold: float = 0.10,
    min_improvement: float = 0.001,
):
    """Compute trade targets, pieces, and candidate trades."""
    from optimizer.data_loader import (
        compute_all_opponent_totals,
        compute_team_totals,
    )
    from optimizer.trade_engine import (
        compute_player_values,
        compute_roster_situation,
        generate_trade_candidates,
        identify_trade_pieces,
        identify_trade_targets,
    )

    my_roster = st.session_state.my_roster
    opponent_rosters = st.session_state.opponent_rosters
    projections = st.session_state.projections

    with st.spinner("Computing trade targets (this may take a minute)..."):
        # Convert opponent rosters to indexed format
        opponent_rosters_indexed = {
            i + 1: names for i, (team, names) in enumerate(opponent_rosters.items())
        }
        opponent_totals = compute_all_opponent_totals(
            opponent_rosters_indexed, projections
        )

        # Compute situation if not cached
        if st.session_state.situation is None:
            situation = compute_roster_situation(
                my_roster, projections, opponent_totals
            )
            st.session_state.situation = situation
        else:
            situation = st.session_state.situation

        # Compute my totals
        my_totals = compute_team_totals(my_roster, projections)
        category_sigmas = situation["category_sigmas"]

        # Get all rostered player names for value computation
        opponent_roster_names = set().union(*opponent_rosters.values())
        all_roster_names = my_roster | opponent_roster_names

        # Compute player values
        player_values = compute_player_values(
            player_names=all_roster_names,
            my_roster_names=my_roster,
            projections=projections,
            my_totals=my_totals,
            opponent_totals=opponent_totals,
            category_sigmas=category_sigmas,
        )

        st.session_state.player_values = player_values

        # Identify trade targets (players to acquire)
        trade_targets = identify_trade_targets(
            player_values=player_values,
            my_roster_names=my_roster,
            opponent_rosters=opponent_rosters_indexed,
            n_targets=15,
        )
        st.session_state.trade_targets = trade_targets

        # Identify trade pieces (players to offer)
        trade_pieces = identify_trade_pieces(
            player_values=player_values,
            my_roster_names=my_roster,
            n_pieces=15,
        )
        st.session_state.trade_pieces = trade_pieces

        # Generate trade candidates with configurable thresholds
        trade_candidates = generate_trade_candidates(
            my_roster_names=my_roster,
            player_values=player_values,
            opponent_rosters=opponent_rosters_indexed,
            projections=projections,
            my_totals=my_totals,
            opponent_totals=opponent_totals,
            category_sigmas=category_sigmas,
            max_send=2,
            max_receive=2,
            n_targets=15,
            n_pieces=15,
            n_candidates=20,
            fairness_threshold=fairness_threshold,
            min_improvement=min_improvement,
        )

        st.session_state.trade_results = trade_candidates
        st.session_state.opponent_totals = opponent_totals

    st.rerun()


def _evaluate_custom_trade(send_names: list[str], receive_names: list[str]):
    """Evaluate a custom trade from the Trade Builder with radar chart visualization."""
    from dashboard.components import radar_chart_with_overlay
    from optimizer.data_loader import (
        ALL_CATEGORIES,
        compute_team_totals,
        strip_name_suffix,
    )
    from optimizer.trade_engine import evaluate_trade

    my_roster = st.session_state.my_roster
    projections = st.session_state.projections
    player_values = st.session_state.player_values
    opponent_totals = st.session_state.opponent_totals
    situation = st.session_state.situation

    my_totals = compute_team_totals(my_roster, projections)
    category_sigmas = situation["category_sigmas"]

    # Convert opponent rosters to indexed format
    opponent_rosters = st.session_state.opponent_rosters
    opponent_rosters_indexed = {
        i + 1: names for i, (team, names) in enumerate(opponent_rosters.items())
    }

    # Get parameters from session state
    fairness_threshold = st.session_state.get("trade_fairness", 0.10)
    min_improvement = st.session_state.get("trade_min_improvement", 0.001)

    result = evaluate_trade(
        send_players=send_names,
        receive_players=receive_names,
        player_values=player_values,
        my_roster_names=my_roster,
        projections=projections,
        my_totals=my_totals,
        opponent_totals=opponent_totals,
        category_sigmas=category_sigmas,
        opponent_rosters=opponent_rosters_indexed,
        fairness_threshold=fairness_threshold,
        min_improvement=min_improvement,
    )

    # Display results
    st.divider()
    st.subheader("Trade Evaluation Results")

    # Check for invalid trade first
    if "invalid_reason" in result:
        st.error(f"**Invalid Trade:** {result['invalid_reason']}")
        st.caption(
            "This trade would violate roster composition rules (min/max hitters or pitchers)."
        )

        # Still show what was attempted
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown("**You tried to send:**")
            for p in send_names:
                st.markdown(f"- {strip_name_suffix(p)}")
        with detail_col2:
            st.markdown("**You tried to receive:**")
            for p in receive_names:
                st.markdown(f"- {strip_name_suffix(p)}")
        return

    # Compute new roster totals for radar chart
    send_set = set(send_names)
    receive_set = set(receive_names)
    new_roster = (my_roster - send_set) | receive_set
    new_totals = compute_team_totals(new_roster, projections)

    # Build team names map for the dashboard
    opponent_rosters_for_names = st.session_state.opponent_rosters
    team_names_map = {
        i + 1: team
        for i, (team, names) in enumerate(opponent_rosters_for_names.items())
    }

    # Full comparison dashboard
    from dashboard.components import display_figure
    from optimizer.visualizations import plot_comparison_dashboard

    send_display = ", ".join([strip_name_suffix(p) for p in send_names])
    receive_display = ", ".join([strip_name_suffix(p) for p in receive_names])
    fig_dashboard = plot_comparison_dashboard(
        before_totals=my_totals,
        after_totals=new_totals,
        opponent_totals=opponent_totals,
        team_names=team_names_map,
        title=f"Trade: Send {send_display} → Receive {receive_display}",
    )
    display_figure(fig_dashboard, width=2200)

    # Summary metrics row
    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.markdown("**Trade Summary**")
        delta_V = result["delta_V"]
        if delta_V > 0.001:
            st.success(f"Win Prob Change: +{delta_V:.2%}")
        elif delta_V < -0.001:
            st.error(f"Win Prob Change: {delta_V:.2%}")
        else:
            st.info("Win Prob Change: ~0%")

        st.markdown(f"Dynasty SGP Change: {result['delta_generic']:+.1f}")

        if result["is_fair"]:
            st.success("✅ Fair trade")
        else:
            st.warning("⚠️ Unfair trade")

        rec = result["recommendation"]
        if rec == "ACCEPT":
            st.success(f"Recommendation: {rec}")
        elif rec == "STEAL":
            st.success(f"Recommendation: {rec} 🎉")
        elif rec == "REJECT":
            st.error(f"Recommendation: {rec}")
        else:
            st.info(f"Recommendation: {rec}")

    with summary_col2:
        st.markdown("**You Send:**")
        total_send_sgp = 0
        for p in send_names:
            player_row = projections[projections["Name"] == p]
            if not player_row.empty:
                sgp = player_row.iloc[0].get("SGP", 0)
                total_send_sgp += sgp
                st.markdown(f"- {strip_name_suffix(p)} (SGP: {sgp:.1f})")
        st.markdown(f"**Total SGP:** {total_send_sgp:.1f}")

    with summary_col3:
        st.markdown("**You Receive:**")
        total_receive_sgp = 0
        for p in receive_names:
            player_row = projections[projections["Name"] == p]
            if not player_row.empty:
                sgp = player_row.iloc[0].get("SGP", 0)
                total_receive_sgp += sgp
                st.markdown(f"- {strip_name_suffix(p)} (SGP: {sgp:.1f})")
        st.markdown(f"**Total SGP:** {total_receive_sgp:.1f}")

    # Category impact details table
    with st.expander("📊 Category Impact Details"):
        impact_data = []
        for cat in ALL_CATEGORIES:
            old_val = my_totals[cat]
            new_val = new_totals[cat]
            diff = new_val - old_val

            if cat in ["OPS", "ERA", "WHIP"]:
                impact_data.append(
                    {
                        "Category": cat,
                        "Before": f"{old_val:.3f}",
                        "After": f"{new_val:.3f}",
                        "Change": f"{diff:+.3f}",
                    }
                )
            else:
                impact_data.append(
                    {
                        "Category": cat,
                        "Before": f"{int(old_val)}",
                        "After": f"{int(new_val)}",
                        "Change": f"{int(diff):+d}",
                    }
                )

        st.dataframe(pd.DataFrame(impact_data), hide_index=True)


def show_simulator():
    """Show roster simulator with free agent browser."""
    st.title("Free Agents")

    my_roster = st.session_state.my_roster
    opponent_rosters = st.session_state.opponent_rosters
    projections = st.session_state.projections

    if not my_roster or len(my_roster) == 0:
        st.warning("No roster data available. Load data first.")
        return

    from optimizer.data_loader import strip_name_suffix

    # ==========================================================================
    # FREE AGENTS BROWSER
    # ==========================================================================
    st.subheader("🔍 Free Agent Browser")
    st.caption(
        "Browse available free agents, then use the simulator below to test adding them"
    )

    # Free agents = no owner
    all_free_agents = projections[projections["owner"].isna()].copy()

    # Filter controls
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        fa_player_type = st.selectbox(
            "Player Type", ["All", "Hitters", "Pitchers"], key="fa_type"
        )

    with filter_col2:
        fa_position = st.selectbox(
            "Position",
            ["All", "C", "1B", "2B", "SS", "3B", "OF", "DH", "SP", "RP"],
            key="fa_pos",
        )

    with filter_col3:
        fa_min_sgp = st.slider("Min SGP", 0.0, 20.0, 3.0, key="fa_min_sgp")

    # Apply filters
    filtered_fa = all_free_agents.copy()
    if fa_player_type == "Hitters":
        filtered_fa = filtered_fa[filtered_fa["player_type"] == "hitter"]
    elif fa_player_type == "Pitchers":
        filtered_fa = filtered_fa[filtered_fa["player_type"] == "pitcher"]

    if fa_position != "All":
        filtered_fa = filtered_fa[
            filtered_fa["Position"].str.contains(fa_position, na=False)
        ]

    filtered_fa = filtered_fa[filtered_fa["SGP"] >= fa_min_sgp]
    filtered_fa = filtered_fa.sort_values("SGP", ascending=False)

    st.markdown(f"**{len(filtered_fa)} free agents found**")

    # Display columns based on player type
    if fa_player_type == "Hitters":
        fa_cols = [
            "Name",
            "Position",
            "Team",
            "PA",
            "R",
            "HR",
            "RBI",
            "SB",
            "OPS",
            "SGP",
        ]
    elif fa_player_type == "Pitchers":
        fa_cols = [
            "Name",
            "Position",
            "Team",
            "IP",
            "W",
            "SV",
            "K",
            "ERA",
            "WHIP",
            "SGP",
        ]
    else:
        fa_cols = ["Name", "Position", "Team", "player_type", "SGP", "PA", "IP"]

    available_fa_cols = [c for c in fa_cols if c in filtered_fa.columns]

    st.dataframe(
        filtered_fa[available_fa_cols].head(50),
        width="stretch",
        hide_index=True,
    )

    # ==========================================================================
    # ROSTER SIMULATOR
    # ==========================================================================
    st.divider()
    st.subheader("🎮 Simulate Roster Changes")
    st.caption("Test roster changes and see their impact on your win probability")

    # Get free agents sorted by SGP for dropdown
    free_agents = all_free_agents[all_free_agents["SGP"] > 0].sort_values(
        "SGP", ascending=False
    )

    # Also include opponent players as potential additions
    opponent_names = set()
    for roster in opponent_rosters.values():
        opponent_names.update(roster)
    opponent_players = projections[projections["Name"].isin(opponent_names)].copy()
    opponent_players = opponent_players.sort_values("SGP", ascending=False)

    # Combine free agents and opponent players for ADD options
    available_for_add = pd.concat([free_agents.head(100), opponent_players.head(50)])
    available_for_add = available_for_add.drop_duplicates(subset="Name")

    fa_options = [
        f"{strip_name_suffix(row['Name'])} ({row['Position']}, SGP: {row['SGP']:.1f})"
        for _, row in available_for_add.iterrows()
    ]
    fa_name_map = {
        f"{strip_name_suffix(row['Name'])} ({row['Position']}, SGP: {row['SGP']:.1f})": row[
            "Name"
        ]
        for _, row in available_for_add.iterrows()
    }

    # Get my roster sorted by SGP
    roster_df = projections[projections["Name"].isin(my_roster)].copy()
    roster_df = roster_df.sort_values(
        "SGP", ascending=True
    )  # Lowest SGP first for dropping
    roster_options = [
        f"{strip_name_suffix(row['Name'])} ({row['Position']}, SGP: {row['SGP']:.1f})"
        for _, row in roster_df.iterrows()
    ]
    roster_name_map = {
        f"{strip_name_suffix(row['Name'])} ({row['Position']}, SGP: {row['SGP']:.1f})": row[
            "Name"
        ]
        for _, row in roster_df.iterrows()
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Players to ADD**")
        add_selections = st.multiselect(
            "Select players to add",
            options=fa_options,
            key="sim_add",
            help="Select players from free agency or opponent rosters",
        )

    with col2:
        st.markdown("**Players to DROP**")
        drop_selections = st.multiselect(
            "Select players to drop",
            options=roster_options,
            key="sim_drop",
            help="Select players from your roster to drop",
        )

    # Action buttons
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        simulate_clicked = st.button("🎮 Simulate", width="stretch", type="primary")

    with btn_col2:
        reset_clicked = st.button("🔄 Reset", width="stretch")

    if reset_clicked:
        # Clear the multiselect keys by deleting from session state
        if "sim_add" in st.session_state:
            del st.session_state["sim_add"]
        if "sim_drop" in st.session_state:
            del st.session_state["sim_drop"]
        st.rerun()

    if simulate_clicked:
        if not add_selections and not drop_selections:
            st.warning("Select at least one player to add or drop.")
            return

        _run_simulation(add_selections, drop_selections, fa_name_map, roster_name_map)


def _run_simulation(add_selections, drop_selections, fa_name_map, roster_name_map):
    """Run roster simulation with radar chart visualization."""
    from dashboard.components import radar_chart_with_overlay
    from optimizer.data_loader import (
        ALL_CATEGORIES,
        compute_all_opponent_totals,
        compute_team_totals,
        estimate_projection_uncertainty,
        strip_name_suffix,
    )
    from optimizer.trade_engine import compute_win_probability

    my_roster = st.session_state.my_roster
    opponent_rosters = st.session_state.opponent_rosters
    projections = st.session_state.projections

    # Convert selections to actual names
    add_names = {fa_name_map[s] for s in add_selections}
    drop_names = {roster_name_map[s] for s in drop_selections}

    # Create new roster
    new_roster = (my_roster - drop_names) | add_names

    with st.spinner("Computing simulation results..."):
        # Compute opponent totals
        opponent_rosters_indexed = {
            i + 1: names for i, (team, names) in enumerate(opponent_rosters.items())
        }
        opponent_totals = compute_all_opponent_totals(
            opponent_rosters_indexed, projections
        )

        # Compute before
        old_totals = compute_team_totals(my_roster, projections)
        category_sigmas = estimate_projection_uncertainty(old_totals, opponent_totals)
        V_before, _ = compute_win_probability(
            old_totals, opponent_totals, category_sigmas
        )

        # Compute after
        new_totals = compute_team_totals(new_roster, projections)
        V_after, _ = compute_win_probability(
            new_totals, opponent_totals, category_sigmas
        )

    # Display results
    st.divider()

    # Build team names map for the dashboard
    team_names_map = {
        i + 1: team for i, (team, names) in enumerate(opponent_rosters.items())
    }

    # Full comparison dashboard
    from dashboard.components import display_figure
    from optimizer.visualizations import plot_comparison_dashboard

    add_display = (
        ", ".join([strip_name_suffix(n) for n in add_names]) if add_names else "None"
    )
    drop_display = (
        ", ".join([strip_name_suffix(n) for n in drop_names]) if drop_names else "None"
    )
    fig_dashboard = plot_comparison_dashboard(
        before_totals=old_totals,
        after_totals=new_totals,
        opponent_totals=opponent_totals,
        team_names=team_names_map,
        title=f"Free Agent Move: Add {add_display} / Drop {drop_display}",
    )
    display_figure(fig_dashboard, width=2200)

    # Summary metrics row
    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.markdown("**Impact Summary**")
        delta_V = V_after - V_before
        if delta_V > 0.001:
            st.success(f"Win Prob: {V_before:.1%} → {V_after:.1%} (+{delta_V:.2%})")
        elif delta_V < -0.001:
            st.error(f"Win Prob: {V_before:.1%} → {V_after:.1%} ({delta_V:.2%})")
        else:
            st.info(f"Win Prob: {V_before:.1%} → {V_after:.1%} (no change)")

    with summary_col2:
        st.markdown("**Players Added:**")
        if add_names:
            for name in add_names:
                st.markdown(f"- {strip_name_suffix(name)}")
        else:
            st.markdown("_None_")

    with summary_col3:
        st.markdown("**Players Dropped:**")
        if drop_names:
            for name in drop_names:
                st.markdown(f"- {strip_name_suffix(name)}")
        else:
            st.markdown("_None_")

    # Category impact table in expander
    with st.expander("📊 Category Impact Details"):
        impact_data = []
        for cat in ALL_CATEGORIES:
            old_val = old_totals[cat]
            new_val = new_totals[cat]
            diff = new_val - old_val

            if cat in ["OPS", "ERA", "WHIP"]:
                impact_data.append(
                    {
                        "Category": cat,
                        "Before": f"{old_val:.3f}",
                        "After": f"{new_val:.3f}",
                        "Change": f"{diff:+.3f}",
                    }
                )
            else:
                impact_data.append(
                    {
                        "Category": cat,
                        "Before": f"{int(old_val)}",
                        "After": f"{int(new_val)}",
                        "Change": f"{int(diff):+d}",
                    }
                )

        st.dataframe(pd.DataFrame(impact_data), hide_index=True)


def show_players():
    """Show searchable player database with full stats."""
    st.title("All Player Data")

    projections = st.session_state.projections

    st.markdown("""
    Complete reference for all player projections. Search by name or filter by type/ownership.
    """)

    # Search
    search = st.text_input("🔍 Search players", placeholder="Enter player name...")

    # Filters in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox("Player Type", ["All", "Hitters", "Pitchers"])
    with col2:
        owner_filter = st.selectbox(
            "Ownership",
            ["All", "Free Agents", "Rostered"],
        )
    with col3:
        min_sgp = st.slider(
            "Min SGP", min_value=0.0, max_value=20.0, value=0.0, step=0.5
        )

    # Apply filters
    filtered = projections.copy()

    if search:
        filtered = filtered[filtered["Name"].str.contains(search, case=False, na=False)]

    if owner_filter == "Free Agents":
        filtered = filtered[filtered["owner"].isna()]
    elif owner_filter == "Rostered":
        filtered = filtered[filtered["owner"].notna()]

    if type_filter == "Hitters":
        filtered = filtered[filtered["player_type"] == "hitter"]
    elif type_filter == "Pitchers":
        filtered = filtered[filtered["player_type"] == "pitcher"]

    filtered = filtered[filtered["SGP"] >= min_sgp]

    st.markdown(f"**{len(filtered)} players found**")

    # Define column sets for each player type
    # Core columns shown for all players
    core_cols = ["Name", "Position", "Team", "owner"]

    # Hitter-specific stats
    hitter_stat_cols = ["PA", "R", "HR", "RBI", "SB", "OPS"]

    # Pitcher-specific stats
    pitcher_stat_cols = ["IP", "W", "SV", "K", "ERA", "WHIP", "GS"]

    # Value columns
    value_cols = ["SGP", "WAR"]

    # Additional columns if available
    extra_cols = ["age", "dynasty_SGP"]

    if type_filter == "Hitters":
        # Show only hitter stats
        display_cols = core_cols + hitter_stat_cols + value_cols + extra_cols
        st.subheader("Hitter Projections")
    elif type_filter == "Pitchers":
        # Show only pitcher stats
        display_cols = core_cols + pitcher_stat_cols + value_cols + extra_cols
        st.subheader("Pitcher Projections")
    else:
        # Show all columns when showing both types
        display_cols = (
            core_cols
            + ["player_type"]
            + hitter_stat_cols
            + pitcher_stat_cols
            + value_cols
            + extra_cols
        )
        st.subheader("All Players")

    # Filter to available columns only
    available_cols = [c for c in display_cols if c in filtered.columns]

    # Sort and display with formatting
    display_df = filtered[available_cols].sort_values("SGP", ascending=False).head(200)

    # Format numeric columns for better display
    column_config = {
        "Name": st.column_config.TextColumn("Name", width="medium"),
        "Position": st.column_config.TextColumn("Pos", width="small"),
        "Team": st.column_config.TextColumn("Team", width="small"),
        "owner": st.column_config.TextColumn("Owner", width="medium"),
        "player_type": st.column_config.TextColumn("Type", width="small"),
        "PA": st.column_config.NumberColumn("PA", format="%d"),
        "R": st.column_config.NumberColumn("R", format="%d"),
        "HR": st.column_config.NumberColumn("HR", format="%d"),
        "RBI": st.column_config.NumberColumn("RBI", format="%d"),
        "SB": st.column_config.NumberColumn("SB", format="%d"),
        "OPS": st.column_config.NumberColumn("OPS", format="%.3f"),
        "IP": st.column_config.NumberColumn("IP", format="%.1f"),
        "W": st.column_config.NumberColumn("W", format="%d"),
        "SV": st.column_config.NumberColumn("SV", format="%d"),
        "K": st.column_config.NumberColumn("K", format="%d"),
        "ERA": st.column_config.NumberColumn("ERA", format="%.2f"),
        "WHIP": st.column_config.NumberColumn("WHIP", format="%.2f"),
        "GS": st.column_config.NumberColumn("GS", format="%d"),
        "SGP": st.column_config.NumberColumn("SGP", format="%.1f"),
        "WAR": st.column_config.NumberColumn("WAR", format="%.1f"),
        "age": st.column_config.NumberColumn("Age", format="%d"),
        "dynasty_SGP": st.column_config.NumberColumn("Dynasty SGP", format="%.1f"),
    }

    # Only include config for columns that exist
    active_config = {k: v for k, v in column_config.items() if k in available_cols}

    st.dataframe(
        display_df,
        column_config=active_config,
        hide_index=True,
    )

    # Show stat legend
    with st.expander("📊 Stat Legend"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Hitter Stats:**
            - **PA** - Plate Appearances
            - **R** - Runs Scored
            - **HR** - Home Runs
            - **RBI** - Runs Batted In
            - **SB** - Stolen Bases
            - **OPS** - On-base Plus Slugging
            """)
        with col2:
            st.markdown("""
            **Pitcher Stats:**
            - **IP** - Innings Pitched
            - **W** - Wins
            - **SV** - Saves
            - **K** - Strikeouts
            - **ERA** - Earned Run Average
            - **WHIP** - Walks + Hits per IP
            - **GS** - Games Started
            """)
        st.markdown("""
        **Value Stats:**
        - **SGP** - Standings Gain Points (fantasy value)
        - **WAR** - Wins Above Replacement
        - **Dynasty SGP** - Age-adjusted SGP for dynasty leagues
        """)


def _compute_position_sensitivity(
    my_roster: set[str],
    opponent_rosters: dict[str, set[str]],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
):
    """Compute position sensitivity analysis and store in session state."""
    from optimizer.data_loader import estimate_projection_uncertainty
    from optimizer.roster_optimizer import (
        compute_percentile_sensitivity,
        compute_position_sensitivity,
    )

    with st.spinner(
        "Computing position sensitivity analysis (this may take a minute)..."
    ):
        # Compute category sigmas
        category_sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)

        # Get all opponent roster names
        opponent_roster_names = set().union(*opponent_rosters.values())

        # Compute position sensitivity
        sensitivity_results = compute_position_sensitivity(
            my_roster_names=my_roster,
            opponent_roster_names=opponent_roster_names,
            projections=projections,
            opponent_totals=opponent_totals,
            category_sigmas=category_sigmas,
        )

        slot_data = sensitivity_results["slot_data"]
        ewa_df = sensitivity_results["ewa_df"]
        sensitivity_df = sensitivity_results["sensitivity_df"]
        baseline_ew = sensitivity_results["baseline_expected_wins"]

        # Compute percentile sensitivity
        pctl_ewa_df = compute_percentile_sensitivity(
            my_roster_names=my_roster,
            projections=projections,
            opponent_totals=opponent_totals,
            category_sigmas=category_sigmas,
            slot_data=slot_data,
            baseline_expected_wins=baseline_ew,
        )

        # Store results in session state
        st.session_state.position_sensitivity = {
            "slot_data": slot_data,
            "ewa_df": ewa_df,
            "sensitivity_df": sensitivity_df,
            "pctl_ewa_df": pctl_ewa_df,
            "baseline_expected_wins": baseline_ew,
        }

    st.rerun()


def _display_position_sensitivity_plots():
    """Display the position sensitivity analysis visualizations."""
    from dashboard.components import display_figure
    from optimizer.visualizations import (
        plot_percentile_ewa_curves,
        plot_position_distributions,
        plot_position_sensitivity_dashboard,
        plot_upgrade_opportunities,
    )

    results = st.session_state.position_sensitivity
    slot_data = results["slot_data"]
    ewa_df = results["ewa_df"]
    sensitivity_df = results["sensitivity_df"]
    pctl_ewa_df = results["pctl_ewa_df"]
    baseline_ew = results["baseline_expected_wins"]

    # Show baseline expected wins
    st.info(f"**Baseline Expected Wins:** {baseline_ew:.1f} / 60")

    # 4-panel dashboard
    st.markdown("#### Position Sensitivity Dashboard")
    st.caption(
        "Which positions give the most expected wins per SGP? "
        "Where are the best free agent upgrades?"
    )
    fig_dashboard = plot_position_sensitivity_dashboard(
        ewa_df, sensitivity_df, slot_data
    )
    display_figure(fig_dashboard, width=1400)

    # Additional visualizations in expanders
    with st.expander("📈 Upgrade Opportunities by Position", expanded=False):
        st.markdown(
            "**SGP gap between best available free agent and your worst player at each position.**\n\n"
            "Positive values (green) mean there's a better free agent available than your current player."
        )
        fig_upgrade = plot_upgrade_opportunities(slot_data)
        display_figure(fig_upgrade, width=1000)

    with st.expander("📊 EWA vs Percentile Curves", expanded=False):
        st.markdown(
            "**How much Expected Wins Added (EWA) do you gain from upgrading to higher-percentile players?**\n\n"
            "The red dashed line shows where your current worst player sits at that position."
        )
        fig_pctl = plot_percentile_ewa_curves(pctl_ewa_df)
        display_figure(fig_pctl, width=1400)

    with st.expander("📦 Player Distribution by Position", expanded=False):
        st.markdown(
            "**SGP distributions for available players at each position.**\n\n"
            "Red dots indicate your rostered players. Compare where you stand relative to the player pool."
        )
        fig_dist = plot_position_distributions(slot_data)
        display_figure(fig_dist, width=1400)


if __name__ == "__main__":
    main()
