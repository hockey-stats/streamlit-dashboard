import streamlit as st
import polars as pl
import pandas as pd
import get_league_weekly_stats
import shared

st.set_page_config(layout="wide", page_title="League Weekly Stats")

st.markdown("# League Weekly Stats")

# Ensure data is loaded
today = shared.get_today_date()


@st.cache_data
def load_league_weekly_stats() -> pl.DataFrame:
    """Wrapper to fetch and cache league weekly stats."""
    return get_league_weekly_stats.run()


league_stats_df = load_league_weekly_stats()

if not league_stats_df.is_empty():
    # Ensure latest_week is an integer
    latest_week = int(league_stats_df.select(pl.col("week").max()).item())

    timeframe = st.radio(
        "Select Timeframe:",
        options=["Last 2 Weeks", "Full Season"],
        horizontal=True,
        index=0,
    )

    if timeframe == "Last 2 Weeks":
        st.write(f"**Weeks {max(1, latest_week - 1)} - {latest_week} Average Summary**")
    else:
        st.write(f"**Season Average Summary (Weeks 1 - {latest_week})**")

    # Use the aggregation logic from the data script
    agg_df = get_league_weekly_stats.get_aggregated_stats(league_stats_df, timeframe)

    # Ensure all numeric columns are Float64 for concat compatibility
    numeric_cols = [
        "R",
        "HR",
        "RBI",
        "SB",
        "AVG",
        "K",
        "ERA",
        "WHIP",
        "QS",
        "SV",
        "IP",
    ]
    agg_df = agg_df.with_columns([pl.col(c).cast(pl.Float64) for c in numeric_cols])

    # Calculate Average Column for the selected timeframe
    avg_row = agg_df.select(
        [
            pl.lit("League Average").alias("team"),
            *[pl.col(c).mean().alias(c) for c in numeric_cols],
        ]
    )

    # Reorder teams: Average first, then My team, then the rest
    my_team_name = "Ghostface millers"
    all_teams = agg_df["team"].to_list()
    my_team = next(
        (t for t in all_teams if t.lower() == my_team_name.lower()), my_team_name
    )
    other_teams = [t for t in all_teams if t != my_team]

    # Combine into ordered dataframe
    ordered_df = pl.concat(
        [
            avg_row,
            agg_df.filter(pl.col("team") == my_team),
            agg_df.filter(pl.col("team").is_in(other_teams)),
        ],
        how="vertical",
    )

    # Convert to pandas and transpose: index will be categories, columns will be teams
    pd_stats = (
        ordered_df.select(["team"] + numeric_cols).to_pandas().set_index("team").T
    )

    # --- ORIGINAL TABLE CODE (Commented for reference) ---
    # # Convert to object type to avoid pandas dtype warnings when setting string values
    # pd_stats = pd_stats.astype(object)
    #
    # # Round the values in the dataframe itself for display
    # for category in pd_stats.index:
    #     for team in pd_stats.columns:
    #         val = pd_stats.loc[category, team]
    #         if pd.isna(val) or val == "-":
    #             pd_stats.loc[category, team] = "-"
    #             continue
    #
    #         if category == "AVG":
    #             pd_stats.loc[category, team] = f"{float(val):.3f}"
    #         elif category in {"ERA", "WHIP"}:
    #             pd_stats.loc[category, team] = f"{float(val):.2f}"
    #         elif category == "IP":
    #             pd_stats.loc[category, team] = f"{float(val):.1f}"
    #         else:  # Counting stats
    #             pd_stats.loc[category, team] = f"{float(val):.1f}"
    #
    # st.table(pd_stats)
    # ---------------------------------------------------

    # Define categories by improvement direction
    lower_is_better = ["ERA", "WHIP"]
    higher_is_better = [c for c in numeric_cols if c not in lower_is_better]

    # Style the dataframe
    # We exclude 'League Average' from the gradient calculation to rank only members
    rank_cols = [c for c in pd_stats.columns if c != "League Average"]

    # Ensure all data in pd_stats is numeric for the styler
    pd_stats = pd_stats.apply(pd.to_numeric, errors="coerce")

    styler = pd_stats.style

    # Apply background gradients
    styler = styler.background_gradient(
        cmap="RdBu", subset=pd.IndexSlice[lower_is_better, rank_cols], axis=1
    )
    styler = styler.background_gradient(
        cmap="RdBu_r", subset=pd.IndexSlice[higher_is_better, rank_cols], axis=1
    )

    # Apply formatting: Default to 2 decimal places, then override AVG with 3
    styler = styler.format(precision=2, na_rep="-")
    styler = styler.format(formatter="{:.3f}", subset=pd.IndexSlice[["AVG"], :])

    # Use st.table as it is more reliable for rendering pandas Styler formatting
    st.table(styler)

else:
    st.write("League stats data not found.")
