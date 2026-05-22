import streamlit as st
import polars as pl
import pandas as pd
import get_todays_probables
import shared
import os
from unidecode import unidecode

st.set_page_config(layout="wide", page_title="Probable Pitchers")

st.markdown("# Today's Probable Pitchers")


def normalize_name(name: str) -> str:
    if not name or name == "TBD":
        return name
    # Remove accents, convert to lowercase, remove common suffixes
    name = unidecode(name).lower()
    for suffix in [" jr.", " sr.", " iii", " ii", " iv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


# Get today's date
today = shared.get_today_date()
date_str = today.strftime("%Y-%m-%d")

# Fetch probables from API (live)
probables_df = get_todays_probables.get_probables(date_str)

if not probables_df.is_empty():
    # Load all pitcher stats for enrichment
    pitcher_data_path = "data/all_pitcher_stats.csv"

    # Check if we have the all_pitcher_stats.csv, otherwise fallback to pitcher_data.csv (though it's filtered)
    if not os.path.exists(pitcher_data_path):
        pitcher_data_path = "data/pitcher_data.csv"

    if not os.path.exists(pitcher_data_path):
        st.error(f"Pitcher data not found. Please ensure data is loaded.")
        # Try to load it if not present
        try:
            shared.load_data(date_str)
        except:
            pass

    if os.path.exists(pitcher_data_path):
        pitcher_stats = pl.read_csv(pitcher_data_path)

        # In all_pitcher_stats.csv, 'term' might not exist or might be different.
        # In pitcher_data.csv, it's filtered.
        # Let's assume for now all_pitcher_stats.csv is the full season data.
        if "term" in pitcher_stats.columns:
            pitcher_stats = pitcher_stats.filter(pl.col("term") == "season")

        # Normalize names for joining
        pitcher_stats = pitcher_stats.with_columns(
            pl.col("Name")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("norm_name")
        )
        probables_df = probables_df.with_columns(
            pl.col("Away Pitcher")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("norm_away"),
            pl.col("Home Pitcher")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("norm_home"),
        )

        # Select relevant columns for join
        stats_subset = pitcher_stats.select(["norm_name", "xERA", "K-BB%"])

        # Join for away pitchers
        probables_df = probables_df.join(
            stats_subset, left_on="norm_away", right_on="norm_name", how="left"
        ).rename({"xERA": "Away xERA", "K-BB%": "Away K-BB%"})

        # Join for home pitchers
        probables_df = probables_df.join(
            stats_subset, left_on="norm_home", right_on="norm_name", how="left"
        ).rename({"xERA": "Home xERA", "K-BB%": "Home K-BB%"})

        # Final column selection and formatting
        display_df = probables_df.select(
            [
                "Time",
                "Away",
                "Away Pitcher",
                "Away xERA",
                "Away K-BB%",
                "Home",
                "Home Pitcher",
                "Home xERA",
                "Home K-BB%",
            ]
        ).fill_null("-")

        # Round the values if they are float strings
        def format_val(val):
            try:
                if val == "-":
                    return val
                f_val = float(val)
                if f_val > 10:  # Likely K-BB%
                    return f"{f_val:.1f}%"
                return f"{f_val:.2f}"
            except:
                return val

        pd_display = display_df.to_pandas()
        for col in ["Away xERA", "Away K-BB%", "Home xERA", "Home K-BB%"]:
            pd_display[col] = pd_display[col].apply(format_val)

        st.dataframe(pd_display, hide_index=True, use_container_width=True)
    else:
        # Just show the basic schedule if no stats found
        st.write("Schedule found, but pitcher metrics are unavailable.")
        st.dataframe(
            probables_df.select(
                ["Time", "Away", "Away Pitcher", "Home", "Home Pitcher"]
            ),
            hide_index=True,
            use_container_width=True,
        )
else:
    st.write(f"No games found for {date_str}.")
