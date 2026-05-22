import streamlit as st
import polars as pl
import pandas as pd
import get_todays_probables
import shared
import os

st.set_page_config(layout="wide", page_title="Probable Pitchers")

st.markdown("# Today's Probable Pitchers")

# Ensure data is loaded
today = shared.get_today_date()

probables_df = get_todays_probables.get_probables(today.strftime("%Y-%m-%d"))

if not probables_df.is_empty():
    # Load pitcher data to join
    pitcher_data_path = "data/pitcher_data.csv"
    if not os.path.exists(pitcher_data_path):
        st.error(
            f"Pitcher data not found at {pitcher_data_path}. Please wait for the daily data update."
        )
    else:
        pitcher_stats = pl.read_csv(pitcher_data_path).filter(
            pl.col("term") == "season"
        )

        # Join for away pitchers
        probables_df = probables_df.join(
            pitcher_stats.select(["Name", "xERA", "K-BB%", "on_team"]),
            left_on="Away Pitcher",
            right_on="Name",
            how="left",
        ).rename(
            {"xERA": "Away xERA", "K-BB%": "Away K-BB%", "on_team": "Away on_team"}
        )

        # Join for home pitchers
        probables_df = probables_df.join(
            pitcher_stats.select(["Name", "xERA", "K-BB%", "on_team"]),
            left_on="Home Pitcher",
            right_on="Name",
            how="left",
        ).rename(
            {"xERA": "Home xERA", "K-BB%": "Home K-BB%", "on_team": "Home on_team"}
        )

        # Fill nulls for display
        probables_df = probables_df.fill_null("-")

        # Style function for probables
        def style_probables(df):
            def color_on_team(row):
                styles = [""] * len(row)
                if row["Away on_team"] is True:
                    styles[2] = (
                        "background-color: #a6761d; color: white"  # Away Pitcher
                    )
                if row["Home on_team"] is True:
                    styles[4] = (
                        "background-color: #a6761d; color: white"  # Home Pitcher
                    )
                return styles

            return df.style.apply(color_on_team, axis=1)

        # Reorder and select columns for display
        display_probables = probables_df.select(
            [
                "Time",
                "Away",
                "Away Pitcher",
                "Away xERA",
                "Home",
                "Home Pitcher",
                "Home xERA",
                "Away on_team",
                "Home on_team",
            ]
        )

        styled_probables = style_probables(
            display_probables.to_pandas().drop(columns=["Away on_team", "Home on_team"])
        )
        st.dataframe(styled_probables, hide_index=True, use_container_width=True)
else:
    st.write("No probable pitchers found for today.")
