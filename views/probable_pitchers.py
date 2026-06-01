import streamlit as st
import polars as pl
import pandas as pd
import get_todays_probables
import shared
import os
from unidecode import unidecode
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# st.set_page_config(layout="wide", page_title="Probable Pitchers")

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


def format_short_name(name: str) -> str:
    if not name or name == "TBD":
        return name
    name = name.strip()
    parts = name.split()
    if len(parts) > 1:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return name


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
        stats_subset = pitcher_stats.select(["norm_name", "ERA", "xERA", "K-BB%"])

        # Join for away pitchers
        probables_df = probables_df.join(
            stats_subset, left_on="norm_away", right_on="norm_name", how="left"
        ).rename({"ERA": "Away ERA", "xERA": "Away xERA", "K-BB%": "Away K-BB%"})

        # Join for home pitchers
        probables_df = probables_df.join(
            stats_subset, left_on="norm_home", right_on="norm_name", how="left"
        ).rename({"ERA": "Home ERA", "xERA": "Home xERA", "K-BB%": "Home K-BB%"})

        # Format names for display: e.g. Gerrit Cole -> G. Cole
        probables_df = probables_df.with_columns(
            pl.col("Away Pitcher")
            .map_elements(format_short_name, return_dtype=pl.String)
            .alias("Away Pitcher"),
            pl.col("Home Pitcher")
            .map_elements(format_short_name, return_dtype=pl.String)
            .alias("Home Pitcher"),
        )

        # Final column selection and formatting
        display_df = probables_df.select(
            [
                pl.col("Away").alias("Away"),
                pl.col("Away Pitcher").alias("Pitcher (A)"),
                pl.col("Away ERA").alias("ERA (A)"),
                pl.col("Away xERA").alias("xERA (A)"),
                pl.col("Away K-BB%").alias("K-BB% (A)"),
                pl.col("Home").alias("Home"),
                pl.col("Home Pitcher").alias("Pitcher (H)"),
                pl.col("Home ERA").alias("ERA (H)"),
                pl.col("Home xERA").alias("xERA (H)"),
                pl.col("Home K-BB%").alias("K-BB% (H)"),
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
        for col in [
            "ERA (A)",
            "xERA (A)",
            "K-BB% (A)",
            "ERA (H)",
            "xERA (H)",
            "K-BB% (H)",
        ]:
            pd_display[col] = pd_display[col].apply(format_val)

        # Get teams with hitters on our team for highlighting
        batter_data_path = "data/batter_data.csv"
        hitter_map = {}
        if os.path.exists(batter_data_path):
            try:
                hitter_df = (
                    pl.read_csv(batter_data_path)
                    .filter(pl.col("on_team") == True)
                    .select(["Name", "Team"])
                    .unique()
                )
                # Group by team and join names
                hitter_map = {
                    row["Team"]: row["Name"]
                    for row in hitter_df.group_by("Team")
                    .agg(pl.col("Name").str.join(", "))
                    .to_dicts()
                }
            except Exception:
                hitter_map = {}

        # Add hitter info to the display dataframe for tooltips
        pd_display["Away_My_Hitters"] = (
            pd_display["Away"].map(hitter_map.get).fillna("-")
        )
        pd_display["Home_My_Hitters"] = (
            pd_display["Home"].map(hitter_map.get).fillna("-")
        )

        # Create a combined tooltip string
        def get_tooltip(row):
            away_h = row["Away_My_Hitters"]
            home_h = row["Home_My_Hitters"]
            tooltips = []
            if away_h != "-":
                tooltips.append(f"{row['Away']}: {away_h}")
            if home_h != "-":
                tooltips.append(f"{row['Home']}: {home_h}")
            return "\n".join(tooltips) if tooltips else ""

        pd_display["My_Hitters_Tooltip"] = pd_display.apply(get_tooltip, axis=1)

        # Define AgGrid options
        gb = GridOptionsBuilder.from_dataframe(pd_display)
        gb.configure_default_column(
            resizable=True,
            filterable=True,
            sortable=True,
            wrapText=True,
            tooltipField="My_Hitters_Tooltip",
        )

        # Tooltips on Team columns specifically (redundant but explicit)
        gb.configure_column("Away", tooltipField="My_Hitters_Tooltip", minWidth=60)
        gb.configure_column("Home", tooltipField="My_Hitters_Tooltip", minWidth=60)

        # Ensure pitcher columns have enough width
        gb.configure_column("Pitcher (A)", minWidth=130)
        gb.configure_column("Pitcher (H)", minWidth=130)

        # Hide helper columns
        gb.configure_column("Away_My_Hitters", hide=True)
        gb.configure_column("Home_My_Hitters", hide=True)
        gb.configure_column("My_Hitters_Tooltip", hide=True)

        # Style rule for rows with hitters
        cellStyle = JsCode(
            r"""
            function(params) {
                if (params.data.My_Hitters_Tooltip !== "") {
                    return {'background-color': '#a6761d', 'color': 'white'};
                }
                return {};
            }
        """
        )
        # Apply to all columns to "style the row"
        for col in pd_display.columns:
            if col not in ["Away_My_Hitters", "Home_My_Hitters", "My_Hitters_Tooltip"]:
                gb.configure_column(col, cellStyle=cellStyle)

        gridOptions = gb.build()
        gridOptions["domLayout"] = "autoHeight"
        gridOptions["tooltipShowDelay"] = 0
        gridOptions["enableBrowserTooltips"] = True

        # CSS for font size consistency with main page
        css = {".ag-row": {"font-size": "10pt"}, ".ag-header": {"font-size": "10pt"}}

        AgGrid(
            pd_display,
            gridOptions=gridOptions,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            custom_css=css,
        )

    else:
        # Just show the basic schedule if no stats found
        st.write("Schedule found, but pitcher metrics are unavailable.")
        # Format names for display: e.g. Gerrit Cole -> G. Cole
        probables_df = probables_df.with_columns(
            pl.col("Away Pitcher")
            .map_elements(format_short_name, return_dtype=pl.String)
            .alias("Away Pitcher"),
            pl.col("Home Pitcher")
            .map_elements(format_short_name, return_dtype=pl.String)
            .alias("Home Pitcher"),
        )
        st.dataframe(
            probables_df.select(
                [
                    pl.col("Away").alias("Away"),
                    pl.col("Away Pitcher").alias("Pitcher (A)"),
                    pl.col("Home").alias("Home"),
                    pl.col("Home Pitcher").alias("Pitcher (H)"),
                ]
            ),
            hide_index=True,
            width="stretch",
            height="content",
        )
else:
    st.write(f"No games found for {date_str}.")
