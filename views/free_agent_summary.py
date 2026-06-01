import os
import json
import zipfile
from datetime import datetime, timedelta
from typing import Any
import requests
import streamlit as st
import polars as pl
import pandas as pd
import get_matchup_data
import get_todays_probables
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import shared
from unidecode import unidecode


def normalize_name(name: str) -> str:
    if not name or name == "TBD":
        return name
    # Remove accents, convert to lowercase, remove common suffixes
    name = unidecode(name).lower()
    for suffix in [" jr.", " sr.", " iii", " ii", " iv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


## Constants #########################################################################
BATTING_POSITIONS = ["All Batters", "C", "1B", "2B", "3B", "SS", "OF"]
PITCHING_POSITIONS = ["SP", "RP", "All Pitchers"]
ALL_POSITIONS = BATTING_POSITIONS + PITCHING_POSITIONS
ACCENT = "teal"

# Define minimum thresholds for players to meet to be displayed over a given term
THRESHOLDS = {
    "All Batters": {"week": 10, "month": 50, "season": 100},
    "SP": {"week": 3, "month": 10, "season": 50},
    "RP": {"week": 1, "month": 10, "season": 15},
    "All Pitchers": {"week": 1, "month": 10, "season": 25},
}
## End Constants #####################################################################

# st.set_page_config(layout="wide")

# Get data for today's date.
today = shared.get_today_date()
shared.load_data(today.strftime("%Y-%m-%d"))

# Fetch today's probable pitchers for highlighting
probables_df = get_todays_probables.get_probables(today.strftime("%Y-%m-%d"))
if not probables_df.is_empty():
    probables_names = set(
        probables_df["Away Pitcher"]
        .map_elements(normalize_name, return_dtype=pl.String)
        .to_list()
        + probables_df["Home Pitcher"]
        .map_elements(normalize_name, return_dtype=pl.String)
        .to_list()
    )
else:
    probables_names = set()


@st.cache_data
def load_matchup_data() -> pl.DataFrame:
    """Wrapper to fetch and cache matchup data."""
    return get_matchup_data.main()


def style_matchup(df: pd.DataFrame) -> Any:
    """Applies highlighting to the leading team in each category."""

    def color_leading(col):
        if col.name in {"team", "HITS/AB", "IP"}:
            return [""] * len(col)

        # Convert to numeric for comparison, handling potential non-numeric strings
        numeric_col: Any = pd.to_numeric(col, errors="coerce")
        if numeric_col.isna().all():
            return [""] * len(col)

        if col.name in {"ERA", "WHIP"}:
            best_val = numeric_col.min()
        else:
            best_val = numeric_col.max()

        return [
            "background-color: #a6761d; color: white; font-weight: bold"
            if val == best_val and not pd.isna(val)
            else ""
            for val in numeric_col
        ]

    return df.style.apply(color_leading)


# Set title
st.markdown(
    """
    # Interesting Free Agents
    """
)

# Fetch and display matchup data
st.markdown("### This Week's Matchup")
matchup_df = load_matchup_data()
if not matchup_df.is_empty():
    styled_matchup = style_matchup(matchup_df.to_pandas())
    st.dataframe(styled_matchup, hide_index=True, width="stretch")
else:
    st.write("Matchup data not found.")


# Add the position selector to left column...

chosen_position = st.selectbox(
    label="Position:",
    options=ALL_POSITIONS,
)

# ... and term selector on the right
chosen_term = st.radio(
    label="Chosen term:",
    options=["Last Week", "Last Month", "Full Season"],
    index=0,
    horizontal=True,
)

# Load the correct CSV for chosen position
if chosen_position in BATTING_POSITIONS:
    df = pl.read_csv("data/batter_data.csv")
    # May arise if a player was just called up, their only position will be 'Util'
    # Just put them as OF
    df = df.with_columns(pl.col("Position(s)").fill_null(value="OF"))
    if chosen_position != "All Batters":
        df = df.filter(pl.col("Position(s)").str.contains(chosen_position))
    # Rename HardHit% and present in full percentages
    # df = df.with_columns(
    #    (pl.col('HardHit%') * 100).alias('HH%')
    # )

# For the pitchers DataFrames, scale K-BB% up to be a raw percentage
elif chosen_position in PITCHING_POSITIONS:
    df = pl.read_csv("data/pitcher_data.csv")
    if chosen_position != "All Pitchers":
        df = df.filter(pl.col("Position(s)").str.contains(chosen_position))
else:
    df = pl.DataFrame()


########################################################################################
##  Begin Table ########################################################################
########################################################################################

# Table only wants data from the chosen term
term = chosen_term.split(" ")[-1].lower()
table_df = df.filter(pl.col("term") == term)

# Apply minimum thresholds for players not on our team
if chosen_position in BATTING_POSITIONS:
    table_df = table_df.filter(
        (pl.col("ABs").ge(THRESHOLDS["All Batters"][term]))
        | (pl.col("on_team") is True)
    )
else:
    table_df = table_df.filter(
        (pl.col("IP").ge(THRESHOLDS[chosen_position][term]))
        | (pl.col("on_team") is True)
    )

display_number = 25 if "All" in chosen_position else 15

table_df = table_df.sort(by=["on_team", "Rank"], descending=[True, False]).head(
    display_number
)

# Don't want to include every single column from the DataFrame. Choose specific columns
# based on whether we're dealing with hitters or pitchers
if chosen_position in BATTING_POSITIONS:
    table_df = table_df[
        [
            "Name",
            "Position(s)",
            "Team",
            "ABs",
            "AVG",
            "HRs",
            "RBIs",
            "Runs",
            "SBs",
            "wRC+",
            "xwOBA",
            "Rank",
            "on_team",
        ]
    ]
else:
    table_df = table_df[
        [
            "Name",
            "Position(s)",
            "Team",
            "IP",
            "ERA",
            "WHIP",
            "Ks",
            "QS",
            "SVs",
            "xERA",
            "K-BB%",
            "Rank",
            "on_team",
        ]
    ]

# Add pitching_today column for pitchers
if chosen_position in PITCHING_POSITIONS:
    table_df = table_df.with_columns(
        pl.col("Name")
        .map_elements(
            lambda x: normalize_name(x) in probables_names, return_dtype=pl.Boolean
        )
        .alias("pitching_today")
    )
else:
    table_df = table_df.with_columns(pl.lit(False).alias("pitching_today"))

# Formats name, e.g. Bo Bichette -> B. Bichette
table_df = table_df.with_columns(
    pl.col("Name").map_elements(
        lambda x: f"{x[0]}. {' '.join(x.split(' ')[1:])}", return_dtype=pl.String
    )
)

table_df = table_df.rename({"Position(s)": "Pos."})

# Define certain columns which can be smaller by default
small_cols = [
    "ABs",
    "Team",
    "IPs",
    "HRs",
    "RBIs",
    "Runs",
    "SBs",
    "Ks",
    "QS",
    "SVs",
    "Rank",
]

# Define column options for each column we want to include
columnDefs = [
    {
        "field": col,
        "headerName": col,
        "type": "rightAligned",
        "width": 13 if col in small_cols else 40,
        "height": 20,
        "sortable": True,
        "sortingOrder": ["desc", "asc", None],
    }
    for col in list(table_df.columns)
    if col not in ["on_team", "pitching_today"]
]

# Format the decimal numbers for certain metrics
for colDef in columnDefs:
    if colDef["field"] in {"AVG", "xwOBA"}:
        colDef["type"] = ["numericColumn", "customNumericFormat"]
        colDef["precision"] = 3
    elif colDef["field"] in {"ERA", "WHIP", "xERA"}:
        colDef["type"] = ["numericColumn", "customNumericFormat"]
        colDef["precision"] = 2
    elif colDef["field"] in {"K-BB%", "HH%"}:
        colDef["type"] = ["numericColumn", "customNumericFormat"]
        colDef["precision"] = 1

# Set the name column (always the first one) to be left-aligned
columnDefs[0]["type"] = "leftAligned"
columnDefs[0]["width"] = 70

# Second column (either ABs or IPs) and last column can also be smaller
columnDefs[1]["width"] = 10

# Define CSS rule to color the rows for every player on our team.
cellStyle = JsCode(
    r"""
function(cellClassParams) {
		if (cellClassParams.data.pitching_today) {
				return {'background-color': '#1b5e20', 'color': 'white'}
		}
		if (cellClassParams.data.on_team) {
				return {'background-color': '#a6761d'}
		}
		return {};
}
"""
)

# Define the font size for the table
css = {".ag-row": {"font-size": "10pt"}, ".ag-header": {"font-size": "10pt"}}

grid_builder = GridOptionsBuilder.from_dataframe(table_df.to_pandas())
grid_options = grid_builder.build()

# Add the cell style rule to each column
grid_options["defaultColDef"]["cellStyle"] = cellStyle
# Set height/width of columns automatically
grid_options["defaultColDef"]["autoHeight"] = True
grid_options["defaultColDef"]["autoWidth"] = True

grid_options["columnDefs"] = columnDefs

# Add the table to our dashboard
AgGrid(
    table_df.to_pandas(),
    gridOptions=grid_options,
    allow_unsafe_jscode=True,
    fit_columns_on_grid_load=True,
    custom_css=css,
    height=485,
)

########################################################################################
##  End Table ##########################################################################
########################################################################################

########################################################################################
##  Begin Plot #########################################################################
########################################################################################

# Define different x/y values and domains based on whether we're looking at hitters
# or pitchers. Also define a column, `size_encoding`, that will be used to determine
# the size of the markers on the plot.
if chosen_position in BATTING_POSITIONS:
    x_val = "xwOBA"
    y_val = "wRC+"
    x_domain = [0.2, 0.45]
    y_domain = [50, 200]
else:
    x_val = "xERA"
    y_val = "K-BB%"
    x_domain = [5.5, 0.5]
    y_domain = [0, 50.0]

# Create a specific DF for the plot object
plot_df = df.filter(pl.col("term") == term)

# Apply minimum thresholds
if chosen_position in BATTING_POSITIONS:
    plot_df = plot_df.filter(
        (pl.col("ABs").ge(THRESHOLDS["All Batters"][term]))
        | (pl.col("on_team") is True)
    )
    plot_df = plot_df.with_columns(pl.lit(1.4).alias("label_size"))
# else:
#    plot_df = plot_df.filter((pl.col('IP').ge(THRESHOLDS[chosen_position][term])) | (pl.col('on_team') is True))
#    plot_df = plot_df.with_columns(
#        pl.lit(1).alias('label_size')
#    )
#
display_number = 25 if "All" in chosen_position else 15

plot_df = plot_df.sort(by=["on_team", "Rank"], descending=[True, False]).head(
    display_number
)

# Create a last name column to use for chart labels
plot_df = plot_df.with_columns(
    pl.col("Name")
    .map_elements(lambda x: " ".join(x.split(" ")[1:]), return_dtype=pl.String)
    .alias("last_name")
)

# Plot breaks if some values are extreme i guess
if chosen_position not in BATTING_POSITIONS:
    plot_df = plot_df.filter(pl.col("xERA").le(5.5))

# Creates a scatter plot of players for each position, highlighting players on our team
chart = (
    alt.Chart(plot_df, width=300, height=600)
    .mark_circle()
    .encode(
        color=alt.Color("on_team").scale(
            scheme="dark2", reverse=True, domain=[False, True]
        ),
        tooltip=["Name", "Team", "Rank", y_val, x_val, "Position(s)"],
        x=alt.X(x_val, scale=alt.Scale(domain=x_domain)),
        y=alt.Y(y_val, scale=alt.Scale(domain=y_domain)),
    )
)

# Create the corresponding labels for each player to add to the plot
labels = chart.mark_text(align="left", dx=9, dy=9, fontSize=14, limit=100).encode(
    text="last_name",
)
st.altair_chart(chart + labels)

########################################################################################
##  End Plot ###########################################################################
########################################################################################

########################################################################################
## End Main Script #####################################################################
########################################################################################
