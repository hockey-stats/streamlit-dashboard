import streamlit as st
import polars as pl
import pandas as pd
import shared
import os
from unidecode import unidecode
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode


st.markdown("# Today's Probable Pitchers (Public View)")
st.caption(
    "Matchup analysis using public MLB and Statcast data. 'wOBA' is the opposing team's performance against the pitcher's handedness. 'L10' refers to the average runs over the last 10 games for that specific team (Away L10 for the Away team, Home L10 for the Home team)."
)


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

# 1. Load Data from CSVs (Fast)
team_stats_path = "data/team_stats.csv"
pitcher_stats_path = "data/all_pitcher_stats.csv"
probables_path = "data/todays_probables.csv"


def load_local_data():
    t_df = (
        pl.read_csv(team_stats_path)
        if os.path.exists(team_stats_path)
        else pl.DataFrame()
    )
    p_df = (
        pl.read_csv(pitcher_stats_path)
        if os.path.exists(pitcher_stats_path)
        else pl.DataFrame()
    )
    prob_df = (
        pl.read_csv(probables_path)
        if os.path.exists(probables_path)
        else pl.DataFrame()
    )
    return t_df, p_df, prob_df


team_stats_df, pitcher_stats, probables_df = load_local_data()

if team_stats_df.is_empty():
    st.warning(
        "Team statistics (wOBA, L10, Park Factor) not found. Displaying basic matchup data."
    )

if not probables_df.is_empty():
    if not pitcher_stats.is_empty():
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

    # Mapping for dashboard abbreviations to Statcast ones used in get_team_stats
    FG_TO_SC = {
        "ARI": "ARI",
        "ATL": "ATL",
        "BAL": "BAL",
        "BOS": "BOS",
        "CHC": "CHC",
        "CHW": "CWS",
        "CIN": "CIN",
        "CLE": "CLE",
        "COL": "COL",
        "DET": "DET",
        "HOU": "HOU",
        "KCR": "KC",
        "LAA": "LAA",
        "LAD": "LAD",
        "MIA": "MIA",
        "MIL": "MIL",
        "MIN": "MIN",
        "NYM": "NYM",
        "NYY": "NYY",
        "OAK": "OAK",
        "PHI": "PHI",
        "PIT": "PIT",
        "SDP": "SD",
        "SEA": "SEA",
        "SFG": "SF",
        "STL": "STL",
        "TBR": "TB",
        "TEX": "TEX",
        "TOR": "TOR",
        "WSN": "WSH",
    }

    if not team_stats_df.is_empty():
        # Calculate ranks (higher is better = descending)
        team_stats_df = team_stats_df.with_columns(
            [
                pl.col("Avg_Runs_For")
                .rank(descending=True, method="min")
                .cast(pl.Int32)
                .alias("R_Rank"),
                pl.col("wOBA_vs_LHP")
                .rank(descending=True, method="min")
                .cast(pl.Int32)
                .alias("wOBA_L_Rank"),
                pl.col("wOBA_vs_RHP")
                .rank(descending=True, method="min")
                .cast(pl.Int32)
                .alias("wOBA_R_Rank"),
                pl.col("Park_Factor")
                .rank(descending=True, method="min")
                .cast(pl.Int32)
                .alias("Park_Rank"),
                pl.col("Runs_L10")
                .rank(descending=True, method="min")
                .cast(pl.Int32)
                .alias("Runs_L10_Rank"),
            ]
        )

        # Join away team stats
        away_team_stats = team_stats_df.select(
            [
                pl.col("Team_Abbr"),
                pl.col("Avg_Runs_For").alias("Away_Avg_R"),
                pl.col("R_Rank").alias("Away_R_Rank"),
                pl.col("wOBA_vs_LHP").alias("Away_wOBA_L"),
                pl.col("wOBA_L_Rank").alias("Away_wOBA_L_Rank"),
                pl.col("wOBA_vs_RHP").alias("Away_wOBA_R"),
                pl.col("wOBA_R_Rank").alias("Away_wOBA_R_Rank"),
                pl.col("Park_Factor").alias("Away_Park"),
                pl.col("Runs_L10").alias("Away_Runs_L10"),
            ]
        )
        probables_df = probables_df.with_columns(
            pl.col("Away").replace(FG_TO_SC).alias("Away_SC")
        )
        probables_df = probables_df.join(
            away_team_stats, left_on="Away_SC", right_on="Team_Abbr", how="left"
        ).drop("Away_SC")

        # Join home team stats
        home_team_stats = team_stats_df.select(
            [
                pl.col("Team_Abbr"),
                pl.col("Avg_Runs_For").alias("Home_Avg_R"),
                pl.col("R_Rank").alias("Home_R_Rank"),
                pl.col("wOBA_vs_LHP").alias("Home_wOBA_L"),
                pl.col("wOBA_L_Rank").alias("Home_wOBA_L_Rank"),
                pl.col("wOBA_vs_RHP").alias("Home_wOBA_R"),
                pl.col("wOBA_R_Rank").alias("Home_wOBA_R_Rank"),
                pl.col("Park_Factor").alias("Home_Park"),
                pl.col("Runs_L10").alias("Home_Runs_L10"),
            ]
        )
        probables_df = probables_df.with_columns(
            pl.col("Home").replace(FG_TO_SC).alias("Home_SC")
        )
        probables_df = probables_df.join(
            home_team_stats, left_on="Home_SC", right_on="Team_Abbr", how="left"
        ).drop("Home_SC")

    # Add Matchup Metric columns (Re-enabled)
    if "Away Hand" in probables_df.columns and "Home_wOBA_L" in probables_df.columns:
        probables_df = probables_df.with_columns(
            pl.when(pl.col("Away Hand") == "L")
            .then(pl.col("Home_wOBA_L"))
            .otherwise(pl.col("Home_wOBA_R"))
            .alias("Opp wOBA (A)"),
            pl.when(pl.col("Home Hand") == "L")
            .then(pl.col("Away_wOBA_L"))
            .otherwise(pl.col("Away_wOBA_R"))
            .alias("Opp wOBA (H)"),
        )

    # Format names for display
    # Handle cases where Hand columns are missing
    for prefix in ["Away", "Home"]:
        p_col = f"{prefix} Pitcher"
        h_col = f"{prefix} Hand"
        alias_col = f"Pitcher ({prefix[0]})"

        if h_col in probables_df.columns:
            probables_df = probables_df.with_columns(
                pl.struct([p_col, h_col])
                .map_elements(
                    lambda x,
                    pc=p_col,
                    hc=h_col: f"{format_short_name(x[pc])} ({x[hc]})"
                    if x[hc]
                    else format_short_name(x[pc]),
                    return_dtype=pl.String,
                )
                .alias(alias_col)
            )
        else:
            probables_df = probables_df.with_columns(
                pl.col(p_col)
                .map_elements(format_short_name, return_dtype=pl.String)
                .alias(alias_col)
            )

    # Final column selection
    display_cols = [
        "Away",
        "Pitcher (A)",
        "Away ERA",
        "Away xERA",
        "Away K-BB%",
        "Opp wOBA (A)",
        "Away_Runs_L10",
        "Home",
        "Pitcher (H)",
        "Home ERA",
        "Home xERA",
        "Home K-BB%",
        "Opp wOBA (H)",
        "Home_Runs_L10",
        "Home_Park",
    ]

    # Ensure all display columns exist in the dataframe, filling with "-" if missing
    existing_cols = probables_df.columns
    for col in display_cols:
        if col not in existing_cols:
            probables_df = probables_df.with_columns(pl.lit("-").alias(col))

    # Re-verify columns after additions
    display_df = probables_df.select(display_cols).fill_null("-")
    pd_display = display_df.to_pandas()

    def format_val(val, fmt="{:.2f}"):
        try:
            if val == "-" or val == "Data loading...":
                return val
            return fmt.format(float(val))
        except:
            return val

    for col in pd_display.columns:
        if any(x in col for x in ["ERA", "xERA", "Park"]):
            pd_display[col] = pd_display[col].apply(lambda x: format_val(x))
        if "wOBA" in col:
            pd_display[col] = pd_display[col].apply(lambda x: format_val(x, "{:.3f}"))
        if "K-BB%" in col:
            pd_display[col] = pd_display[col].apply(lambda x: format_val(x, "{:.1f}%"))

    def get_matchup_tooltip(row, is_away=True):
        opp_prefix = "Home" if is_away else "Away"
        opp_abbr = row[opp_prefix]
        tooltips = []
        avg_r = row.get(f"{opp_prefix}_Avg_R", "-")
        runs_l10 = row.get(f"{opp_prefix}_Runs_L10", "-")
        if avg_r != "-" and avg_r != "Data loading...":
            tooltips.append(f"{opp_abbr} Season Avg Runs: {format_val(avg_r)}")
            tooltips.append(f"{opp_abbr} Runs (L10): {runs_l10}")
        park = row.get("Home_Park", "-")
        if park != "-":
            tooltips.append(f"Stadium Park Factor: {format_val(park)}")
        return "\n".join(tooltips)

    pd_display["Away_Tooltip"] = pd_display.apply(
        lambda x: get_matchup_tooltip(x, True), axis=1
    )
    pd_display["Home_Tooltip"] = pd_display.apply(
        lambda x: get_matchup_tooltip(x, False), axis=1
    )

    gb = GridOptionsBuilder.from_dataframe(pd_display)
    gb.configure_default_column(
        resizable=True, filterable=True, sortable=True, minWidth=70
    )

    # Define JS for cell styling (Color coding for Away/Home and wOBA)
    cellStyle = JsCode("""
        function(params) {
            let field = params.colDef.field;
            let style = {};
            
            // Base background colors for Home/Away distinction
            if (field.includes('Away') || field.includes('(A)')) {
                style['background-color'] = '#f1f8ff'; // Very light blue for Away
            } else if (field.includes('Home') || field.includes('(H)') || field === 'Home_Park') {
                style['background-color'] = '#fff9db'; // Very light yellow for Home
            }

            // Highlighting for wOBA performance
            if (field.includes('Opp wOBA')) {
                let val = parseFloat(params.value);
                if (!isNaN(val)) {
                    if (val < 0.290) {
                        style['background-color'] = '#1b5e20'; 
                        style['color'] = 'white';
                    } else if (val > 0.350) {
                        style['background-color'] = '#b71c1c';
                        style['color'] = 'white';
                    }
                }
            }
            return style;
        }
    """)

    # Away Columns
    gb.configure_column(
        "Away", headerName="Team", minWidth=60, flex=1, cellStyle=cellStyle
    )
    gb.configure_column(
        "Pitcher (A)",
        headerName="Pitcher",
        tooltipField="Away_Tooltip",
        minWidth=120,
        flex=2,
        cellStyle=cellStyle,
    )
    gb.configure_column("Away ERA", headerName="ERA", minWidth=60, cellStyle=cellStyle)
    gb.configure_column(
        "Away xERA", headerName="xERA", minWidth=60, cellStyle=cellStyle
    )
    gb.configure_column(
        "Away K-BB%", headerName="K-BB%", minWidth=70, cellStyle=cellStyle
    )
    gb.configure_column(
        "Opp wOBA (A)", headerName="wOBA", minWidth=80, cellStyle=cellStyle
    )
    gb.configure_column(
        "Away_Runs_L10",
        headerName="L10",
        minWidth=45,
        cellStyle=cellStyle,
        filter=False,
    )

    # Home Columns
    gb.configure_column(
        "Home", headerName="Team", minWidth=60, flex=1, cellStyle=cellStyle
    )
    gb.configure_column(
        "Pitcher (H)",
        headerName="Pitcher",
        tooltipField="Home_Tooltip",
        minWidth=120,
        flex=2,
        cellStyle=cellStyle,
    )
    gb.configure_column("Home ERA", headerName="ERA", minWidth=60, cellStyle=cellStyle)
    gb.configure_column(
        "Home xERA", headerName="xERA", minWidth=60, cellStyle=cellStyle
    )
    gb.configure_column(
        "Home K-BB%", headerName="K-BB%", minWidth=70, cellStyle=cellStyle
    )
    gb.configure_column(
        "Opp wOBA (H)", headerName="wOBA", minWidth=80, cellStyle=cellStyle
    )
    gb.configure_column(
        "Home_Runs_L10",
        headerName="L10",
        minWidth=45,
        cellStyle=cellStyle,
        filter=False,
    )
    gb.configure_column(
        "Home_Park", headerName="Park", minWidth=70, cellStyle=cellStyle
    )

    gb.configure_column("Away_Tooltip", hide=True)
    gb.configure_column("Home_Tooltip", hide=True)

    gridOptions = gb.build()

    gridOptions["rowHeight"] = 28
    gridOptions["headerHeight"] = 32
    gridOptions["pagination"] = True
    gridOptions["paginationPageSize"] = 15

    # CSS for font size consistency
    css = {
        ".ag-row": {"font-size": "8.5pt"},
        ".ag-header-cell-text": {"font-size": "8.5pt"},
    }

    AgGrid(
        pd_display,
        gridOptions=gridOptions,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        height=460,
        width="100%",
        theme="alpine",
        custom_css=css,
    )
else:
    st.write(f"No games found for {date_str}.")
