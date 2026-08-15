import streamlit as st
import polars as pl
import pandas as pd
import shared
import os
from unidecode import unidecode
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode


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

# Load data from CSVs
team_stats_path = "data/team_stats.csv"
pitcher_stats_path = "data/all_pitcher_stats.csv"
probables_path = "data/todays_probables.csv"


@st.cache_data
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

    # Mapping for dashboard abbreviations
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
        # Calculate ranks
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

        # Join away/home team stats
        for prefix in ["Away", "Home"]:
            stats = team_stats_df.select(
                [
                    pl.col("Team_Abbr"),
                    pl.col("Avg_Runs_For").alias(f"{prefix}_Avg_R"),
                    pl.col("R_Rank").alias(f"{prefix}_R_Rank"),
                    pl.col("wOBA_vs_LHP").alias(f"{prefix}_wOBA_L"),
                    pl.col("wOBA_L_Rank").alias(f"{prefix}_wOBA_L_Rank"),
                    pl.col("wOBA_vs_RHP").alias(f"{prefix}_wOBA_R"),
                    pl.col("wOBA_R_Rank").alias(f"{prefix}_wOBA_R_Rank"),
                    pl.col("Park_Factor").alias(f"{prefix}_Park"),
                    pl.col("Park_Rank").alias(f"{prefix}_Park_Rank"),
                    pl.col("Runs_L10").alias(f"{prefix}_Runs_L10"),
                    pl.col("Runs_L10_Rank").alias(f"{prefix}_Runs_L10_Rank"),
                ]
            )
            probables_df = probables_df.with_columns(
                pl.col(prefix).replace(FG_TO_SC).alias(f"{prefix}_SC")
            )
            probables_df = probables_df.join(
                stats, left_on=f"{prefix}_SC", right_on="Team_Abbr", how="left"
            ).drop(f"{prefix}_SC")

    # Format names
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

    # Highlight Free Agents
    fa_pitchers = set()
    if os.path.exists("data/pitcher_data.csv"):
        try:
            fa_pitchers = set(
                pl.read_csv("data/pitcher_data.csv")
                .filter(pl.col("on_team") == False)
                .select(
                    pl.col("Name").map_elements(normalize_name, return_dtype=pl.String)
                )
                .to_series()
                .to_list()
            )
        except Exception:
            pass
    probables_df = probables_df.with_columns(
        pl.col("norm_away").is_in(fa_pitchers).alias("Away_Is_FA"),
        pl.col("norm_home").is_in(fa_pitchers).alias("Home_Is_FA"),
    )

    # Hitter highlighting
    hitter_map = {}
    if os.path.exists("data/batter_data.csv"):
        try:
            hitter_df = (
                pl.read_csv("data/batter_data.csv")
                .filter(pl.col("on_team") == True)
                .select(["Name", "Team"])
                .unique()
            )
            hitter_map = {
                row["Team"]: row["Name"]
                for row in hitter_df.group_by("Team")
                .agg(pl.col("Name").str.join(", "))
                .to_dicts()
            }
        except Exception:
            pass

    display_cols = [
        "Away",
        "Pitcher (A)",
        "Away ERA",
        "Away xERA",
        "Away K-BB%",
        "Home",
        "Pitcher (H)",
        "Home ERA",
        "Home xERA",
        "Home K-BB%",
    ]
    # Check which "Hand" columns exist to include in display_df for tooltips
    extra_cols = ["Away_Is_FA", "Home_Is_FA"]
    if "Away Hand" in probables_df.columns:
        extra_cols.append("Away Hand")
    if "Home Hand" in probables_df.columns:
        extra_cols.append("Home Hand")

    display_df = probables_df.select(
        [c for c in display_cols if c in probables_df.columns] + extra_cols
    ).fill_null("-")
    pd_display = display_df.to_pandas()

    def get_pitcher_tooltip(row, is_away=True):
        opp_prefix = "Home" if is_away else "Away"
        pitcher_hand = row.get("Away Hand" if is_away else "Home Hand", "")
        opp_abbr = row.get(opp_prefix, "")
        tooltips = []

        def get_ordinal(n):
            try:
                n_int = int(float(n))
                if 11 <= (n_int % 100) <= 13:
                    return f"{n_int}th"
                return f"{n_int}{['th', 'st', 'nd', 'rd', 'th'][min(n_int % 10, 4)]}"
            except:
                return n

        # Pull from probables_df for tooltip info not in display_df
        full_row = probables_df.filter(
            pl.col("Away" if is_away else "Home") == row["Away" if is_away else "Home"]
        ).to_dicts()[0]

        avg_r = full_row.get(f"{opp_prefix}_Avg_R", "-")
        r_rank = full_row.get(f"{opp_prefix}_R_Rank", "-")
        if avg_r != "-":
            tooltips.append(
                f"{opp_abbr} Avg Runs: {float(avg_r):.2f} ({get_ordinal(r_rank)})"
            )

        park = full_row.get("Home_Park", "-")  # Game is always at Home stadium
        park_rank = full_row.get("Home_Park_Rank", "-")
        if park != "-":
            tooltips.append(
                f"Park Factor: {float(park):.2f} ({get_ordinal(park_rank)})"
            )

        woba_l = full_row.get(f"{opp_prefix}_wOBA_L", "-")
        woba_l_rank = full_row.get(f"{opp_prefix}_wOBA_L_Rank", "-")
        woba_r = full_row.get(f"{opp_prefix}_wOBA_R", "-")
        woba_r_rank = full_row.get(f"{opp_prefix}_wOBA_R_Rank", "-")

        if pitcher_hand == "L" and woba_l != "-":
            tooltips.append(
                f"{opp_abbr} wOBA vs LHP: {float(woba_l):.3f} ({get_ordinal(woba_l_rank)})"
            )
        elif pitcher_hand == "R" and woba_r != "-":
            tooltips.append(
                f"{opp_abbr} wOBA vs RHP: {float(woba_r):.3f} ({get_ordinal(woba_r_rank)})"
            )

        hitters = hitter_map.get(opp_abbr, "-")
        if hitters != "-":
            tooltips.append(f"My Hitters: {hitters}")
        return "\n".join(tooltips)

    pd_display["Away_Tooltip"] = pd_display.apply(
        lambda x: get_pitcher_tooltip(x, True), axis=1
    )
    pd_display["Home_Tooltip"] = pd_display.apply(
        lambda x: get_pitcher_tooltip(x, False), axis=1
    )
    pd_display["Row_Tooltip"] = pd_display.apply(
        lambda x: f"Facing My Hitters: {hitter_map.get(x['Away'], '-')}\nFacing My Hitters: {hitter_map.get(x['Home'], '-')}",
        axis=1,
    )

    gb = GridOptionsBuilder.from_dataframe(pd_display)
    gb.configure_default_column(
        resizable=True, filterable=True, sortable=True, minWidth=100
    )

    # Define JS for cell styling (Color coding for Away/Home and Pitcher Status)
    cellStyle = JsCode(r"""
        function(params) {
            let field = params.colDef.field;
            let style = {};
            
            // 1. Base background colors for Home/Away distinction
            if (field.includes('Away') || field.includes('(A)')) {
                style['background-color'] = '#f1f8ff'; // Very light blue for Away
            } else if (field.includes('Home') || field.includes('(H)')) {
                style['background-color'] = '#fff9db'; // Very light yellow for Home
            }

            // 2. Highlighting for Free Agent Pitchers (High Priority)
            if (field === 'Pitcher (A)' && (params.data.Away_Is_FA === true || params.data.Away_Is_FA === 'true')) {
                return {'background-color': '#1b5e20', 'color': 'white'};
            }
            if (field === 'Pitcher (H)' && (params.data.Home_Is_FA === true || params.data.Home_Is_FA === 'true')) {
                return {'background-color': '#1b5e20', 'color': 'white'};
            }

            // 3. Highlight rows where I have hitters facing the pitcher (Medium Priority)
            if (params.data.Row_Tooltip && params.data.Row_Tooltip.includes('Facing My Hitters')) {
                let lines = params.data.Row_Tooltip.split('\n');
                if (lines.length >= 2) {
                    let hittersAway = lines[0].split(': ')[1];
                    let hittersHome = lines[1].split(': ')[1];
                    let isAwayColumn = field.includes('(A)') || field.includes('Away');
                    
                    // If viewing an Away column (A), check if Home team has my hitters (who are facing the A pitcher)
                    if ((isAwayColumn && hittersHome !== '-') || (!isAwayColumn && hittersAway !== '-')) {
                        style['background-color'] = '#a6761d';
                        style['color'] = 'white';
                    }
                }
            }

            return style;
        }
    """)

    # Away Columns
    gb.configure_column(
        "Away", headerName="Team", minWidth=70, flex=1, cellStyle=cellStyle
    )
    gb.configure_column(
        "Pitcher (A)",
        headerName="Pitcher",
        tooltipField="Away_Tooltip",
        minWidth=150,
        flex=2,
        cellStyle=cellStyle,
    )
    gb.configure_column("Away ERA", headerName="ERA", minWidth=80, cellStyle=cellStyle)
    gb.configure_column(
        "Away xERA", headerName="xERA", minWidth=80, cellStyle=cellStyle
    )
    gb.configure_column(
        "Away K-BB%", headerName="K-BB%", minWidth=90, cellStyle=cellStyle
    )

    # Home Columns
    gb.configure_column(
        "Home", headerName="Team", minWidth=70, flex=1, cellStyle=cellStyle
    )
    gb.configure_column(
        "Pitcher (H)",
        headerName="Pitcher",
        tooltipField="Home_Tooltip",
        minWidth=150,
        flex=2,
        cellStyle=cellStyle,
    )
    gb.configure_column("Home ERA", headerName="ERA", minWidth=80, cellStyle=cellStyle)
    gb.configure_column(
        "Home xERA", headerName="xERA", minWidth=80, cellStyle=cellStyle
    )
    gb.configure_column(
        "Home K-BB%", headerName="K-BB%", minWidth=90, cellStyle=cellStyle
    )

    hide_cols = [
        "Away_Tooltip",
        "Home_Tooltip",
        "Row_Tooltip",
        "Away_Is_FA",
        "Home_Is_FA",
    ]
    if "Away Hand" in pd_display.columns:
        hide_cols.append("Away Hand")
    if "Home Hand" in pd_display.columns:
        hide_cols.append("Home Hand")

    for c in hide_cols:
        gb.configure_column(c, hide=True)

    gridOptions = gb.build()
    gridOptions["rowHeight"] = 45
    gridOptions["pagination"] = True
    gridOptions["paginationPageSize"] = 25

    AgGrid(
        pd_display,
        gridOptions=gridOptions,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        height=800,
        theme="alpine",
    )
else:
    st.write(f"No games found for {date_str}.")
