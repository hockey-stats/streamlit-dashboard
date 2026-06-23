import streamlit as st
import polars as pl
import pandas as pd
import get_todays_probables
import shared
import os
from unidecode import unidecode
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from mlb_stats_util.get_team_stats import get_team_stats


@st.cache_data(ttl=43200)
def get_cached_team_stats(year: int):
    return get_team_stats(year)


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

# Get team stats for enrichment
year = today.year

# Initialize session state for team stats if not present
if "team_stats_df" not in st.session_state:
    st.session_state.team_stats_df = None

team_stats_df = st.session_state.team_stats_df

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

        if team_stats_df is not None and not team_stats_df.is_empty():
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
                    pl.col("Park_Rank").alias("Away_Park_Rank"),
                    pl.col("Runs_L10").alias("Away_Runs_L10"),
                    pl.col("Runs_L10_Rank").alias("Away_Runs_L10_Rank"),
                ]
            )
            # Create a temporary column for joining
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
                    pl.col("Park_Rank").alias("Home_Park_Rank"),
                    pl.col("Runs_L10").alias("Home_Runs_L10"),
                    pl.col("Runs_L10_Rank").alias("Home_Runs_L10_Rank"),
                ]
            )
            probables_df = probables_df.with_columns(
                pl.col("Home").replace(FG_TO_SC).alias("Home_SC")
            )
            probables_df = probables_df.join(
                home_team_stats, left_on="Home_SC", right_on="Team_Abbr", how="left"
            ).drop("Home_SC")
        else:
            # Initialize with loading indicator or empty
            loading_val = (
                "Data loading..." if st.session_state.team_stats_df is None else "-"
            )
            probables_df = probables_df.with_columns(
                pl.lit(loading_val).alias("Away_Avg_R"),
                pl.lit(loading_val).alias("Away_R_Rank"),
                pl.lit(loading_val).alias("Away_Runs_L10"),
                pl.lit(loading_val).alias("Away_Runs_L10_Rank"),
                pl.lit(loading_val).alias("Away_wOBA_L"),
                pl.lit(loading_val).alias("Away_wOBA_L_Rank"),
                pl.lit(loading_val).alias("Away_wOBA_R"),
                pl.lit(loading_val).alias("Away_wOBA_R_Rank"),
                pl.lit(loading_val).alias("Away_Park"),
                pl.lit(loading_val).alias("Away_Park_Rank"),
                pl.lit(loading_val).alias("Home_Avg_R"),
                pl.lit(loading_val).alias("Home_R_Rank"),
                pl.lit(loading_val).alias("Home_Runs_L10"),
                pl.lit(loading_val).alias("Home_Runs_L10_Rank"),
                pl.lit(loading_val).alias("Home_wOBA_L"),
                pl.lit(loading_val).alias("Home_wOBA_L_Rank"),
                pl.lit(loading_val).alias("Home_wOBA_R"),
                pl.lit(loading_val).alias("Home_wOBA_R_Rank"),
                pl.lit(loading_val).alias("Home_Park"),
                pl.lit(loading_val).alias("Home_Park_Rank"),
            )

        # Format names for display: e.g. Gerrit Cole -> G. Cole (R)
        probables_df = probables_df.with_columns(
            pl.struct(["Away Pitcher", "Away Hand"])
            .map_elements(
                lambda x: f"{format_short_name(x['Away Pitcher'])} ({x['Away Hand']})"
                if x["Away Hand"]
                else format_short_name(x["Away Pitcher"]),
                return_dtype=pl.String,
            )
            .alias("Away Pitcher"),
            pl.struct(["Home Pitcher", "Home Hand"])
            .map_elements(
                lambda x: f"{format_short_name(x['Home Pitcher'])} ({x['Home Hand']})"
                if x["Home Hand"]
                else format_short_name(x["Home Pitcher"]),
                return_dtype=pl.String,
            )
            .alias("Home Pitcher"),
        )

        # Get free agent pitchers for highlighting
        fa_pitchers = set()
        if os.path.exists("data/pitcher_data.csv"):
            try:
                fa_pitchers = set(
                    pl.read_csv("data/pitcher_data.csv")
                    .filter(pl.col("on_team") == False)
                    .select(
                        pl.col("Name").map_elements(
                            normalize_name, return_dtype=pl.String
                        )
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

        # Final column selection and formatting
        # Ensure new team stats columns are selected if they exist
        cols_to_select = [
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
            pl.col("Away_Is_FA"),
            pl.col("Home_Is_FA"),
            pl.col("Away Hand"),
            pl.col("Home Hand"),
        ]

        # Add team stats columns to selection
        for c in [
            "Away_Avg_R",
            "Away_R_Rank",
            "Away_Runs_L10",
            "Away_Runs_L10_Rank",
            "Away_wOBA_L",
            "Away_wOBA_L_Rank",
            "Away_wOBA_R",
            "Away_wOBA_R_Rank",
            "Away_Park",
            "Away_Park_Rank",
            "Home_Avg_R",
            "Home_R_Rank",
            "Home_Runs_L10",
            "Home_Runs_L10_Rank",
            "Home_wOBA_L",
            "Home_wOBA_L_Rank",
            "Home_wOBA_R",
            "Home_wOBA_R_Rank",
            "Home_Park",
            "Home_Park_Rank",
        ]:
            if c in probables_df.columns:
                cols_to_select.append(pl.col(c))

        display_df = probables_df.select(cols_to_select).fill_null("-")

        pd_display = display_df.to_pandas()

        def format_metric(val, is_percent=False):
            try:
                if val == "-":
                    return val
                f_val = float(val)
                if is_percent:
                    return f"{f_val:.1f}%"
                return f"{f_val:.2f}"
            except:
                return val

        for col in ["ERA (A)", "xERA (A)", "ERA (H)", "xERA (H)"]:
            pd_display[col] = pd_display[col].apply(
                lambda x: format_metric(x, is_percent=False)
            )

        for col in ["K-BB% (A)", "K-BB% (H)"]:
            pd_display[col] = pd_display[col].apply(
                lambda x: format_metric(x, is_percent=True)
            )

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
        def get_hitter_tooltip(row):
            away_h = row["Away_My_Hitters"]
            home_h = row["Home_My_Hitters"]
            tooltips = []
            if away_h != "-":
                tooltips.append(f"{row['Away']}: {away_h}")
            if home_h != "-":
                tooltips.append(f"{row['Home']}: {home_h}")
            return "\n".join(tooltips) if tooltips else ""

        def get_pitcher_tooltip(row, is_away=True):
            opp_prefix = "Home" if is_away else "Away"
            opp_abbr = row[opp_prefix]

            # Get pitcher hand for the pitcher we are highlighting
            pitcher_prefix = "Away" if is_away else "Home"
            pitcher_hand = row.get(f"{pitcher_prefix} Hand", "")

            tooltips = []

            def get_ordinal(n):
                if n == "-":
                    return n
                try:
                    n_int = int(float(n))
                    if 11 <= (n_int % 100) <= 13:
                        suffix = "th"
                    else:
                        suffix = ["th", "st", "nd", "rd", "th"][min(n_int % 10, 4)]
                    return f"{n_int}{suffix}"
                except:
                    return n

            # Add team metrics for Opponent
            avg_r = row.get(f"{opp_prefix}_Avg_R", "-")
            r_rank = row.get(f"{opp_prefix}_R_Rank", "-")
            runs_l10 = row.get(f"{opp_prefix}_Runs_L10", "-")
            runs_l10_rank = row.get(f"{opp_prefix}_Runs_L10_Rank", "-")
            woba_l = row.get(f"{opp_prefix}_wOBA_L", "-")
            woba_l_rank = row.get(f"{opp_prefix}_wOBA_L_Rank", "-")
            woba_r = row.get(f"{opp_prefix}_wOBA_R", "-")
            woba_r_rank = row.get(f"{opp_prefix}_wOBA_R_Rank", "-")

            # Always use Home_Park for the stadium factor (game is at Home stadium)
            park = row.get("Home_Park", "-")
            park_rank = row.get("Home_Park_Rank", "-")

            if avg_r != "-":
                if avg_r == "Data loading...":
                    tooltips.append("Team metrics: Data loading...")
                else:
                    try:
                        tooltips.append(
                            f"{opp_abbr} Avg Runs: {float(avg_r):.2f} ({get_ordinal(r_rank)})"
                        )
                        tooltips.append(
                            f"Runs (L10): {int(float(runs_l10))} ({get_ordinal(runs_l10_rank)})"
                        )

                        # Decide which wOBA to show based on pitcher hand
                        if pitcher_hand == "L":
                            tooltips.append(
                                f"{opp_abbr} wOBA vs LHP: {float(woba_l):.3f} ({get_ordinal(woba_l_rank)})"
                            )
                        elif pitcher_hand == "R":
                            tooltips.append(
                                f"{opp_abbr} wOBA vs RHP: {float(woba_r):.3f} ({get_ordinal(woba_r_rank)})"
                            )
                        else:
                            # If hand unknown, show both
                            tooltips.append(
                                f"{opp_abbr} wOBA vs L: {float(woba_l):.3f} ({get_ordinal(woba_l_rank)})"
                            )
                            tooltips.append(
                                f"{opp_abbr} wOBA vs R: {float(woba_r):.3f} ({get_ordinal(woba_r_rank)})"
                            )

                        tooltips.append(
                            f"Park Factor: {float(park):.2f} ({get_ordinal(park_rank)})"
                        )
                    except (ValueError, TypeError):
                        tooltips.append(f"{opp_abbr} Avg Runs: {avg_r}")
                        tooltips.append(f"Park Factor: {park}")

            # Add hitter info if any for the OPPONENT team (who are facing this pitcher)
            hitters = row[f"{opp_prefix}_My_Hitters"]
            if hitters != "-":
                if tooltips:
                    tooltips.append("-" * 20)
                tooltips.append(f"My Hitters: {hitters}")

            return "\n".join(tooltips) if tooltips else ""

        pd_display["My_Hitters_Tooltip"] = pd_display.apply(get_hitter_tooltip, axis=1)
        pd_display["Away_Pitcher_Tooltip"] = pd_display.apply(
            lambda x: get_pitcher_tooltip(x, is_away=True), axis=1
        )
        pd_display["Home_Pitcher_Tooltip"] = pd_display.apply(
            lambda x: get_pitcher_tooltip(x, is_away=False), axis=1
        )

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

        # Specific tooltips for Pitcher columns
        gb.configure_column(
            "Pitcher (A)", tooltipField="Away_Pitcher_Tooltip", minWidth=130
        )
        gb.configure_column(
            "Pitcher (H)", tooltipField="Home_Pitcher_Tooltip", minWidth=130
        )

        # Ensure other columns have enough width
        # (already set Pitcher widths above)

        # Hide helper columns
        gb.configure_column("Away_My_Hitters", hide=True)
        gb.configure_column("Home_My_Hitters", hide=True)
        gb.configure_column("My_Hitters_Tooltip", hide=True)
        gb.configure_column("Away_Pitcher_Tooltip", hide=True)
        gb.configure_column("Home_Pitcher_Tooltip", hide=True)
        gb.configure_column("Away_Is_FA", hide=True)
        gb.configure_column("Home_Is_FA", hide=True)
        gb.configure_column("Away Hand", hide=True)
        gb.configure_column("Home Hand", hide=True)
        for c in [
            "Away_Avg_R",
            "Away_R_Rank",
            "Away_wOBA_L",
            "Away_wOBA_L_Rank",
            "Away_wOBA_R",
            "Away_wOBA_R_Rank",
            "Away_Park",
            "Away_Park_Rank",
            "Home_Avg_R",
            "Home_R_Rank",
            "Home_wOBA_L",
            "Home_wOBA_L_Rank",
            "Home_wOBA_R",
            "Home_wOBA_R_Rank",
            "Home_Park",
            "Home_Park_Rank",
        ]:
            if c in pd_display.columns:
                gb.configure_column(c, hide=True)

        # Style rule for rows with hitters or free agent pitchers
        cellStyle = JsCode(
            r"""
            function(params) {
                if (params.colDef.field === 'Pitcher (A)' && (params.data.Away_Is_FA === true || params.data.Away_Is_FA === 'true')) {
                    return {'background-color': '#1b5e20', 'color': 'white'};
                }
                if (params.colDef.field === 'Pitcher (H)' && (params.data.Home_Is_FA === true || params.data.Home_Is_FA === 'true')) {
                    return {'background-color': '#1b5e20', 'color': 'white'};
                }
                if (params.data.My_Hitters_Tooltip !== "") {
                    return {'background-color': '#a6761d', 'color': 'white'};
                }
                return {};
            }
        """
        )
        # Apply to all columns to "style the row"
        cols_to_skip = [
            "Away_My_Hitters",
            "Home_My_Hitters",
            "My_Hitters_Tooltip",
            "Away_Pitcher_Tooltip",
            "Home_Pitcher_Tooltip",
            "Away Hand",
            "Home Hand",
            "Away_Avg_R",
            "Away_R_Rank",
            "Away_Runs_L10",
            "Away_Runs_L10_Rank",
            "Away_wOBA_L",
            "Away_wOBA_L_Rank",
            "Away_wOBA_R",
            "Away_wOBA_R_Rank",
            "Away_Park",
            "Away_Park_Rank",
            "Home_Avg_R",
            "Home_R_Rank",
            "Home_Runs_L10",
            "Home_Runs_L10_Rank",
            "Home_wOBA_L",
            "Home_wOBA_L_Rank",
            "Home_wOBA_R",
            "Home_wOBA_R_Rank",
            "Home_Park",
            "Home_Park_Rank",
        ]
        for col in pd_display.columns:
            if col not in cols_to_skip:
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

        # After showing the grid, if team stats are not yet loaded, load them and rerun
        if st.session_state.team_stats_df is None:
            # We don't want to show a global spinner that blocks the whole page
            # but we need to fetch the data.
            # In a view script, this will run after the grid is rendered.
            year = today.year
            try:
                stats = get_cached_team_stats(year)
            except Exception:
                try:
                    stats = get_cached_team_stats(year - 1)
                except Exception:
                    stats = pl.DataFrame()

            st.session_state.team_stats_df = stats
            st.rerun()

    else:
        # Just show the basic schedule if no stats found
        st.write("Schedule found, but pitcher metrics are unavailable.")
        # Format names for display: e.g. Gerrit Cole -> G. Cole (R)
        probables_df = probables_df.with_columns(
            pl.struct(["Away Pitcher", "Away Hand"])
            .map_elements(
                lambda x: f"{format_short_name(x['Away Pitcher'])} ({x['Away Hand']})"
                if x["Away Hand"]
                else format_short_name(x["Away Pitcher"]),
                return_dtype=pl.String,
            )
            .alias("Away Pitcher"),
            pl.struct(["Home Pitcher", "Home Hand"])
            .map_elements(
                lambda x: f"{format_short_name(x['Home Pitcher'])} ({x['Home Hand']})"
                if x["Home Hand"]
                else format_short_name(x["Home Pitcher"]),
                return_dtype=pl.String,
            )
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
            use_container_width=True,
        )
else:
    st.write(f"No games found for {date_str}.")
