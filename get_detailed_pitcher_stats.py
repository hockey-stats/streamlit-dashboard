import pybaseball as pyb
import polars as pl
from typing import Dict, Tuple, Any
from unidecode import unidecode


def get_team_name(lev: str, tm: str) -> str:
    """
    Maps Baseball-Reference league and city strings to the team's nickname.

    :param str lev: The league level string (e.g., 'Maj-AL').
    :param str tm: The team city or list of cities (e.g., 'Chicago').
    :return str: The mapped team nickname (e.g., 'White Sox').
    """
    current_city: str = tm.split(",")[-1].strip()

    mapping: Dict[Tuple[str, str], str] = {
        ("Maj-AL", "Chicago"): "White Sox",
        ("Maj-NL", "Chicago"): "Cubs",
        ("Maj-AL", "New York"): "Yankees",
        ("Maj-NL", "New York"): "Mets",
        ("Maj-AL", "Los Angeles"): "Angels",
        ("Maj-NL", "Los Angeles"): "Dodgers",
        ("Maj-AL", "Houston"): "Astros",
        ("Maj-AL", "Detroit"): "Tigers",
        ("Maj-NL", "Philadelphia"): "Phillies",
        ("Maj-AL", "Baltimore"): "Orioles",
        ("Maj-AL", "Toronto"): "Blue Jays",
        ("Maj-NL", "Atlanta"): "Braves",
        ("Maj-NL", "Arizona"): "Diamondbacks",
        ("Maj-AL", "Tampa Bay"): "Rays",
        ("Maj-NL", "Pittsburgh"): "Pirates",
        ("Maj-AL", "Seattle"): "Mariners",
        ("Maj-NL", "San Francisco"): "Giants",
        ("Maj-AL", "Athletics"): "Athletics",
        ("Maj-AL", "Cleveland"): "Guardians",
        ("Maj-NL", "San Diego"): "Padres",
        ("Maj-AL", "Boston"): "Red Sox",
        ("Maj-NL", "Milwaukee"): "Brewers",
        ("Maj-AL", "Minnesota"): "Twins",
        ("Maj-AL", "Kansas City"): "Royals",
        ("Maj-NL", "Cincinnati"): "Reds",
        ("Maj-NL", "Miami"): "Marlins",
        ("Maj-NL", "Colorado"): "Rockies",
        ("Maj-AL", "Texas"): "Rangers",
        ("Maj-NL", "Washington"): "Nationals",
        ("Maj-NL", "St. Louis"): "Cardinals",
    }

    return mapping.get((lev, current_city), current_city)


def get_fg_abbreviation(row: Dict[str, Any]) -> str:
    """
    Maps Baseball-Reference 'Lev' and 'Tm' to FanGraphs 3-letter abbreviations.
    Handles multi-team cities using the League (Lev) as a differentiator.

    :param Dict[str, Any] row: A dictionary representing a DataFrame row.
    :return str: The 3-letter FanGraphs team abbreviation.
    """
    city: str = row["Tm"].split(",")[-1].strip()

    mapping: Dict[Tuple[str, str], str] = {
        ("Maj-AL", "Chicago"): "CHW",
        ("Maj-NL", "Chicago"): "CHC",
        ("Maj-AL", "New York"): "NYY",
        ("Maj-NL", "New York"): "NYM",
        ("Maj-AL", "Los Angeles"): "LAA",
        ("Maj-NL", "Los Angeles"): "LAD",
        ("Maj-AL", "Baltimore"): "BAL",
        ("Maj-AL", "Boston"): "BOS",
        ("Maj-AL", "Cleveland"): "CLE",
        ("Maj-AL", "Detroit"): "DET",
        ("Maj-AL", "Houston"): "HOU",
        ("Maj-AL", "Kansas City"): "KCR",
        ("Maj-AL", "Minnesota"): "MIN",
        ("Maj-AL", "Oakland"): "ATH",
        ("Maj-AL", "Athletics"): "ATH",
        ("Maj-AL", "Seattle"): "SEA",
        ("Maj-AL", "Tampa Bay"): "TBR",
        ("Maj-AL", "Texas"): "TEX",
        ("Maj-AL", "Toronto"): "TOR",
        ("Maj-NL", "Arizona"): "ARI",
        ("Maj-NL", "Atlanta"): "ATL",
        ("Maj-NL", "Cincinnati"): "CIN",
        ("Maj-NL", "Colorado"): "COL",
        ("Maj-NL", "Miami"): "MIA",
        ("Maj-NL", "Milwaukee"): "MIL",
        ("Maj-NL", "Philadelphia"): "PHI",
        ("Maj-NL", "Pittsburgh"): "PIT",
        ("Maj-NL", "San Diego"): "SDP",
        ("Maj-NL", "San Francisco"): "SFG",
        ("Maj-NL", "St. Louis"): "STL",
        ("Maj-NL", "Washington"): "WSN",
    }

    return mapping.get((row["Lev"], city), city)


def get_detailed_pitcher_stats(year: int) -> pl.DataFrame:
    """
    Gets pitching stats from Baseball Reference and calculates K-BB%,
    also adds xERA from statcast expected stats.

    :param int year: The year for which to compute data.
    :return pl.DataFrame: DataFrame with robust data for each pitcher.
    """
    df: pl.DataFrame = pl.from_pandas(pyb.pitching_stats_bref(year))

    df = df.filter(pl.col("IP") > 0)

    xdf: pl.DataFrame = pl.from_pandas(
        pyb.statcast_pitcher_expected_stats(year=year, minPA=1)
    )
    xdf = xdf[["player_id", "xera"]]
    xdf = xdf.cast({'player_id': pl.String})

    df = df.rename({"mlbID": "player_id"})

    df = df.with_columns(((pl.col("SO") - pl.col("BB")) / pl.col("BF")).alias("K-BB%"))

    final_df: pl.DataFrame = df.join(xdf, how="inner", on="player_id")

    final_df = final_df.with_columns(
        pl.col("xera").round_sig_figs(3).alias("xERA"),
        pl.col("K-BB%").round_sig_figs(3),
        pl.struct(pl.all()).map_elements(get_fg_abbreviation, return_dtype=pl.String).alias('Team'),
        pl.col('Name').map_elements(
            lambda x: x.encode('latin-1').decode('unicode_escape').encode('latin-1').decode('utf-8'), 
            return_dtype=pl.String).alias('Name')
    )

    final_df = final_df.rename(
        {
            "player_id": "playerID",
        }
    )

    final_df = final_df[
        [
            "Name",
            "playerID",
            "Team",
            "IP",
            "ERA",
            "WHIP",
            "SO",
            "BB",
            "SV",
            "K-BB%",
            "xERA",
        ]
    ]

    return final_df


if __name__ == "__main__":
    results_df: pl.DataFrame = get_detailed_pitcher_stats(2026)
    with pl.Config(tbl_cols=20, tbl_rows=50):
        print(results_df.sort(by=pl.col("xERA")))

