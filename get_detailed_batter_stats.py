import pybaseball as pyb
import polars as pl
from typing import Dict, Tuple, Any

def get_park_factor(team: str) -> float:
    """
    Returns the park factor for a specific team, normalized to a 1.0 baseline.

    :param str team: The nickname of the MLB team (e.g., 'Astros').
    :return float: The park factor as a decimal (e.g., 0.994).
    """
    mapping: Dict[str, float] = {
        'Angels': 101.23049020767212,
        'Astros': 99.48140382766724,
        'Athletics': 102.86766290664673,
        'Blue Jays': 99.45313930511475,
        'Braves': 100.12708902359009,
        'Brewers': 98.89779090881348,
        'Cardinals': 97.5001335144043,
        'Cubs': 97.8569507598877,
        'Diamondbacks': 100.64197778701782,
        'Dodgers': 99.1439938545227,
        'Giants': 97.25400805473328,
        'Guardians': 98.87371063232422,
        'Mariners': 93.54020357131958,
        'Marlins': 100.99986791610718,
        'Mets': 96.34352326393127,
        'Nationals': 99.63456988334656,
        'Orioles': 98.61453771591187,
        'Padres': 95.90458273887634,
        'Phillies': 101.27016305923462,
        'Pirates': 101.54402256011963,
        'Rangers': 98.6534059047699,
        'Rays': 100.93531608581543,
        'Red Sox': 104.24087047576904,
        'Reds': 104.54981327056885,
        'Rockies': 113.34958076477051,
        'Royals': 103.06445360183716,
        'Tigers': 100.30543804168701,
        'Twins': 100.81133842468262,
        'White Sox': 100.31185150146484,
        'Yankees': 98.9298939704895
    }

    return mapping[team] / 100


def get_team_name(lev: str, tm: str) -> str:
    """
    Maps Baseball-Reference league and city strings to the team's nickname.

    :param str lev: The league level string (e.g., 'Maj-AL').
    :param str tm: The team city or list of cities (e.g., 'Chicago').
    :return str: The mapped team nickname (e.g., 'White Sox').
    """
    # Handle multi-team strings like 'Chicago,Houston' by taking the final team
    current_city: str = tm.split(',')[-1].strip()

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
    # Clean the team name (handles 'Chicago,Houston' by taking the last team)
    city: str = row['Tm'].split(',')[-1].strip()

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
        ("Maj-NL", "Washington"): "WSN"
    }

    return mapping.get((row['Lev'], city), city)


def calculate_woba(row: Dict[str, Any]) -> float:
    """
    Calculates the Weighted On-Base Average (wOBA) for a given hitter.

    :param Dict[str, Any] row: A dictionary representing a DataFrame row with counting stats.
    :return float: The calculated wOBA value.
    """
    ubb: float = row['BB'] - row['IBB']
    singles: float = row['H'] - row['2B'] - row['3B'] - row['HR']
    
    # Constants
    wBB: float = 0.709
    wHBP: float = 0.740
    w1B: float = 0.904
    w2B: float = 1.281
    w3B: float = 1.620
    wHR: float = 2.080

    wOBA: float = ((wBB * ubb) + (wHBP * row['HBP']) + (w1B * singles) + (w2B * row['2B']) + \
            (w3B * row['3B']) + (wHR * row['HR'])) / (row['AB'] + ubb + row['SF'] + row['HBP'])

    return wOBA


def calculate_wrcplus(row: Dict[str, Any]) -> float:
    """
    Row-wise function which calculates wRC+ for each hitter. 

    :param Dict[str, Any] row: Each row of the DataFrame containing hitter info.
    :return float: The calculated wRC+.
    """
    # Constants supplied from Fangraphs Guts
    wOBAScale: float = 1.275
    avgwOBA: float = 0.320
    runsPerPA: float = 0.118
    runsPerWin: float = 9.851

    wRAA: float = ((row['wOBA'] - avgwOBA) / wOBAScale) * row['PA']

    player_team: str = get_team_name(row['Lev'], row['Tm'])
    parkFactor: float = get_park_factor(player_team)

    wRC: float = wRAA + (runsPerPA * row['PA'])

    wRC_p: float = (((wRC / row['PA']) / runsPerPA) / parkFactor * 100)

    return wRC_p


def get_detailed_batter_stats(year: int) -> pl.DataFrame:
    """
    Gets basic hitting stats from bref and calculates wRC+, also adds xWOBA from statcast.
    
    :param int year: The year for which to compute data.
    :return pl.DataFrame: DataFrame with robust data for each hitter.
    """
    df: pl.DataFrame = pl.from_pandas(pyb.batting_stats_bref(year))

    # Ignore players with 0 plate appearances
    df = df.filter(pl.col('PA') > 0)

    df = df.with_columns(
            pl.struct(pl.all()).map_elements(calculate_woba, return_dtype=pl.Float64).alias('wOBA')
        )

    pl.Config(tbl_rows=100, tbl_cols=40)

    df = df.with_columns(
            pl.struct(pl.all()).map_elements(
                lambda row: calculate_wrcplus(row),
                return_dtype=pl.Float64).alias('wRC+')
        )

    xdf: pl.DataFrame = pl.from_pandas(pyb.statcast_batter_expected_stats(year=year, minPA=1))
    xdf = xdf[['player_id', 'est_woba']]

    df = df.rename({'mlbID': 'player_id'})

    final_df: pl.DataFrame = df.join(xdf, how='inner', on='player_id')

    final_df = final_df.with_columns(
            pl.struct(pl.all()).map_elements(get_fg_abbreviation, return_dtype=pl.String).alias('Team'),
            pl.col('wRC+').round_sig_figs(3).cast(pl.Int32),
            pl.col('Name').map_elements(
                lambda x: x.encode('latin-1').decode('unicode_escape').encode('latin-1').decode('utf-8'), 
                return_dtype=pl.String).alias('Name')
        )

    final_df = final_df.rename({
        "est_woba": "xwOBA",
        "player_id": "playerID",
        "BA": "AVG"
        })

    final_df = final_df[['Name', 'playerID', 'Team', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB', 'AVG', 'OBP', 'OPS', 'wRC+', 'xwOBA']]

    return final_df


if __name__ == '__main__':
    results_df: pl.DataFrame = get_detailed_batter_stats(2026)
    print(results_df.sort(by=pl.col('wRC+')))

