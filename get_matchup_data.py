import polars as pl
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import logging
from typing import Any

# Disable yahoo_oauth logging
oauth_logger = logging.getLogger("yahoo_oauth")
oauth_logger.disabled = True

LEAGUE_NAME = "Save the A's"
SEASON = 2026

STAT_MAP = {
    "7": "R",
    "12": "HR",
    "13": "RBI",
    "16": "SB",
    "3": "AVG",
    "60": "HITS/AB",
    "42": "K",
    "26": "ERA",
    "27": "WHIP",
    "83": "QS",
    "32": "SV",
    "50": "IP",
}


def create_session() -> OAuth2:
    """
    Creates the OAuth2 session from a local json. Refreshes token if necessary.

    :return OAuth2: Active OAuth2 session
    """
    sc = OAuth2(None, None, from_file="oauth.json")
    return sc


def gather_stats(matchup: dict) -> pl.DataFrame:
    """Parses matchup data for the stats for each team, returns as DataFrame.

    Args:
        matchup (dict): Dict object containing info about teams in the matchup

    Returns:
        pl.DataFrame: Stats for each team.
    """
    data_dict: dict[str, list[Any]] = {
        "team": [],
        "R": [],
        "HR": [],
        "RBI": [],
        "SB": [],
        "AVG": [],
        "HITS/AB": [],
        "K": [],
        "ERA": [],
        "WHIP": [],
        "QS": [],
        "SV": [],
        "IP": [],
    }

    for i in ["0", "1"]:
        team_info = matchup["0"]["teams"][i]["team"][0]
        for entry in team_info:
            if isinstance(entry, dict) and entry.get("name", False):
                data_dict["team"].append(entry["name"])
                break

        stats_data = matchup["0"]["teams"][i]["team"][1]
        for entry in stats_data["team_stats"]["stats"]:
            stat_id = entry["stat"]["stat_id"]
            if stat_name := STAT_MAP.get(stat_id):
                stat_value = entry["stat"]["value"]
                data_dict[stat_name].append(stat_value)
            else:
                print(f"Unknown stat ID: {stat_id}, value: {entry['stat']['value']}")

        # Ensure all columns have the same length in case some stats are missing
        max_len = len(data_dict["team"])
        for key in data_dict:
            if len(data_dict[key]) < max_len:
                data_dict[key].append(None)

    df = pl.DataFrame(data_dict)
    return df


def get_my_matchup(league: yfa.League) -> Any:
    """Get details for my current matchup."""
    # league.team_key() returns something like '123.l.456.t.7'
    # But league.matchups() needs to find the one containing this team
    my_team_key = league.team_key()

    # The structure of matchups() output can be complex
    matchups_data = league.matchups()
    try:
        all_matchups = matchups_data["fantasy_content"]["league"][1]["scoreboard"]["0"][
            "matchups"
        ]
    except (KeyError, IndexError):
        # Alternative way to get matchups if the above fails
        # Depending on the version and state of the league
        return None

    for matchup in all_matchups.values():
        if not isinstance(matchup, dict) or "matchup" not in matchup:
            continue

        teams = matchup["matchup"]["0"]["teams"]
        found = False
        for i in ["0", "1"]:
            team_metadata = teams[i]["team"][0]
            for entry in team_metadata:
                if isinstance(entry, dict) and entry.get("team_key") == my_team_key:
                    found = True
                    break
            if found:
                break

        if found:
            return matchup["matchup"]

    return None


def main() -> pl.DataFrame:
    """Uses API data to fetch details for this week's matchup."""

    session = create_session()
    game = yfa.Game(session, "mlb")

    league = None
    for i in game.league_ids():
        lg = yfa.League(session, i)
        league_info = lg.__dict__.get("settings_cache", {})
        if not league_info:
            league_info = lg.settings()

        if (
            league_info.get("name") == LEAGUE_NAME
            and int(league_info.get("season", 0)) == SEASON
        ):
            league = lg
            print(f"Found correct league: {LEAGUE_NAME}")
            break

    if not league:
        print("League not found")
        return pl.DataFrame()

    matchup = get_my_matchup(league)
    if not matchup:
        print("Matchup not found")
        return pl.DataFrame()

    df = gather_stats(matchup)
    with pl.Config(tbl_cols=30):
        print(df)
    # df.write_csv("matchup_data.csv")
    return df


if __name__ == "__main__":
    main()
