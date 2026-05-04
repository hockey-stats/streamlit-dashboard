import polars as pl
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import logging
from typing import Any

from get_free_agent_summary import create_session
from get_matchup_data import STAT_MAP, LEAGUE_NAME, SEASON

# Disable yahoo_oauth logging
oauth_logger = logging.getLogger("yahoo_oauth")
oauth_logger.disabled = True


def gather_matchup_stats(matchup: dict, week: int) -> list[dict]:
    """Parses matchup data for the stats for each team.

    Returns a list of dicts, one for each team in the matchup.
    """
    stats_list = []

    # matchup['matchup']['0']['teams'] contains '0' and '1'
    teams = matchup["0"]["teams"]

    for i in ["0", "1"]:
        team_data = {"week": week, "team": None}

        # Team info
        team_info = teams[i]["team"][0]
        for entry in team_info:
            if isinstance(entry, dict) and entry.get("name", False):
                team_data["team"] = entry["name"]
                break

        # Stats
        stats_entries = teams[i]["team"][1]["team_stats"]["stats"]
        for entry in stats_entries:
            stat_id = entry["stat"]["stat_id"]
            if stat_name := STAT_MAP.get(stat_id):
                stat_value = entry["stat"]["value"]
                # Convert to numeric if possible
                try:
                    if stat_name == "AVG":
                        team_data[stat_name] = float(stat_value)
                    elif stat_name in {"ERA", "WHIP"}:
                        team_data[stat_name] = float(stat_value)
                    elif stat_name == "HITS/AB":
                        team_data[stat_name] = stat_value
                    elif stat_name == "IP":
                        team_data[stat_name] = float(stat_value)
                    else:
                        team_data[stat_name] = int(stat_value)
                except (ValueError, TypeError):
                    team_data[stat_name] = stat_value

        stats_list.append(team_data)

    return stats_list


def get_league_weekly_stats(league: yfa.League) -> pl.DataFrame:
    current_week = league.current_week()
    all_stats = []

    # We want completed weeks, so up to current_week - 1
    # Actually, user might want current week too? "shows how each team is doing"
    # Let's get up to current_week.

    for week in range(1, current_week + 1):
        print(f"Fetching stats for week {week}...")
        matchups_data = league.matchups(week=week)
        try:
            matchups = matchups_data["fantasy_content"]["league"][1]["scoreboard"]["0"][
                "matchups"
            ]
        except (KeyError, IndexError):
            print(f"No matchups found for week {week}")
            continue

        for matchup_key in matchups:
            if matchup_key == "count":
                continue
            matchup = matchups[matchup_key]["matchup"]
            all_stats.extend(gather_matchup_stats(matchup, week))

    if not all_stats:
        return pl.DataFrame()

    df = pl.DataFrame(all_stats)

    # Drop weeks that have no stats yet
    df = df.filter(pl.col('HITS/AB') != '/')

    return df


def run() -> pl.DataFrame:
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
            break

    if not league:
        return pl.DataFrame()

    return get_league_weekly_stats(league)


if __name__ == "__main__":
    df = run()
    if not df.is_empty():
        df.write_csv("data/league_weekly_stats.csv")

