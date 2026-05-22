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
    df = df.filter(pl.col("HITS/AB") != "/")

    return df


def get_aggregated_stats(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Aggregates weekly stats into totals/weighted averages based on timeframe.

    :param pl.DataFrame df: Raw weekly stats DataFrame
    :param str timeframe: 'Last 2 Weeks' or 'Full Season'
    :return pl.DataFrame: Aggregated stats per team
    """
    if df.is_empty():
        return df

    latest_week = int(df.select(pl.col("week").max()).item())

    # Parse HITS/AB into Hits and ABs for proper AVG calculation
    # And convert IP to decimal for weighted averages
    df = df.with_columns(
        [
            pl.col("HITS/AB").str.split("/").list.get(0).cast(pl.Int64).alias("_H"),
            pl.col("HITS/AB").str.split("/").list.get(1).cast(pl.Int64).alias("_AB"),
            (pl.col("IP").floor() + (pl.col("IP") % 1) * 10 / 3).alias("_IP_dec"),
        ]
    )

    if timeframe == "Last 2 Weeks":
        df = df.filter(pl.col("week") >= latest_week - 1)

    # Aggregate stats by team
    agg_df = df.group_by("team").agg(
        [
            pl.col("R").sum().cast(pl.Float64),
            pl.col("HR").sum().cast(pl.Float64),
            pl.col("RBI").sum().cast(pl.Float64),
            pl.col("SB").sum().cast(pl.Float64),
            (pl.col("_H").sum() / pl.col("_AB").sum().cast(pl.Float64)).alias("AVG"),
            pl.col("K").sum().cast(pl.Float64),
            ((pl.col("ERA") * pl.col("_IP_dec")).sum() / pl.col("_IP_dec").sum()).alias(
                "ERA"
            ),
            (
                (pl.col("WHIP") * pl.col("_IP_dec")).sum() / pl.col("_IP_dec").sum()
            ).alias("WHIP"),
            pl.col("QS").sum().cast(pl.Float64),
            pl.col("SV").sum().cast(pl.Float64),
            pl.col("_IP_dec").sum().alias("_total_ip_dec"),
        ]
    )

    # Convert total IP back to Yahoo format (e.g., 33.33 -> 33.1)
    agg_df = agg_df.with_columns(
        (
            pl.col("_total_ip_dec").floor()
            + (pl.col("_total_ip_dec") % 1 * 3).round(0) / 10
        ).alias("IP")
    ).drop("_total_ip_dec")

    return agg_df


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
