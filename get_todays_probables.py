import requests
import polars as pl
from datetime import datetime
import os


from typing import Optional

# Mapping for MLB API abbreviations to matching ones in the dashboard data (FanGraphs/pybaseball)
TEAM_MAPPING = {
    "AZ": "ARI",
    "CWS": "CHW",
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WSH": "WSN",
}


def get_probables(date_str: Optional[str] = None) -> pl.DataFrame:
    """
    Fetches probable pitchers from MLB Stats API.

    :param str date_str: Date in YYYY-MM-DD format. Defaults to today.
    :return: Polars DataFrame with game info and probable pitchers.
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,person,team"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching data from MLB API: {e}")
        return pl.DataFrame()

    probables = []

    for date_data in data.get("dates", []):
        for game in date_data.get("games", []):
            game_time_utc = game.get("gameDate")
            # Parse UTC time and convert to a more readable format if needed
            try:
                dt = datetime.strptime(game_time_utc, "%Y-%m-%dT%H:%M:%SZ")
                game_time = dt.strftime("%I:%M %p")
            except:
                game_time = game_time_utc

            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})

            away_team_data = away.get("team", {})
            home_team_data = home.get("team", {})

            away_team = away_team_data.get("abbreviation", away_team_data.get("name"))
            away_team = TEAM_MAPPING.get(away_team, away_team)
            away_pitcher_data = away.get("probablePitcher", {})
            away_pitcher = away_pitcher_data.get("fullName", "TBD")
            away_hand = away_pitcher_data.get("pitchHand", {}).get("code", "")

            home_team = home_team_data.get("abbreviation", home_team_data.get("name"))
            home_team = TEAM_MAPPING.get(home_team, home_team)
            home_pitcher_data = home.get("probablePitcher", {})
            home_pitcher = home_pitcher_data.get("fullName", "TBD")
            home_hand = home_pitcher_data.get("pitchHand", {}).get("code", "")

            # Add two rows per game for easier joining later if needed,
            # or keep it in one row for a "matchup" view.
            # Let's do matchup view for now as it's more compact.
            probables.append(
                {
                    "Time": game_time,
                    "Away": away_team,
                    "Away Pitcher": away_pitcher,
                    "Away Hand": away_hand,
                    "Home": home_team,
                    "Home Pitcher": home_pitcher,
                    "Home Hand": home_hand,
                }
            )

    return pl.DataFrame(probables)


def main():
    # Use today's date from environment or system
    # The environment says today is 2026-05-22
    today = datetime.now().strftime("%Y-%m-%d")
    df = get_probables(today)
    if not df.is_empty():
        print(df)
        df.write_csv("todays_probables.csv")


if __name__ == "__main__":
    main()
