import polars as pl
from mlb_stats_util.get_team_stats import get_team_stats
from mlb_stats_util import get_detailed_pitcher_stats
from get_todays_probables import get_probables
from datetime import datetime
import os


def main():
    today = datetime.now()
    year = today.year
    date_str = today.strftime("%Y-%m-%d")

    # 1. Fetch Team Stats (The slow query)
    print("Fetching Team Stats (wOBA, Park Factors, etc.)...")
    try:
        team_stats = get_team_stats(year)
        if not team_stats.is_empty():
            team_stats.write_csv("team_stats.csv")
            print(f"Saved team_stats.csv ({len(team_stats)} teams)")
    except Exception as e:
        print(f"Error fetching team stats: {e}")

    # 2. Fetch Detailed Pitcher Stats
    print("Fetching Detailed Pitcher Stats...")
    try:
        pitcher_stats = get_detailed_pitcher_stats(year)
        if not pitcher_stats.is_empty():
            pitcher_stats.write_csv("all_pitcher_stats.csv")
            print(f"Saved all_pitcher_stats.csv ({len(pitcher_stats)} pitchers)")
    except Exception as e:
        print(f"Error fetching pitcher stats: {e}")

    # 3. Fetch Today's Probables
    print("Fetching Today's Probables...")
    try:
        probables = get_probables(date_str)
        if not probables.is_empty():
            probables.write_csv("todays_probables.csv")
            print(f"Saved todays_probables.csv ({len(probables)} games)")
            print(f"Columns in probables: {probables.columns}")
    except Exception as e:
        print(f"Error fetching probables: {e}")

    print("--- Data Extraction Complete ---")


if __name__ == "__main__":
    main()
