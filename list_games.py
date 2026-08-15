import logging
import os
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import json

# Enable logging to see what's happening
logging.basicConfig(level=logging.DEBUG)


def list_accessible_games():
    if not os.path.isfile("oauth.json"):
        print("oauth.json not found")
        return

    sc = OAuth2(None, None, from_file="oauth.json")

    # Force a token validity check/refresh
    if not sc.token_is_valid():
        print("Token is invalid, attempting refresh...")
        sc.refresh_access_token()

    print("Token valid:", sc.token_is_valid())

    # Try to hit the users/games endpoint directly via the handler
    # This is what league_ids and game_id call under the hood
    print("\n--- Attempting to list all games for user ---")
    try:
        # We'll use the raw handler to avoid game-specific logic
        from yahoo_fantasy_api import yhandler

        handler = yhandler.YHandler(sc)

        # This endpoint lists games the user is in
        url = "users;use_login=1/games?format=json"
        response = handler.get(url)
        print("Successfully retrieved user games!")
        print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"Failed to list user games: {e}")

    print("\n--- Attempting to check 'mlb' game status ---")
    try:
        game = yfa.Game(sc, "mlb")
        print(f"Game ID for 'mlb': {game.game_id()}")
    except Exception as e:
        print(f"Failed to check 'mlb' game: {e}")


if __name__ == "__main__":
    list_accessible_games()
