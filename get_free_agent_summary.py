import logging
from itertools import combinations

import polars as pl
import pybaseball as pb
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
from unidecode import unidecode

from util.fix_traded_mlb_players import fix_teams_for_traded_batters, fix_teams_for_traded_pitchers

oauth_logger = logging.getLogger('yahoo_oauth')
oauth_logger.disabled = True


# All the abbreviations that differ between the two data sources
TEAM_MAPPING = {
    'SF': 'SFG',
    'SD': 'SDP',
    'TB': 'TBR',
    'CWS': 'CHW',
    'AZ': 'ARI',
    'KC': 'KCR',
    'WSH': 'WSN',
}

# All the positions that we will be collecting data for
POSITIONS = ['1B', '2B', '3B', 'SS', 'C', 'OF', 'SP', 'RP']


def create_session() -> OAuth2:
    """
    Creates the OAuth2 session from a local json. Refreshes token if necessary.

    :return OAuth2: Active OAuth2 session
    """
    sc = OAuth2(None, None, from_file='oauth.json')
    return sc


def standardize_team_names(team: str) -> str:
    """
    Function used to adjust team abbreviations provided by the Yahoo API to match
    those used by pybaseball.

    :param str team: The team name abbreviation provided by the Yahoo API
    :return str: Team name abbreviation used by pybaseball
    """
    return TEAM_MAPPING.get(team, team)


def get_free_agents(position: str, league: yfa.League) -> pl.DataFrame:
    """
    Uses the Yahoo Fantasy API to get a list of all free agents for the given
    position then formats them into a returned DataFrame.

    :param str position: Position for which to pull free agents.
    :param yfa.League league: The League object from the yfa library, representing our league.
    :return pd.DataFrame: DataFrame containing formatted data for free agents.
    """

    players = league.free_agents(position)

    # Get the ID for every returned player, we use these IDs to query more detailed info
    player_ids = [int(player['player_id']) for player in players if player['status'] == '']

    if position in {'1B', '2B', '3B', 'C', 'OF', 'SS'}:
        df = collect_batter_stats(player_ids, league)
    else:
        df = collect_pitcher_stats(player_ids, league, position)

    return df


def get_players_from_own_team(position: str, league: yfa.League, session: OAuth2) -> pl.DataFrame:
    """
    Pulls from the fantasy API all players of the given position from my own team.

    :param str position: Position for which to pull players.
    :param yfa.League league: The League object from the yfa library, representing our league.
    :param OAuth2 session: OAuth2 session used for authentication.
    :return pl.DataFrame: DataFrame containing the players and their stats.
    """
    # Get all teams and find one owned by the caller of the API
    teams = league.teams()
    for team in teams:
        if teams[team].get('is_owned_by_current_login', False):
            my_team = yfa.Team(session, team)
            break
    else:
        print("Own team not found, exiting...")
        raise ValueError

    # Collect only the players of the given position
    players = []
    for player in my_team.roster():
        # Ignore players on the IL
        if 'IL' in player['status']:
            continue
        if position in player['eligible_positions']:
            players.append(player['player_id'])

    if position in {'1B', '2B', '3B', 'C', 'OF', 'SS', 'Util'}:
        df = collect_batter_stats(players, league)
    else:
        df = collect_pitcher_stats(players, league, position)

    return df


def collect_pitcher_stats(player_ids: list, league: yfa.League, position: str) -> pl.DataFrame:
    """
    Given a list of player IDs for pitchers, pull the given stats and organize into a DF.

    :param list(str) player_ids: List of player IDs given as strings.
    :param yfa.League league: The League object to query against.
    :param str position: SP or RP, since we filter the results slightly differently for starters
    :return pl.DataFrame: Details and stats for each player.
    """

    player_details = league.player_details(player_ids)

    # Collect stats across the past week, month, and season
    player_stats_week = league.player_stats(player_ids, 'lastweek')
    player_stats_month = league.player_stats(player_ids, 'lastmonth')
    player_stats_season = league.player_stats(player_ids, 'season')

    p_dict = {
        "name": [],
        "id": [],
        "team": [],
        "positions": [],
        "ip": [],
        "era": [],
        "whip": [],
        "k": [],
        "w": [],
        "sv": [],
        "term": []
    }

    for player_stats, term in zip([player_stats_week, player_stats_month, player_stats_season],
                                  ['week', 'month', 'season']):
        for details, stats in zip(player_details, player_stats):
            try:
                if stats['IP'] == 0.0 or stats['IP'] == '-':
                    # Skip players without any innings pitched
                    continue
            except KeyError as e:
                print(details, stats)
                raise e
            p_dict['name'].append(unidecode(details['name']['full']))
            p_dict['id'].append(stats['player_id'])
            p_dict['team'].append(standardize_team_names(details['editorial_team_abbr']))
            p_dict['positions'].append(','.join([pos['position'] for pos in \
                                                 details['eligible_positions'] \
                                                 if pos['position'] != 'P']))
            p_dict['ip'].append(stats['IP'])
            p_dict['era'].append(stats['ERA'])
            p_dict['whip'].append(stats['WHIP'])
            p_dict['k'].append(int(stats['K']))
            p_dict['w'].append(int(stats['W']))
            p_dict['sv'].append(int(stats['SV']))
            p_dict['term'].append(term)

    # Collect yahoo stats into DF
    y_df = pl.DataFrame(p_dict)

    # Now get full-season stats with pybaseball
    p_df = pb.pitching_stats(2025, qual=5)[['Name', 'Team', 'K-BB%', 'xERA', 'Stuff+', 'G', 'GS']]
    p_df['team'] = p_df['Team']
    del p_df['Team']

    # Update teams for traded players
    p_df = fix_teams_for_traded_pitchers(p_df)

    # And convert back to polars
    p_df = pl.from_pandas(p_df)
    p_df = p_df.rename({"Name": "name"})

    if position == 'SP':
        # If looking for starters, remove every pitcher with 0 starts
        p_df = p_df.remove(pl.col("GS") == 0)
    elif position == 'RP':
        # If looking for relievers, remove every pitcher with as many starts as appearances
        p_df = p_df.remove(pl.col("GS") == pl.col("G"))

    # Remove the games/starts column
    p_df = p_df.drop("G", "GS")

    # Add a last_name column to both dataframes for the join (different sources might have
    # different first names)
    p_df = p_df.with_columns(
        (pl.col("name").str.split(" ").list.slice(1, None).list.join(" ")).alias("last_name")
    )
    y_df = y_df.with_columns(
        (pl.col("name").str.split(" ").list.slice(1, None).list.join(" ")).alias("last_name")
    )

    # Join the two DFs and return
    df = y_df.join(p_df, how='inner', on=['last_name', 'team'])
    df = df.drop("last_name", "name_right")

    return df


def collect_batter_stats(player_ids: list, league: yfa.League) -> pl.DataFrame:
    """
    Given a list of player IDs for batters, pull the given stats and organize into a DF.

    :param list(str) player_ids: List of player IDs given as strings.
    :param yfa.League league: The League object to query against.
    :return pl.DataFrame: Details and stats for each player.
    """

    player_details = league.player_details(player_ids)

    # Collect stats across the past week, month, and season
    player_stats_week = league.player_stats(player_ids, 'lastweek')
    player_stats_month = league.player_stats(player_ids, 'lastmonth')
    player_stats_season = league.player_stats(player_ids, 'season')

    p_dict = {
        "name": [],
        "id": [],
        "team": [],
        "positions": [],
        "r": [],
        "ab": [],
        "avg": [],
        "hr": [],
        "rbi": [],
        "sb": [],
        "term": []
    }

    for player_stats, term in zip([player_stats_week, player_stats_month, player_stats_season],
                                  ['week', 'month', 'season']):
        for details, stats in zip(player_details, player_stats):
            try:
                if stats['AVG'] == '-':
                    # Skip players without any at-bats.
                    continue
            except KeyError as e:
                print(details, stats)
                raise e

            p_dict['name'].append(unidecode(details['name']['full']))
            p_dict['id'].append(stats['player_id'])
            p_dict['team'].append(standardize_team_names(details['editorial_team_abbr']))
            p_dict['positions'].append(','.join([pos['position'] \
                                                for pos in details['eligible_positions'] \
                                                if pos['position'] != 'Util']))
            p_dict['ab'].append(int(stats['H/AB'].split('/')[-1]))
            p_dict['r'].append(int(stats['R']))
            p_dict['avg'].append(stats['AVG'])
            p_dict['hr'].append(int(stats['HR']))
            p_dict['rbi'].append(int(stats['RBI']))
            p_dict['sb'].append(int(stats['SB']))
            p_dict['term'].append(term)

    # Collect yahoo stats into DF
    y_df = pl.DataFrame(p_dict)

    # Now get full-season stats with pybaseball
    p_df = pb.batting_stats(2025, qual=20)[['Name', 'Team', 'wRC+', 'xwOBA', 'HardHit%']]
    p_df['team'] = p_df['Team']
    del p_df['Team']

    # Update teams for traded players
    p_df = fix_teams_for_traded_batters(p_df)

    # Now convert to polars and rename columns
    p_df = pl.from_pandas(p_df)
    p_df = p_df.rename({"Name": "name"})

    p_df = p_df.with_columns(pl.col('xwOBA').cast(pl.Decimal(10, 3)))

    # Add a last_name column to both dataframes for the join (different sources might have
    # different versions of first names)
    p_df = p_df.with_columns(
        (pl.col("name").str.split(" ").list.slice(1, None).list.join(" ")).alias("last_name")
    )
    y_df = y_df.with_columns(
        (pl.col("name").str.split(" ").list.slice(1, None).list.join(" ")).alias("last_name")
    )

    # Join the two DFs and return
    df = y_df.join(p_df, how='inner', on=['last_name', 'team'])
    df = df.drop("name_right")
    df = df.drop("last_name")

    return df


# Currently not being used. TODO: Remove?
def filter_free_agents(fa_df: pl.DataFrame, t_df: pl.DataFrame, position: str) -> pl.DataFrame:
    """
    Given DFs containing stats for both the free agents and players on our team, convert the
    averages for each metric for players on our team and filter out every free agent that
    isn't above-average in at least one category.

    Hitters need to be above average in 2 of the given stats to appear, whereas pitchers need
    3.
    
    These comparisons are done only for stats across the last month.

    :param pl.DataFrame fa_df: DF containing free agent data
    :param pl.DataFrame t_df: DF containing data from players on our team
    :param str position: The position we're comparing
    :return pl.DataFrame: DF containing all players that met the filter conditions, as well 
                          as players on our team.
    """

    if position in {'1B', '2B', '3B', 'C', 'OF', 'SS'}:
        stats = ['avg', 'rbi', 'hr', 'wRC+', 'xwOBA']
        combo_num = 2
    else:
        stats = ['k', 'era', 'K-BB%', 'Stuff+']
        combo_num = 4 if position == 'RP' else 3

    base_fa_df = fa_df.clone()

    # Filter both dataframes to only contain stats from this past month
    t_df = t_df.filter(pl.col('term') == 'month')
    fa_df = fa_df.filter(pl.col('term') == 'month')

    # Get the averages among players on our team
    avg = {}
    for stat in stats:
        avg[stat]= float(t_df.select(pl.mean(stat)).item())

    final_df = pl.DataFrame()
    for stat_combo in list(combinations(stats, combo_num)):
        filtered_df = fa_df.clone()
        for stat in stat_combo:
            if stat in {'era', 'xERA'}:
                filtered_df = filtered_df.filter(pl.col(stat) <= avg[stat])
            else:
                filtered_df = filtered_df.filter(pl.col(stat) >= avg[stat])
        if not final_df.is_empty():
            final_df = pl.concat([final_df, filtered_df])
        else:
            final_df = filtered_df.clone()

    # If looking at relievers, include anyone with a save recently
    if position == 'RP':
        filtered_df = fa_df.clone()
        filtered_df = filtered_df.filter(pl.col('sv') > 2)
        final_df = pl.concat([final_df, filtered_df])

    names_to_include = list(set(final_df['name']))

    final_df = base_fa_df.filter(pl.col('name').is_in(names_to_include))
    final_df = final_df.unique(subset=['name', 'team', 'term'], maintain_order=True)

    return final_df


def filter_taken(df: pl.DataFrame, free_agents: list[int], my_team: list[int]) -> pl.DataFrame:
    """
    Removes from a player DF every player that's on a team which is not my team, and then adds 
    a column for boolean values corresponding to whether or not a player is on my team.

    Uses player_ids returned by the Yahoo Fantasy API for filtering.

    :param pl.DataFrame df: DataFrame containing stats for all players
    :param set[int] free_agents: Set of IDs of players which are free agents
    :param set[int] my_team: Set of IDs of players on my team.
    :return pl.DataFrame: DataFrame with taken players removed and 'on_team' column added.
    """

    # Filter out players that are already taken
    df = df.filter(pl.col('id').is_in(free_agents + my_team))

    # Also filter out any injured players
    df = df.filter(pl.col('positions').str.contains('IL').not_())

    # Set 'on_team' to True if player is on my team, False otherwise
    df = df.with_columns(
        pl.when(pl.col('id').is_in(my_team))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias('on_team')
    )

    return df


def compute_z_scores(batter_df: pl.DataFrame, player_type: str) -> pl.DataFrame:

    """
    Computes z-scores for each column in the DataFrame and returns one with an extra column
    for the average of these z-scores for each term.

    :param pl.DataFrame batter_df: Raw statistics for each player.
    :param str player_type: Either 'batters' or 'pitchers'
    :return pl.DataFrame: Provided DataFrame with z-scores added in.
    """
    # Dict to store DataFrames for each term
    dfs = dict()

    if player_type == 'batters':
        columns = ['ab', 'avg', 'r', 'hr', 'rbi', 'sb']
    else:
        columns = ['ip', 'era', 'whip', 'k', 'w', 'sv']

    for term in ['season', 'month', 'week']:
        # Compute z-scores seperately for each term
        x = batter_df.filter(pl.col('term') == term)

        for col in columns:
            mult = -1 if col in {'era', 'whip'} else 1
            x = x.with_columns(
                (((pl.col(col) - pl.mean(col)) / pl.std(col)) * mult).alias(f"z_{col}")
            )

        if player_type == 'batters':
            x = x.with_columns(
                ((pl.col('z_avg') +\
                  pl.col('z_hr') +\
                  pl.col('z_rbi') +\
                  pl.col('z_sb') +\
                  pl.col('z_r') +\
                  pl.col('z_ab')) / 6).alias('z_total')
            )
        else:
            x = x.with_columns(
                ((pl.col('z_era') +\
                  pl.col('z_whip') +\
                  pl.col('z_k') +\
                  pl.col('z_w') +\
                  pl.col('z_sv') +\
                  pl.col('z_ip')) / 6).alias('z_total')
            )

        # Add a column for ranking by z_total
        x = x.with_columns(pl.struct('z_total').rank('ordinal', descending=True).alias("Rank"))

        # Drop columns we don't want in final output (i.e. the z_ columns)
        drop_columns = [f'z_{col}' for col in columns] + ['z_total']
        x = x.drop(drop_columns)

        # Save to dict
        dfs[term] = x.clone()

    # Return the final output as a concatenation of all the term-specific DFs
    final = pl.concat(list(dfs.values()))

    return final


def main() -> None:
    """
    Script used to generate a DataFrame that will be used plot reports comparing Free Agents
    in a Yahoo Fantasy Baseball league to the players on my team.

    Uses the Yahoo Fantasy API to pull data for all players, compute z-scores to approximate
    the Yahoo Fantasy rankings (with some other metrics baked in) and outputs a batter and pitcher
    CSV containing all players on my team as well as every free agent with all of the metrics
    included.
    """
    # Authenticate the session and get the League object to query against
    session = create_session()
    game = yfa.Game(session, 'mlb')
    league = yfa.League(session, game.league_ids()[1])

    # Get every player that's already on a team
    taken = league.taken_players()

    batter_ids = [p['player_id'] for p in taken if p['position_type'] == 'B']

    # Get the free agent batters, combine with taken ones, and compute rank
    batters = league.free_agents('Util')
    batter_ids.extend([p['player_id'] for p in batters])

    b_df = collect_batter_stats(batter_ids, league)

    # Output DataFrame with ranks included
    batter_df = compute_z_scores(b_df, player_type='batters')

    # Do the same with pitchers
    pitcher_ids = [p['player_id'] for p in taken if p['position_type'] == 'P']
    pitchers = league.free_agents('P')
    pitcher_ids.extend([p['player_id'] for p in pitchers])

    p_df = collect_pitcher_stats(pitcher_ids, league, position='P')

    pitcher_df = compute_z_scores(p_df, player_type='pitchers')

    # Compile list of all free agents to be included in final output
    free_agent_batters = list(set([p['player_id'] for p in batters]))
    free_agent_pitchers = list(set([p['player_id'] for p in pitchers]))

    # Do the same with players from my team
    my_team = list(get_players_from_own_team(position='Util',
                                             league=league,
                                             session=session)['id']) + \
              list(get_players_from_own_team(position='P',
                                             league=league,
                                             session=session)['id'])
    my_team = list(set(my_team))

    # Filter players that are on other teams
    batter_df = filter_taken(batter_df, free_agent_batters, my_team)
    pitcher_df = filter_taken(pitcher_df, free_agent_pitchers, my_team)

    # Rename columns to be more presentable
    batter_df = batter_df.rename({
        "name": "Name",
        "team": "Team",
        "positions": "Position(s)",
        "avg": "AVG",
        "ab": "ABs",
        "hr": "HRs",
        "rbi": "RBIs",
        "r": "Runs",
        "sb": "SBs"
    })
    batter_df.drop('id')

    pitcher_df = pitcher_df.rename({
        "name": "Name",
        "team": "Team",
        "positions": "Position(s)",
        "ip": "IP",
        "k": "Ks",
        "w": "Ws",
        "sv": "SVs",
        "era": "ERA",
        "whip": "WHIP",
    })
    pitcher_df.drop('id')

    # Save output
    pitcher_df.write_csv('pitcher_data.csv')
    batter_df.write_csv('batter_data.csv')


if __name__ == '__main__':
    main()
