import pandas as pd
import numpy as np
import pathlib, sys

DATAPATH=pathlib.Path(sys.argv[1])
assert DATAPATH.exists(), "Path does not exist."

# create interim data dir if not exists
INTERIM=DATAPATH/'interim'
INTERIM.mkdir(exist_ok=True)

def check_distinct_by(df, by):
    return df.groupby(by=by).nunique().shape[0] == df.shape[0]

def apply_bonus(x, bonus_pts):
    """
    Args:
      x: pandas column
      bonus_points: list of tuples stating (`threshold`, `points`)
      
    Returns:
      sum of bonus points
    """
    total_pts = 0
    for bp in bonus_pts:
        bonus_threshold, bonus_points = bp
        if x >= bonus_threshold:
            total_pts += bonus_points
    return total_pts

def print_save_path(file):
    print("Saving to: ", file.resolve())

##########################################################################################

# season rushing
df = pd.read_csv(DATAPATH/'Career_Stats_Rushing.csv',
                 na_values='--',
                 thousands=','
                )
# remove Touchdown indicator
df['Longest Rushing Run'] = pd.to_numeric(df['Longest Rushing Run'].str.replace('T',''))

assert check_distinct_by(df, by=['Player Id', 'Year', 'Team']),\
"Index is not distinct"

# insert rushing into df
df_season = pd.DataFrame(df[['Player Id', 'Year', 'Position', 'Team',
                             "Games Played", "Rushing Attempts", "Rushing Attempts Per Game",
                             "Rushing Yards", "Yards Per Carry", "Rushing Yards Per Game",
                             "Rushing TDs", "Longest Rushing Run", "Rushing First Downs",
                             "Percentage of Rushing First Downs", "Rushing More Than 20 Yards",
                             "Rushing More Than 40 Yards"]],
                         copy=True)

##########################################################################################

# season receiving
df = pd.read_csv(DATAPATH/'Career_Stats_Receiving.csv',
                 na_values='--',
                 thousands=','
                )
# remove touchdown indicator
df['Longest Reception'] = pd.to_numeric(df['Longest Reception'].str.replace('T',''))

assert check_distinct_by(df, by=['Player Id', 'Year', 'Team']),\
"Index is not distinct"

df_tmp = df[['Player Id', 'Year', 'Team', 'Games Played',
             'Receptions', 'Receiving Yards', 'Yards Per Reception', 'Yards Per Game',
             'Longest Reception', 'Receiving TDs', 'Receptions Longer than 20 Yards',
             'Receptions Longer than 40 Yards', 'First Down Receptions']].copy()

df_tmp.rename(columns={'Games Played': 'Games Played Rec'}, inplace=True)

df_season = pd.merge(df_season, df_tmp, how='outer', on=['Player Id', 'Year', 'Team'])

##########################################################################################

# season fumbles
df = pd.read_csv(DATAPATH/'Career_Stats_Fumbles.csv',
                 na_values='--',
                 thousands=','
                )

assert check_distinct_by(df, by=['Player Id', 'Year', 'Team']),\
"Index is not distinct"

df_tmp = df[['Player Id', 'Year', 'Team', 'Fumbles', 'Fumbles Lost']].copy()
df_tmp.fillna(0, inplace=True)

df_season = pd.merge(df_season, df_tmp, how='left', on=['Player Id', 'Year', 'Team'])

##########################################################################################

# Create RB Data
df_rb = df_season[df_season['Position'] == 'RB'].copy()
# coalesce games played
df_rb['Games Played'] = df_rb['Games Played'].combine_first(df_rb['Games Played Rec'])
df_rb.drop(['Games Played Rec'], axis=1, inplace=True)
df_rb.fillna(0, inplace=True)
df_rb.to_csv(INTERIM/'rb_career_stats.csv', index=False)
print_save_path(INTERIM/'rb_career_stats.csv')

# Create WR Data
df_wr = df_season[df_season['Position'] == 'WR'].copy()
# coalesce games played
df_wr['Games Played'] = df_wr['Games Played Rec'].combine_first(df_wr['Games Played'])
df_wr.drop(['Games Played Rec'], axis=1, inplace=True)
df_wr.fillna(0, inplace=True)
df_wr.to_csv(INTERIM/'wr_career_stats.csv', index=False)
print_save_path(INTERIM/'wr_career_stats.csv')

# Create TE Data
df_te = df_season[df_season['Position'] == 'TE'].copy()
# coalesce games played
df_te['Games Played'] = df_te['Games Played Rec'].combine_first(df_te['Games Played'])
df_te.drop(['Games Played Rec'], axis=1, inplace=True)
df_te.fillna(0, inplace=True)
df_te.to_csv(INTERIM/'te_career_stats.csv', index=False)
print_save_path(INTERIM/'te_career_stats.csv')

##########################################################################################

# Process Game Logs

# Running Backs
df_rb = pd.read_csv(DATAPATH/'Game_Logs_Runningback.csv',
                    na_values='--',
                    parse_dates={'gameDate' : ['Game Date', 'Year']},
                    keep_date_col=True
                    )
df_rb['Year'] = df_rb['Year'].astype(np.int)
df_rb['Longest Rushing Run'] = pd.to_numeric(df_rb['Longest Rushing Run'].str.replace('T',''))
df_rb['Longest Reception'] = pd.to_numeric(df_rb['Longest Reception'].str.replace('T',''))
df_rb = df_rb[df_rb['Season'] == 'Regular Season']
df_rb.to_csv(INTERIM/'rb_game_logs.csv', index=False)
print_save_path(INTERIM/'rb_game_logs.csv')

# Wide Receivers and Tight Ends
df_wr = pd.read_csv(DATAPATH/'Game_Logs_Wide_Receiver_and_Tight_End.csv',
                    na_values='--',
                    parse_dates={'gameDate' : ['Game Date', 'Year']},
                    keep_date_col=True
                    )
df_wr['Year'] = df_wr['Year'].astype(np.int)
df_wr['Longest Rushing Run'] = pd.to_numeric(df_wr['Longest Rushing Run'].str.replace('T',''))
df_wr['Longest Reception'] = pd.to_numeric(df_wr['Longest Reception'].str.replace('T',''))
df_wr = df_wr[df_wr['Season'] == 'Regular Season']
df_wr1 = df_wr[df_wr['Position'] == 'WR']
df_te1 = df_wr[df_wr['Position'] == 'TE']
df_wr1.to_csv(INTERIM/'wr_game_logs.csv', index=False)
print_save_path(INTERIM/'wr_game_logs.csv')
df_te1.to_csv(INTERIM/'te_game_logs.csv', index=False)
print_save_path(INTERIM/'te_game_logs.csv')

##########################################################################################

# Process Fantasy Points

def compute_fantasy_points(df, points_map):
    df_fp = df[['Player Id', 'Year', 'Week', 'Games Played', 'Games Started',
                'Rushing Yards', 'Rushing TDs', 'Longest Rushing Run',
                'Receptions', 'Receiving Yards', 'Receiving TDs', 'Longest Reception',
                'Fumbles', 'Fumbles Lost'
                 ]].copy()
    # df_fp.fillna(0, inplace=True)

    # apply fantasy point rules
    df_fp['Rushing Yards pts'] = df_fp['Rushing Yards'] * points_map['Rushing Yards']
    df_fp['Receiving Yards pts'] = df_fp['Receiving Yards'] * points_map['Receiving Yards']
    df_fp['Rushing TDs pts'] = df_fp['Rushing TDs'] * points_map['Rushing TDs']
    df_fp['Receiving TDs pts'] = df_fp['Receiving TDs'] * points_map['Receiving TDs']
    df_fp['Receptions pts'] = df_fp['Receptions'] * points_map['Receptions']

    bonus_pts = [points_map['Rushing Bonus 1'],
                 points_map['Rushing Bonus 2'],
                 points_map['Rushing Bonus 3']]
    df_fp['Rushing Bonus pts'] = df_fp['Rushing Yards']\
    .apply(apply_bonus, bonus_pts=bonus_pts)

    bonus_pts = [points_map['Rushing Bonus 1'],
                 points_map['Rushing Bonus 2'],
                 points_map['Rushing Bonus 3']]
    df_fp['Receiving Bonus pts'] = df_fp['Receiving Yards']\
    .apply(apply_bonus, bonus_pts=bonus_pts)

    all_pts = list(df_fp.columns[-7:])
    df_fp['Total pts'] = np.sum(df_fp[all_pts], axis=1)
    df_fp['Total pts active'] = np.where(df_fp['Games Played'] == 1, df_fp['Total pts'], 0)
    df_fp['Total pts started'] = np.where(df_fp['Games Started'] == 1, df_fp['Total pts'], 0)
    return df_fp
    
def aggregate_fantasy_points(df_fp):
    all_pts = ['Rushing Yards pts',
               'Receiving Yards pts',
               'Rushing TDs pts',
               'Receiving TDs pts',
               'Receptions pts',
               'Rushing Bonus pts',
               'Receiving Bonus pts']
    
    df_fp_season = df_fp[['Player Id', 'Year'] + 
                         all_pts + 
                         ['Total pts', 'Total pts active', 'Total pts started', 'Games Played', 'Games Started']]\
    .groupby(by=['Player Id', 'Year'], sort=False)\
    .agg(['sum', 'mean'])

    df_fp_season.columns = [' '.join(col).strip() for col in df_fp_season.columns.values]
    df_fp_season.reset_index(inplace=True)

    df_fp_season['Avg pts per Games Played'] = df_fp_season['Total pts active sum'] / df_fp_season['Games Played sum']
    df_fp_season['Avg pts per Games Started'] = df_fp_season['Total pts started sum'] / df_fp_season['Games Started sum']
    df_fp_season['Percentage Played Games'] = df_fp_season['Games Played sum'] / 16
    df_fp_season.rename(columns={'Games Started mean': 'Percentage Started Games'}, inplace=True)
    df_fp_season.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_fp_season.drop(['Total pts started mean', 'Total pts active mean'], axis=1, inplace=True)
    
    df_fp_season['Sum Fantasy Rank'] = df_fp_season\
        .groupby(['Year'])['Total pts sum']\
        .rank(ascending=False)
    df_fp_season['Avg Fantasy Rank'] = df_fp_season\
        .groupby(['Year'])['Total pts mean']\
        .rank(ascending=False)
    df_fp_season['Avg pts Played Rank'] = df_fp_season\
        .groupby(['Year'])['Avg pts per Games Played']\
        .rank(ascending=False)
    df_fp_season['Avg pts Started Rank'] = df_fp_season\
        .groupby(['Year'])['Avg pts per Games Started']\
        .rank(ascending=False)
    
    return df_fp_season

# PSK Rules
points_map = {'Complete Passes': 0.2,
              'Incomplete Passes': -0.2,
              'Passing Yards': 1/25,
              'Passing Bonus 1': (300, 2),
              'Passing Bonus 2': (400, 2),
              'Passing Bonus 3': (500, 3),
              'Passing TDs': 4,
              'Interceptions': -2,
              'Rushing Yards': 1/10,
              'Rushing Bonus 1': (100, 2),
              'Rushing Bonus 2': (150, 3),
              'Rushing Bonus 3': (200, 5),
              'Rushing TDs': 6,
              'Receptions': 0.4,
              'Receiving Yards': 1/10,
              'Receiving Bonus 1': (100, 1.5),
              'Receiving Bonus 2': (150, 2),
              'Receiving Bonus 3': (200, 3),
              'Receiving TDs': 6,
              'Two Point Conversion': 2,
              'Fumbles Lost': -2,
              '40+ Yard Completions': 1,
              '40+ Yard Passing TDs': 1,
              '40+ Yard Run': 1.5,
              '40+ Yard Rushing TDs': 2,
              '40+ Yard Receptions': 1.5,
              '40+ Yard Receiving TDs': 1.5
             }

# Running Backs
df = pd.read_csv(INTERIM/'rb_game_logs.csv')
df_fp = compute_fantasy_points(df, points_map)
df_fp_season = aggregate_fantasy_points(df_fp)
df_fp_season.to_csv(INTERIM/'rb_fantasy_points_psk.csv', index=False)
print_save_path(INTERIM/'rb_fantasy_points_psk.csv')

# Wide Receivers
df = pd.read_csv(INTERIM/'wr_game_logs.csv')
df_fp = compute_fantasy_points(df, points_map)
df_fp_season = aggregate_fantasy_points(df_fp)
df_fp_season.to_csv(INTERIM/'wr_fantasy_points_psk.csv', index=False)
print_save_path(INTERIM/'wr_fantasy_points_psk.csv')

# Tight Ends
df = pd.read_csv(INTERIM/'te_game_logs.csv')
df_fp = compute_fantasy_points(df, points_map)
df_fp_season = aggregate_fantasy_points(df_fp)
df_fp_season.to_csv(INTERIM/'te_fantasy_points_psk.csv', index=False)
print_save_path(INTERIM/'te_fantasy_points_psk.csv')

