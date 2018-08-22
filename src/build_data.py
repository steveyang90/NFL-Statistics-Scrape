import pandas as pd
import numpy as np
import pathlib

DATAPATH=pathlib.Path(sys.argv[1])
assert DATAPATH.exists(), "Path does not exist."

# create interim data dir if not exists
INTERIM=DATAPATH/'interim'
INTERIM.mkdir(exist_ok=True)

def check_distinct_by(df, by):
    return df.groupby(by=by).nunique().shape[0] == df.shape[0]

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
df_rb = df_season[df_season['Position'] == 'RB']
# coalesce games played
df_rb['Games Played'] = df_rb['Games Played'].combine_first(df_rb['Games Played Rec'])
df_rb.drop(['Games Played Rec'], axis=1, inplace=True)
df_rb.to_csv(INTERIM/'rb_career_stats.csv', index=False)

# Create WR Data
df_wr = df_season[df_season['Position'] == 'WR']
# coalesce games played
df_wr['Games Played'] = df_wr['Games Played Rec'].combine_first(df_wr['Games Played'])
df_wr.drop(['Games Played Rec'], axis=1, inplace=True)
df_wr.to_csv(INTERIM/'wr_career_stats.csv', index=False)

# Create TE Data
df_te = df_season[df_season['Position'] == 'TE']
# coalesce games played
df_te['Games Played'] = df_te['Games Played Rec'].combine_first(df_te['Games Played'])
df_te.drop(['Games Played Rec'], axis=1, inplace=True)
df_te.to_csv(INTERIM/'te_career_stats.csv', index=False)