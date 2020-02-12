#
# Module 'csgo.py':
# - Author: Emmanuel Garza
# - Last modified: Feb 11,2020
#
# Description:
# -> This module contains useful functions for the modeling of professional matches of Counter Strike: Global Offensive
#

import numpy as np
import pickle
import pandas as pd
import json
from datetime import datetime

def load_data():
# 'load_data': Loads the pre-saved dictionaries and pandas data frames containing
#              match and player data

    # Required modules
    

    # -> Player dictionary with the dataFrames
    f = open('data/dict_player.pickle', 'rb')
    dict_player = pickle.load(f)

    # -> Map dataFrames
    f = open('data/df_map.pickle','rb')
    df_map = pickle.load(f)

    # -> Map dictionary
    f = open('data/dict_map.pickle','rb')
    dict_map = pickle.load(f)

    # We set all the team ranks of 0 to 420 which is basically the last place
    df_map.loc[ df_map['team_rank_1']==0, 'team_rank_1'] = 420
    df_map.loc[ df_map['team_rank_2']==0, 'team_rank_2'] = 420

    return dict_player, df_map, dict_map




# Function to append values of all the players
def append_val(team_01,stat_vec,stat_name,dict_inout):

    n = len(stat_vec)
    order = np.argsort( stat_vec )[::-1][:n]

    count_p = -1
    for ind in order:
        count_p = count_p + 1
        dict_inout['t_'+team_01+'_p_'+str(count_p)+'_'+stat_name] = stat_vec[ind]
    
    return dict_inout



def create_pre_train_set( dict_player, df_map, dict_map, DAYS_WEIGHT, MAX_RANK, START_DATE, N_MAPS ):
# 'create_training_set': Creates the training set for the model

    

    time_1 = datetime.now()

    df_map_ranked = ( df_map[ (df_map['date']>START_DATE) & (df_map['team_rank_1']<MAX_RANK) &  (df_map['team_rank_2']<MAX_RANK) ] )
    df_map_ranked = df_map_ranked.sort_values(['date'],ascending=False)

    print( 'Total number of matches = '+str( len(df_map_ranked) ))

    metric_vec = ['KAST', 'ADR', 'first_kills_diff', 'rating',
        'kills_per_round', 'deaths_per_round', 'impact', 
        'team_score',
        'op_score', 'win', 'team_rank', 'prize']

    map_training_dict = {}

    for map_id in df_map_ranked['map_id'][:N_MAPS]:
        
        map_training_dict[map_id] = {}

        # Keys for the dictionary
        map_training_dict[map_id]['t1_win']    = 0
        map_training_dict[map_id]['fav_win']   = 0
        map_training_dict[map_id]['fav_ind']   = 0
        map_training_dict[map_id]['map']       = df_map_ranked.at[map_id,'map']
        map_training_dict[map_id]['score_dif'] = 0
        map_training_dict[map_id]['t1_rank']   = 0
        map_training_dict[map_id]['t2_rank']   = 0

        with open('/home/emmanuel/Desktop/csgo-csv/json_maps/hltv_map_'+str(map_id)+'.json') as f:

            data = json.load(f)
            id_ct_start = data['roundHistory'][0]['ctTeam']

        team_1  = df_map.at[map_id,'team_id_1']
        rank_1  = df_map.at[map_id,'team_rank_1']
        score_1 = df_map.at[map_id,'team_score_1']

        team_2  = df_map.at[map_id,'team_id_2']
        rank_2  = df_map.at[map_id,'team_rank_2']
        score_2 = df_map.at[map_id,'team_score_2']

        if team_1 != id_ct_start:
            # We swap them
            team_aux  = team_1
            rank_aux  = rank_1
            score_aux = score_1

            team_1  = team_2
            rank_1  = rank_2
            score_1 = score_2

            team_2  = team_aux
            rank_2  = rank_aux
            score_2 = score_aux

        map_training_dict[map_id]['score_dif'] = score_1-score_2
        map_training_dict[map_id]['t1_rank']   = rank_1
        map_training_dict[map_id]['t2_rank']   = rank_2
        

        if score_1 > score_2:
            map_training_dict[map_id]['t1_win'] = 1
        else:
            map_training_dict[map_id]['t1_win'] = 0


        if map_training_dict[map_id]['t1_rank'] < map_training_dict[map_id]['t2_rank']:
            map_training_dict[map_id]['fav_ind'] = 1
        else:
            map_training_dict[map_id]['fav_ind'] = 2


        if ( (map_training_dict[map_id]['score_dif'] > 0) & (map_training_dict[map_id]['t1_rank']<map_training_dict[map_id]['t2_rank']) )|( (map_training_dict[map_id]['score_dif'] < 0) & (map_training_dict[map_id]['t2_rank']<map_training_dict[map_id]['t1_rank']) ):
            map_training_dict[map_id]['fav_win'] = 1


        # Now we fill the player statistics -----------------------------------------#
        map_date = df_map_ranked.loc[map_id]['date']

        # Here we are taking the rankings to be non-zero
        team_vec = [team_1, team_2]
        
        for ind in range(0,2):

            team_id = team_vec[ind]

            rating_vec    = []
            prize_rtg_vec = []
            hs_vec        = []
            kills_per_rd_vec  = []
            deaths_per_rd_vec = []
            adr_vec = []
            kast_vec = []
            assists_per_rd_vec = []
            flash_per_rd_vec   = []
            first_kills_dif_vec = []
            team_rank_vec = []
            score_dif_vec = []
            win_rate_vec = []
            win_rate_map_vec = []

            kd_per_round_vec  = []

            scaled_win_vec = []
            scaled_rating_vec = [] 
            scaled_score_dif_vec = []
            scaled_kd_vec = []

            momentum_vec = []
            map_rating_vec = []

            for player_id in dict_map[map_id][team_id]['players_id']:            
                
                # Get the data for this player
                df_aux   = dict_player[player_id]
                date_vec = (map_date-df_aux['date']).astype('timedelta64[D]')

                # Note the prize rating we take a whole year back
                prize = df_aux[ (date_vec>1) & (date_vec<365) ]['prize'].sum()
                if prize > 0.0:
                    prize_rtg_vec.append( np.log( prize )/12.0 )
                else:
                    prize_rtg_vec.append( 0.0 )


                df_aux   = df_aux[ (date_vec>1) & (date_vec<DAYS_WEIGHT) ]

                # What if we use only the historical data for this map in particular
                # df_aux = df_aux[ (date_vec>1) & (date_vec<DAYS_WEIGHT) & (df_aux['map']==df_map_ranked.at[map_id,'map']) ]

                # Append the average values of this player 
                rating_vec.append         ( df_aux['rating'].mean() )
                hs_vec.append             ( (df_aux['hs_kills']/(df_aux['team_score']+df_aux['op_score'])).mean() )        
                kills_per_rd_vec.append   ( df_aux['kills_per_round'].mean() )
                deaths_per_rd_vec.append  ( df_aux['deaths_per_round'].mean() )
                adr_vec.append            ( df_aux['ADR'].mean() )
                kast_vec.append           ( df_aux['KAST'].mean() )
                assists_per_rd_vec.append ( (df_aux['assists']/(df_aux['team_score']+df_aux['op_score'])).mean() )
                flash_per_rd_vec.append   ( (df_aux['flash_assists']/(df_aux['team_score']+df_aux['op_score'])).mean() )
                first_kills_dif_vec.append( (df_aux['first_kills_diff']/(df_aux['team_score']+df_aux['op_score'])).mean() )
                team_rank_vec.append      ( df_aux['team_rank'].mean() )
                score_dif_vec.append      ( (df_aux['team_score']-df_aux['op_score']).mean() )
                win_rate_vec.append       ( df_aux['win'].mean() )

                

                kd_per_round_vec.append  ( (df_aux['kills_per_round']-df_aux['deaths_per_round']).mean() )

                # Get the opponent rank
                df_player_maps = df_map.loc[df_aux.index.values]
                op_rank_vec    = (df_aux['team_id']!=df_player_maps['team_id_1'])*df_player_maps['team_rank_1'] + (df_aux['team_id']!=df_player_maps['team_id_2'])*df_player_maps['team_rank_2']

                # Scaled Variables
                
                # Linear scaling
                # alpha = 0.1
                # op_rank_vec = alpha + (1.0-alpha)*(420.0-op_rank_vec)/420.0
                
                # scaled_win_vec.append       ( (df_aux['win']*op_rank_vec).mean() )
                # scaled_rating_vec.append    ( (df_aux['rating']*op_rank_vec).mean() )
                # scaled_score_dif_vec.append ( ((df_aux['team_score']-df_aux['op_score'])*op_rank_vec).mean() )

                scaled_win_vec.append       ( (df_aux['win']/op_rank_vec).mean() )
                scaled_rating_vec.append    ( (df_aux['rating']/op_rank_vec).mean() )
                scaled_score_dif_vec.append ( ((df_aux['team_score']-df_aux['op_score'])/op_rank_vec).mean() )

                scaled_kd_vec.append ( ((df_aux['kills_per_round']-df_aux['deaths_per_round'])/op_rank_vec).mean() )

                momentum_vec.append ( df_aux['win'].head(n=15).mean() )

                df_aux_map = df_aux[ df_aux['map']==df_map_ranked.at[map_id,'map'] ]
                map_rating_vec.append( (df_aux_map['team_score']-df_aux_map['op_score']).mean() )
                win_rate_map_vec.append ( (df_aux_map['win']/op_rank_vec).mean() )


            t = str(ind)

            map_training_dict[map_id] = append_val(t,prize_rtg_vec,'prize_rating',map_training_dict[map_id])

            map_training_dict[map_id] = append_val(t,rating_vec,'rating',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,hs_vec,'hs_perc',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,kills_per_rd_vec,'kills_per_rd',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,deaths_per_rd_vec,'deaths_per_rd',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,adr_vec,'adr',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,kast_vec,'kast',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,assists_per_rd_vec,'assists_per_rd',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,flash_per_rd_vec,'flash_per_rd',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,first_kills_dif_vec,'first_kills_dif',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,team_rank_vec,'team_rank',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,score_dif_vec,'score_dif',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,win_rate_vec,'win_rate',map_training_dict[map_id])

            map_training_dict[map_id] = append_val(t,win_rate_map_vec,'win_rate_map',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,kd_per_round_vec,'kd_per_round',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,scaled_win_vec,'scaled_win',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,scaled_rating_vec,'scaled_rating',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,scaled_score_dif_vec,'scaled_score_dif',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,scaled_kd_vec,'scaled_kd',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,momentum_vec,'momentum',map_training_dict[map_id])
            map_training_dict[map_id] = append_val(t,map_rating_vec,'map_rating',map_training_dict[map_id])
        
    print('Converting data to DataFrame')
    df_ct_start = pd.DataFrame.from_dict( map_training_dict,orient='index')
    time_2 = datetime.now()

    print( 'Total time to create training set = '+str(time_2-time_1) )

    return df_ct_start





def create