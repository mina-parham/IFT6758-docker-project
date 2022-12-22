import math

import numpy as np
import pandas as pd
from numpy.linalg import norm
from pathlib import Path
import requests
import json
from itertools import compress
from sklearn.preprocessing import LabelEncoder






class GameClient:
    def __init__(self):
        self.game_list = {}

    def get_games_data(self, game_id):
 
        path_file = Path(f"{str(game_id)}.json")
        url = f'http://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live'
        
        response = requests.get(url)
        if response.status_code != 404:
            with open(path_file, 'w') as f:
                json.dump(response.json(), f)
        
            return path_file
        else:
            print(
                f"Status code: {response.status_code} at gameID:{game_id}, not found"
            )
            return None
    
    def ping_game(self, game_id):

        if game_id not in self.game_list:
            df = self.tidy(game_id, -1)
            self.game_list[game_id] = df
            return df
        else:
            df_main = self.game_list.get(game_id)
            event_idx_last = df_main.iloc[-1, df_main.columns.get_loc('event_idx')]
            df = self.tidy(game_id, event_idx_last)
            self.game_list[game_id].append(df)
            return self.game_list[game_id]
        

    def tidy(self, game_id, event_idex) -> pd.DataFrame:

        df = {}

        path_file = self.get_games_data(game_id)
        with open(path_file, 'r') as f:
            df_json = json.load(f)
        df[game_id] = df_json
        df = pd.DataFrame.from_dict(df)
                
        event_idx, period_time, period, game_id, team_away_name, team_home_name, is_goal, coordinate_x, coordinate_y, shot_type,\
        strength, shooter_name, goalie_name, empty_net, team_name, event_type, last_type, last_coord_x, last_coord_y, last_period,\
        last_period_time, rebound, number_friendly, number_opposing, power_play= ([] for i in range(25))
        
        
        #allplays_data = allplays_data[int(allplays_data['about']['eventIdx'])> int(event_idex)]

        for i in range(df.shape[1]):
            allplays_data = df.iloc[:,i]["liveData"]["plays"]["allPlays"]
            x =[int(allplays_data[i]['about']['eventIdx'])> int(event_idex) for i in range(len(allplays_data))]
            allplays_data = list(compress(allplays_data, x))
            p = {
                "home_minor_2": [],
                "home_minor_4": [],
                "home_major": [],
                "away_minor_2": [],
                "away_minor_4": [],
                "away_major": [],
            }
            time_2 = 0.0
            time_1 = 0
            time_period_1 = 0
            time_period_2 = 0
            for j in range(len(allplays_data)):

                time_1 = int(allplays_data[j]['about']['periodTime'][0:2])*60 +int(allplays_data[j]['about']['periodTime'][3:5])
                time_period_1 = (allplays_data[j]['about']['period']-1)*1200

                if j>0:
                    time_2 = int(allplays_data[j-1]['about']['periodTime'][0:2])*60 +int(allplays_data[j-1]['about']['periodTime'][3:5])
                    time_period_2 = (allplays_data[j-1]['about']['period']-1)*1200
                
                time_from_last = time_1 + time_period_1 -time_2 - time_period_2
                    

                for key, values in p.items():
                    for i in range(len(values)):
                        values[i] -=time_from_last
                    p[key]  = [i for i in values if i > 0]
                
                friendly = max(5 - (len(p['home_minor_2'])+len(p['home_minor_4'])+len(p['home_minor_4'])),3)
                opposing = max(5 - (len(p['away_minor_2'])+len(p['away_minor_4'])+len(p['away_minor_4'])),3)

                if(friendly != opposing):
                        power_play_second +=time_from_last
                else:
                        power_play_second = 0
            



                if(allplays_data[j]['result']['eventTypeId'] == "SHOT" or allplays_data[j]['result']['eventTypeId'] == "GOAL"):
                    event_type.append(allplays_data[j]['result']['eventTypeId'])
                    period.append(allplays_data[j]['about']['period'])
                    period_time.append(allplays_data[j]['about']['periodTime'])    
                    game_id.append(df.iloc[:, 0].name)
                    event_idx.append(allplays_data[j]['about']['eventIdx'])
                    team_away_name.append(df.iloc[:,0]['gameData']['teams']['away']['name'])
                    team_home_name.append(df.iloc[:,0]['gameData']['teams']['home']['name'])
                    team_name.append(allplays_data[j]['team']['name'])
                    is_goal.append(allplays_data[j]['result']['eventTypeId']=="GOAL")
                    coordinate_x.append(allplays_data[j]['coordinates']['x'] if  'x' in allplays_data[j]['coordinates'] else np.nan)
                    coordinate_y.append(allplays_data[j]['coordinates']['y'] if  'y' in allplays_data[j]['coordinates'] else np.nan)
                    shot_type.append(allplays_data[j]['result']['secondaryType'] if 'secondaryType' in allplays_data[j]['result'] else np.nan)
                    strength.append(allplays_data[j]['result']['strength']['name'] if allplays_data[j]['result']['eventTypeId'] == "GOAL" else np.nan)
                    if (allplays_data[j]['players'][z]['playerType'] == "Shooter" or allplays_data[j]['players'][z]['playerType'] =='Scorer' for z in range(len(allplays_data[j]['players']))):
                        shooter_name.append([allplays_data[j]['players'][z]['player']['fullName'] for z in range(len(allplays_data[j]['players']))][0])
                    if (allplays_data[j]['players'][z]['playerType']=="Goalie" for z in range(len(allplays_data[j]['players']))):
                        goalie_name.append([allplays_data[j]['players'][z]['player']['fullName'] for z in range(len(allplays_data[j]['players']))][0])
                    empty_net.append(True if 'emptyNet' in allplays_data[j]['result'] and allplays_data[j]['result']['emptyNet']==True else False)
                    

                    if j> 0:
                        last_type.append(allplays_data[j-1]['result']['eventTypeId'])
                        last_period.append(allplays_data[j-1]['about']['period'])
                        last_period_time.append(allplays_data[j-1]['about']['periodTime'])
                        last_coord_x.append(allplays_data[j-1]['coordinates']['x'] if  'x' in allplays_data[j-1]['coordinates'] else np.nan)
                        last_coord_y.append(allplays_data[j-1]['coordinates']['y'] if  'y' in allplays_data[j-1]['coordinates'] else np.nan)
                        rebound.append(allplays_data[j-1]['result']['eventTypeId']=="SHOT")
                        

                    else:
                        last_type.append(np.nan)
                        last_period.append(np.nan)
                        last_period_time.append(np.nan)
                        last_coord_x.append(np.nan)
                        last_coord_y.append(np.nan)
                        rebound.append(np.nan)

                    
                    number_friendly.append(friendly)
                    number_opposing.append(opposing)
                    power_play.append(power_play_second)

                
                if(allplays_data[j]['result']['eventTypeId'] == "GOAL"):
                    if(df.iloc[:,0]['gameData']['teams']['home']['name'] == allplays_data[j]['team']['name']):
                        if(len(p['away_minor_2'])!=0 and len(p['away_minor_4'])==0):
                            p['away_minor_2'][0] = 0
                        elif(len(p['away_minor_4']) != 0 and len(p['away_minor_2'])==0):
                            if(p['away_minor_4'][0]< 120):
                                p['away_minor_4'][0] = 0
                            else:
                                p['away_minor_4'][0] = 120
                        elif(len(p['away_minor_2'])!=0 and len(p['away_minor_4'])!=0):
                            if(p['away_minor_2'][0]<p['away_minor_4'][0]):
                                p['away_minor_2'][0] = 0
                            else:
                                if(p['away_minor_4']<120):
                                    p['away_minor_4'][0] = 0
                                else:
                                    p['away_minor_4'][0] =120
                    else:
                        if(len(p['home_minor_2'])!=0 and len(p['home_minor_4'])==0):
                            p['home_minor_2'][0] = 0
                        elif(len(p['home_minor_4']) != 0 and len(p['home_minor_2'])==0):
                            if(p['home_minor_4'][0]< 120):
                                p['home_minor_4'][0] = 0
                            else:
                                p['home_minor_4'][0] = 120
                        elif(len(p['home_minor_2'])!=0 and len(p['home_minor_4'])!=0):
                            if(p['home_minor_2'][0]<p['home_minor_4'][0]):
                                p['home_minor_2'][0] = 0
                            else:
                                if(p['home_minor_4']<120):
                                    p['home_minor_4'][0] = 0
                                else:
                                    p['home_minor_4'][0] =120
                
                if(allplays_data[j]['result']['eventTypeId'] == "PENALTY"):
                    if (df.iloc[:,0]['gameData']['teams']['home']['name'] ==allplays_data[j]['team']['name']):
                        if(allplays_data[j]['result']['penaltySeverity']=='Minor'):
                            if(allplays_data[j]['result']['penaltyMinutes']==4):
                                p['home_minor_4'].append(240)
                            else:
                                p['home_minor_2'].append(120)
                        else:
                            p['home_major'].append(300)
                    else:
                        if(allplays_data[j]['result']['penaltySeverity']=='Minor'):
                            if(allplays_data[j]['result']['penaltyMinutes']==4):
                                p['away_minor_4'].append(240)
                            else:
                                p['away_minor_2'].append(120)
                        else:
                            p['away_major'].append(300)
                
        assert(all(len(lists) == len(game_id) for lists in [event_idx, period_time, period, team_away_name, team_home_name, is_goal, coordinate_x,\
        coordinate_y, shot_type, strength, shooter_name, goalie_name, empty_net, team_name,\
        event_type, last_type, last_coord_x, last_coord_y, last_period, last_period_time, rebound, number_friendly, number_opposing, power_play]))

        #df_main = pd.DataFrame(np.column_stack([event_idx, period_time]), columns= ['event_idx', 'period_time'])
        df_main = pd.DataFrame(np.column_stack([event_idx, period_time, period, game_id, team_away_name, team_home_name, is_goal, coordinate_x,\
        coordinate_y, shot_type, strength, shooter_name, goalie_name, empty_net, team_name,\
        event_type, last_type, last_coord_x, last_coord_y, last_period, last_period_time, rebound, number_friendly, number_opposing,\
            power_play]),columns=['event_idx', 'period_time', 'period', 'game_id', 'team_away_name', 'team_home_name','is_goal', 'coordinate_x',
                                        'coordinate_y', 'shot_type', 'strength', 'shooter_name','goalie_name', 'empty_net', 'team_name',
                                        'event_type', 'last_type', 'last_coord_x','last_coord_y', 'last_period', 'last_period_time', 'rebound', 
                                        'number_friendly', 'number_opposing', 'power_play'])
        
        df_main['coordinate_x'] = df_main['coordinate_x'].astype('float')
        df_main['coordinate_y'] = df_main['coordinate_y'].astype('float')
        df_main['last_coord_x'] = df_main['last_coord_x'].astype('float')
        df_main['last_coord_y'] = df_main['last_coord_y'].astype('float')
        df_main['is_goal'].replace({'False': 0, 'True': 1}, inplace=True)
        df_main['rebound'].replace({'False': 0, 'True': 1}, inplace=True)

        df_main['empty_net'].replace({'False': 0, 'True': 1}, inplace=True)
        df_main['distance'] = self.distance(df_main['coordinate_x'], df_main['coordinate_y'])
        df_main['from_last_distance']  = np.sqrt((df_main['coordinate_x'] - df_main['last_coord_x'])**2 + (df_main['coordinate_y'] - df_main['last_coord_y'])**2)

        df_main['angle'] = self.angle_between(df_main['coordinate_x'], df_main['coordinate_y'])
        df_main['last_angle'] = self.angle_between(df_main['last_coord_x'], df_main['last_coord_y'])
        df_main = self.convert_date(df_main)
        df_main = self.change_angle(df_main)
        df_main["shot_type"] = LabelEncoder().fit_transform(df_main["shot_type"])
        df_main["last_type"] = LabelEncoder().fit_transform(df_main["last_type"])
        df_main.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_main['game_id'] = df_main['game_id'].astype(str)
        # df_main['rebound'] = df_main['rebound'].astype(str)
        # df_main['period'] = df_main['period'].astype(str)
        # df_main['power_play'] = df_main['power_play'].astype(str)
        # df_main['number_friendly'] = df_main['number_friendly'].astype(str)
        # df_main['number_opposing'] = df_main['number_opposing'].astype(str)





        return df_main
        #return None



    def change_angle(self, df):

        change_angle = []
        speed = []
        for i in range(len(df["coordinate_x"])):
            if df["rebound"][i] == "True":
                if df["angle"][i] >= 0 and df["last_angle"][i] >= 0:
                    x = np.absolute(df["angle"][i] - df["last_angle"][i])
                if df["angle"][i] < 0 and df["last_angle"][i] < 0:
                    x = np.absolute(df["angle"][i]) + np.absolute(df["last_angle"][i])
                change_angle.append(x)
                speed.append(df["from_last_distance"][i] / df["time_from_last"][i])

            else:
                change_angle.append(0)
                speed.append(0)

        df["change_angle"] = change_angle
        df["speed"] = speed
        return df


    def distance(self, x_coor, y_coor):
        """
        Computes the distances between the pock and the goal's center
        Inputs:
        x_coor: It takes the x coordinates
        y_coor: It takes the y_coordinates
        Outputs:
        distance: List of all the distances of all the coordinates present in the data frame
        """
        center_goal = (89, 0)
        x_distance_main = []
        for i in x_coor:
            x_distance = lambda i: center_goal[0] - i if i > 0 else -center_goal[0] - i
            x_distance_main.append(x_distance(i))
        distance = np.round_(
            (np.sqrt(np.asarray(x_distance_main) ** 2 + (center_goal[1] - y_coor) ** 2)),
            decimals=4,
        )
        return distance


    def angle_between(self, x_coor, y_coor):
        """
        Returns the angle in radians between vectors 'v1 = (x_coor,y_coor)' and 'v2 (+/-89,0) -> Center of the net (left/right)'
        """
        center_goal_abs = [89, 0]
        # center_goal_1 = [89,0]
        # center_goal_2 = [-89,0]
        angles = []
        for i in range(len(x_coor)):
            p_v = [x_coor[i], y_coor[i]]
            v2 = center_goal_abs
            if x_coor[i] == v2[1]:
                angle = 0.0
            else:
                if v2[0] == np.absolute(p_v[0]):
                    angle = np.round((np.arctan(((np.absolute(p_v[1]) / (v2[0]))))), 4)
                else:
                    # angle = np.round_((np.arccos(np.dot(p_v,v2)/(norm(p_v)*norm(v2)))), decimals=4)
                    if np.absolute(p_v[0]) < v2[0]:
                        angle = np.round(
                            (
                                np.arctan(
                                    np.absolute(p_v[1]) / (v2[0] - np.absolute(p_v[0]))
                                )
                            ),
                            4,
                        )
                    else:
                        angle = np.round(
                            (
                                np.arctan(
                                    np.absolute(p_v[1])
                                    / (np.absolute((np.absolute(p_v[0]) - v2[0])))
                                )
                            ),
                            4,
                        )

            angles.append(angle)
        return angles


    def convert_date(self, df):
        time_1 = df["period_time"].str.split(":", expand=True).astype(int)
        time_2 = df["last_period_time"].str.split(":", expand=True).astype(int)
        time_period_1 = (df["period"].astype(int) - 1) * 1200
        time_period_2 = (df["last_period"].astype(int) - 1) * 1200
        df["time_from_last"] = (time_1[0] * 60 + time_1[1] + time_period_1) - (
            time_2[0] * 60 + time_period_1[1] + time_period_2
        )
    

        return df


