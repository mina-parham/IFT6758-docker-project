import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import requests
import sys

from ift6758.ift6758.client.serving.serving_client import ServingClient
from ift6758.ift6758.client.game.game_client import GameClient

#if 'game_client' not in st.session_state:
   # st.session_state.game_client = GameClient()
if 'serving_client' not in st.session_state:
    st.session_state.serving_client = ServingClient()

st.title('Hockey Visualization App')

with st.sidebar:
    workspace = st.text_input('Workspace', 'ift-6758-2')
    st.write('The selected CometML workspaces is:', workspace)

    model = st.selectbox('Model', ('xgb-model-5-2-pickle','xgb-model-5-3-pickle'))
    st.write('The selected model is:', model)

    version = st.text_input('Version', '1.0.0')
    st.write('The selected version is:', version)
    
    sc= ServingClient()
    if st.button('Get Model'):
        #st.write('Retrieve Model')
        if model == 'xgb-model-5-2-pickle':
            st.session_state.serving_client.download_registry_model(workspace=workspace, model=model, version=version, model_name='model_5_2.pickle')
        elif model == 'xgb-model-5-3-pickle':
            st.session_state.serving_client.download_registry_model(workspace=workspace, model=model, version=version, model_name='model_5_3.pickle')
            
with st.container():
    gameID = st.text_input('Game ID', '2021020329')

    if 'count' not in st.session_state:
        st.session_state.count = 0

    def increment_counter():
        st.session_state.count +=1

    if st.button('Ping Game',on_click=increment_counter):
        st.write('Connect to game pining module')

        with st.container():
            def dataHelper(url):
                response = requests.get(url)
                if response.status_code != 404:
                    game_response = requests.get(url)
                    game_content = json.loads(game_response.content)
                return game_content


            def update_event():
                st.session_state.last_event = ((len(gameData['liveData']['plays']['allPlays'])-1))
                
            if st.session_state.count == 1:
                url = "https://statsapi.web.nhl.com/api/v1/game/"+gameID+"/feed/live/"
                gameData = dataHelper(url)
                liveData = gameData['liveData']['plays']['allPlays']
                if 'lat_event' not in st.session_state:
                    st.session_state.last_event = (len(gameData['liveData']['plays']['allPlays'])-1)


            elif st.session_state.count != 1:
                url = "https://statsapi.web.nhl.com/api/v1/game/"+gameID+"/feed/live/"
                gameData = dataHelper(url)
                liveData = gameData['liveData']['plays']['allPlays'][st.session_state.last_event:]
                st.session_state.last_event = update_event()
                
            gc = GameClient()
            team1 = gameData['gameData']['teams']['home']['name']
            team2 = gameData['gameData']['teams']['away']['name']
   

            if model == 'xgb-model-5-2-pickle':
                feature = ['period', 'coordinate_x', 'coordinate_y', 'shot_type', 'distance', 'angle', 'last_type', 'last_coord_x', 'last_coord_y', 'time_from_last', 'from_last_distance', 'rebound','change_angle', 'speed','power_play', 'number_friendly', 'number_opposing']
                f_display = ['event_idx','team_name','period', 'coordinate_x', 'coordinate_y', 'shot_type', 'distance', 'angle', 'last_type', 'last_coord_x', 'last_coord_y', 'time_from_last', 'from_last_distance', 'rebound','change_angle', 'speed','power_play', 'number_friendly', 'number_opposing']
                df = gc.ping_game(gameID)
                df = df.dropna()
                df_prediction = df[feature]
                df_display = df[f_display]
                predictions = st.session_state.serving_client.predict(df_prediction)
                df_display['xG_Predicted'] = predictions
                home_xG_s = df_display.loc[df_display['team_name'] == team1]
                home_xg = float(np.round((home_xG_s['xG_Predicted'].sum()),2)) 
                away_xG_s = df_display.loc[df_display['team_name'] == team2]
                away_xg = float(np.round((away_xG_s['xG_Predicted'].sum()),2)) 
            elif model == 'xgb-model-5-3-pickle':
                feature = ['coordinate_x', 'coordinate_y','distance', 'angle', 'last_coord_x', 'last_coord_y', 'time_from_last', 'from_last_distance']
                f_display = ['event_idx','team_name','coordinate_x', 'coordinate_y','distance', 'angle', 'last_coord_x', 'last_coord_y', 'time_from_last', 'from_last_distance']
                df = gc.ping_game(gameID)
                df = df.dropna()
                df_prediction = df[feature]
                df_display = df[f_display]
                predictions = st.session_state.serving_client.predict(df_prediction)
                df_display['xG_Predicted'] = predictions
                home_xG_s = df_display.loc[df_display['team_name'] == team1]
                home_xg = float(np.round((home_xG_s['xG_Predicted'].sum()),2)) 
                away_xG_s = df_display.loc[df_display['team_name'] == team2]
                away_xg = float(np.round((away_xG_s['xG_Predicted'].sum()),2)) 


            
            period = str(gameData['liveData']['plays']['allPlays'][-1]['about']['period'])
            timeLeftP = str(gameData['liveData']['plays']['allPlays'][-1]['about']['periodTimeRemaining'])
            tm1Score = liveData[-1]['about']['goals']['home']
            tm2Score = liveData[-1]['about']['goals']['away']
            tm1XG = home_xg
            tm2XG = away_xg

            st.header('Game '+gameID+" : "+team1+" vs "+team2)
            st.write('Period '+period+" - "+timeLeftP+" left" )
            st.write('Last event',str(len(gameData['liveData']['plays']['allPlays'])-1))
            col1, col2 = st.columns(2)
            with col1:
               st.metric(label=(team1+" xG(actual)"), value=(str(tm1XG)+" "+"("+str(tm1Score)+")"), delta=(np.round((tm1Score-tm1XG),2)))
            with col2:
                st.metric(label=(team2+" xG(actual)"), value=(str(tm2XG)+" "+"("+str(tm2Score)+")"), delta=(np.round((tm2Score-tm2XG),2)))

        with st.container():
            st.header('Data used for predictions (and predictions)')
            st.dataframe(df_display)
            

