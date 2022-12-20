import streamlit as st
import pandas as pd
import numpy as np
from comet_ml import API
import json
import os
import requests
import datetime

st.title('Hockey Visualization App')

with st.sidebar:
    workspace = st.text_input('Workspace', 'ift-6758-2')
    st.write('The selected CometML workspaces is:', workspace)

    model = st.selectbox('Model', ('xgb-model-5-2-pickle','xgb-model-5-3-pickle'))
    st.write('The selected model is:', model)

    version = st.text_input('Version', '1.0.0')
    st.write('The selected version is:', version)

    if st.button('Get Model'):
        #st.write('Retrieve Model')

        api = API()
        modelUsed = api.download_registry_model(workspace, model, version,output_path="./trained_models", expand=True)

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

            def current_datetime():
                date = datetime.datetime.utcnow()
                year = date.year
                month = date.month
                day = date.day
                hour = date.hour
                minute = date.minute
                second = date.second
                f_date = str(year)+str(month).zfill(2)+str(day).zfill(2)+"_"+str(hour).zfill(2)+str(minute).zfill(2)+str(second).zfill(2)
                return f_date

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


            team1 = gameData['gameData']['teams']['home']['name']
            team2 = gameData['gameData']['teams']['away']['name']
            period = str(gameData['liveData']['plays']['allPlays'][-1]['about']['period'])
            timeLeftP = str(gameData['liveData']['plays']['allPlays'][-1]['about']['periodTimeRemaining'])
            tm1Score = liveData[-1]['about']['goals']['home']
            tm2Score = liveData[-1]['about']['goals']['away']
            tm1XG = 3.2
            tm2XG = 1.4

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

            df = pd.DataFrame()

            df['Name'] = ['Ankit', 'Ankita', 'Yashvardhan']
            df['Articles'] = [97, 600, 200]
            df['Improved'] = [2200, 75, 100]

            st.dataframe(df)
