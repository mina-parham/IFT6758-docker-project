"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from comet_ml import API
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import ift6758




#API_KEY = os.environ.get("COMET_API_KEY")


app = Flask(__name__)
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
API_KEY = 'c2REbE8eQaoRTP059ajV8VYn9'
global model_name
global model 

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    global model
    global model_name
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    with open(LOG_FILE, 'w'):
        pass


    # TODO: any other initialization before the first request (e.g. load default model)
    api = API(API_KEY)
    print(f'The api key is: {API_KEY}')

    model_name = "model_5_2.pickle"
    if not Path(model_name).is_file():
        api.download_registry_model("ift-6758-2", "xgb-model-5-2-pickle", "1.0.0", output_path="./", expand=True)    
        app.logger.info(f"succesfully dowload model: {model_name}")
        model = pickle.load(open(model_name, 'rb'))
        app.logger.info(f'succesfully load model ({model_name})')
    else:
        model = pickle.load(open(model_name, 'rb'))
        app.logger.info(f'succesfully loaded downloaded model ({model_name})')

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    with open("flask.log") as l:
    #raise NotImplementedError("TODO: implement this endpoint")
        response = l.readlines()
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    global model
    global model_name
    response = None
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)
    model_name = json['model_name']
    # TODO: check to see if the model you are querying for is already downloaded
    if Path(model_name).is_file():
        # TODO: if yes, load that model and write to the log about the model change.  
        # eg: app.logger.info(<LOG STRING>)
        app.logger.info(f"succesfully loaded downloaded model: {model_name}")
        model = pickle.load(open(model_name, 'rb'))
        response = model_name
    else:
        api = API(API_KEY)
        try:
            # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
            # about the model change. If it fails, write to the log about the failure and keep the 
            # currently loaded model
            api.download_registry_model(json['workspace'], json['model'], json['version'], output_path='./', expand=True)
            app.logger.info(f"succesfully download model:{model_name}")
            model = pickle.load(open(model_name, 'rb'))
            app.logger.info(f"succesfully load model: {model_name}")
            response = model_name
        except:
            response= None
            app.logger.warning(f"failed to download model{model_name}")

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    #raise NotImplementedError("TODO: implement this endpoint")

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)
    feature = []

    # TODO:
    df= pd.json_normalize(json)
    #raise NotImplementedError("TODO: implement this enpdoint")
    #df = pd.DataFrame([pd.read_json(json, typ='series')])
    df["shot_type"] = LabelEncoder().fit_transform(df["shot_type"])
    df["last_type"] = LabelEncoder().fit_transform(df["last_type"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    try:
        if model_name == 'model_5_2.pickle':
            feature = ['period', 'coordinate_x', 'coordinate_y', 'shot_type', 'distance', 'angle', 'last_type', 'last_coord_x', 'last_coord_y', 'time_from_last', 'from_last_distance', 'rebound',
            'change_angle', 'speed','power_play', 'number_friendly', 'number_opposing']
        
        elif model_name == 'model_5_3.pickle':
            feature = ['coordinate_x', 'coordinate_y', 'distance', 'angle', 'last_coord_x',
        'last_coord_y', 'time_from_last', 'from_last_distance']
    except:
        app.logger.info(f"The feature not defined for model:{model_name}")

    

    try:
        response = model.predict_proba(df[feature])[:,1]
        app.logger.info(f"The goal probability based on model: {model_name}, is {response}")
    
    except:
        response = None
        app.logger.warning(f"it's not possible to compute goal probability for model: {model_name}")

    print(response)
    app.logger.info(response)
    return jsonify(response.tolist())  # response must be json serializable!
