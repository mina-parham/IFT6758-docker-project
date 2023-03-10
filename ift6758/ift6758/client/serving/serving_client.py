import json
import requests
import pandas as pd
import logging
import os


logger = logging.getLogger(__name__)
APP = os.environ.get("APP")


class ServingClient:
    def __init__(self, ip_address: str = APP, port: int = 5500, features=None):
        self.base_url = f"http://{ip_address}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        try:
            r = requests.post(
                f"{self.base_url}/predict", 
                json=json.loads(X.to_json()))
                
            logger.info(f"successfully return the prediction")
            return r.json()
        except:
            return None

        #raise NotImplementedError("TODO: implement this function")

    def logs(self) -> dict:
        """Get server logs"""
        try:
            r = requests.get(
                f"{self.base_url}/logs")
                
            logger.info(f"successfully return the logs")
            return r.json()
        except:
            return None
        #raise NotImplementedError("TODO: implement this function")

    def download_registry_model(self, workspace: str, model: str, version: str, model_name: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        try:
            r = requests.post(
                f"{self.base_url}/download_registry_model", 
                json={'workspace': workspace, 'model': model, 'version': version, 'model_name': model_name })
                
            logger.info(f"successfully download the registry model")
            return r.json()
        except:
            return None


        #raise NotImplementedError("TODO: implement this function")
