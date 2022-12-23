#!/bin/bash
docker container run -e COMET_API_KEY=$COMET_API_KEY -p 5500:5500 ift6758/serving:1.0.0
#"TODO: fill in the docker run command"
