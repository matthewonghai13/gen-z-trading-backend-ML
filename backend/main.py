from flask import Flask, request
from flask_restful import Api, Resource
import sys
# from predict import predict
from train import extract
import os
import pickle

app = Flask(__name__)
api = Api(app)

@app.route("/audio", methods = ["POST"])
def post():
    video = request.data
    newVideo = open("./recording.wav", "wb")
    newVideo.write(video)
    
    path = os.path.abspath('mlp.model')

    # load saved model 
    model = pickle.load(open(path, "rb"))
    f_name = "recording.wav"
    features = extract(f_name, mel = True, mel_coeff = True, chroma = True).reshape(1, -1)

    # predict
    result = model.predict(features)[0]

    print("emotion:", result)
    return result

if __name__ == "__main__":
    app.run(debug=True)