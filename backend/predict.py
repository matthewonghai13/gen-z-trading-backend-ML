from train import extract
import os
import pickle

path = os.path.abspath('mlp.model')

print(os.listdir())
print(os.getcwd())
print('sfdsfsfdsfsdfdsfsdfsfds')

# load saved model 
model = pickle.load(open(path, "rb"))
f_name = "recording.wav"
features = extract(f_name, mel = True, mel_coeff = True, chroma = True).reshape(1, -1)

# predict
result = model.predict(features)[0]

print("emotion:", result)