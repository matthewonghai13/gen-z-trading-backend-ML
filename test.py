import requests

BASE = "http://127.0.0.1:5000/"

file = open('./backend/recordingold.wav', 'rb')
# files = {'video': file}
req = requests.post(BASE + "audio", data=file)

print(req.status_code)
print(req.text)