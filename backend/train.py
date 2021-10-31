import numpy as np
import librosa
import soundfile
import os
import glob
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

total_emotions = {
    '01' : 'neutral',
    '02' : 'calm',
    '03' : 'happy',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fearful',
    '07' : 'disgust',
    '08' : 'surprised'
}

observe = ['angry', 'happy', 'sad', 'neutral']

# file name, mel spectrogram, mel coefficients, chroma waveform, 
def extract(f_name, mel, mel_coeff, chroma):
    # open sound file, read it
    # create numpy array for librosa features if exists on sound file
    with soundfile.SoundFile(f_name) as f:
        processed = f.read(dtype="float32")
        sample_rate = f.samplerate
        # stack input arrays horizontally for each feature into one data point, stft
        features = np.array([])
        if chroma:
            s_fourier = np.abs(librosa.stft(processed))
        if mel_coeff:
            mel_coeff_s = np.mean(librosa.feature.mfcc(y = processed, sr = sample_rate, n_mfcc = 40).T, axis = 0)
            features = np.hstack((features, mel_coeff_s))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S = s_fourier, sr = sample_rate).T, axis = 0)
            features = np.hstack((features, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(processed, sr = sample_rate).T, axis = 0)
            features = np.hstack((features, mel))
    return features
    
# load data from ravdess data set, file name structure below
# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
# Vocal channel (01 = speech, 02 = song).
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
# Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
def load_data(test_size = 0.2):
    features = []
    emotions = []
    for f in glob.glob("..\\data\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        f_name = os.path.basename(f)
        emotion = total_emotions[f_name.split("-")[2]]
        if emotion not in observe:
            continue
        feature = extract(f, mel = True, mel_coeff = True, chroma = True)
        features.append(feature)
        emotions.append(emotion)
    return train_test_split(np.array(features), emotions, test_size = test_size, random_state=9)

if __name__ == '__main__':
    # create train/test sets
    features_train, features_test, emotions_train, emotions_test = load_data(test_size = 0.25)

    # check shape/features
    print((features_train.shape[0], features_test.shape[0]))
    print(features_train.shape[1])

    # make model and train
    model = MLPClassifier(alpha = 0.01, batch_size = 32, epsilon = .00000001, hidden_layer_sizes = (300, ), learning_rate = 'adaptive', max_iter = 1000)
    model.fit(features_train, emotions_train)

    # predicted emotions for test set using model
    emotions_predict = model.predict(features_test)

    # calculate accuracy of model's predictions
    acc = accuracy_score(y_true = emotions_test, y_pred = emotions_predict)
    print(acc)

    # save/write model to directory
    pickle.dump(model, open("mlp.model", "wb"))
