import librosa as lb
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt

"Diverse path-uri pentru fieserele de Train, Test si Validation"
path = "C:/Users/eduar/Desktop/Anul 2/Semestrul 2/ML/Proiect"
path_test = path + "/data/test/test"
path_train = path + "/data/train/train"
path_validation = path + "/data/validation/validation"

"Citim cu pandas csv urile fara headere"
test = pd.read_csv("C:/Users/eduar/Desktop/Anul 2/Semestrul 2/ML/Proiect/data/test.txt", header=None)
train = pd.read_csv("C:/Users/eduar/Desktop/Anul 2/Semestrul 2/ML/Proiect/data/train.txt", header=None)
validation = pd.read_csv("C:/Users/eduar/Desktop/Anul 2/Semestrul 2/ML/Proiect/data/validation.txt", header=None)

"O lista de nume de fisere pentru scrierea predictiilor in csv"
file_list = []


def normalize(train_features, test_features):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_features)
    scaled_train_feats = scaler.transform(train_features)
    scaled_test_feats = scaler.transform(test_features)

    return scaled_train_feats, scaled_test_feats


def get_features_from_train(row):
    file_name = os.path.join(os.path.abspath(path_train), str(row[0]))
    audio_train, sfreq_train = lb.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(lb.feature.mfcc(y=audio_train, sr=sfreq_train, n_mfcc=40).T, axis=0)

    labels = row[1]
    return pd.Series([mfccs, labels])


def get_features_from_test(row):
    file_name = os.path.join(os.path.abspath(path_test), str(row[0]))
    file_list.append(str(row[0]))
    audio_test, sfreq_test = lb.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(lb.feature.mfcc(y=audio_test, sr=sfreq_test, n_mfcc=40).T, axis=0)

    return pd.Series([mfccs])


def get_features_from_validation(row):
    file_name = os.path.join(os.path.abspath(path_validation), str(row[0]))
    audio_val, sfreq_val = lb.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(lb.feature.mfcc(y=audio_val, sr=sfreq_val, n_mfcc=40).T, axis=0)

    labels = row[1]
    return pd.Series([mfccs, labels])

"Aici facem extragerea features din fisierele date si le salvam ca fisiere de tip pickle pentru a fi citite mai repede"
# "Pentru fiecare rand(axis=1) extragem cate un feature"
# "Train"
# train_data = train.apply(get_features_from_train, axis=1)
# train_data.columns = ['feature', 'label']
# train_data.to_pickle("./train.pkl")
#
# "Test"
# test_data = test.apply(get_features_from_validation, axis=1)
# test_data.columns = ['feature']
# test_data.to_pickle("./test.pkl")
#
# file_names = pd.Series(file_list)
# file_names.to_pickle("./file_names.pkl")
#
# "Validation"
# validation_data = validation.apply(get_features_from_validation, axis=1)
# validation_data.columns = ['feature', 'label']
# validation_data.to_pickle("./validation.pkl")

"Aici incepe preluarea datelor din fieserele pickle"

"Train"
"Citim cu pandas si transformam in nparray"
train_data = pd.read_pickle("./train.pkl")
x = np.array(train_data.feature.tolist())
y = np.array(train_data.label.tolist())
y_binary = to_categorical(y)

"Test"
train_data = pd.read_pickle("./test.pkl")
x_tests = np.array(train_data.feature.tolist())

file_names = pd.read_pickle("./file_names.pkl")
file_names = list(file_names)

"Validation"
val_data = pd.read_pickle("./validation.pkl")
x_val = np.array(val_data.feature.tolist())
y_val = np.array(val_data.label.tolist())
y_val_binary = to_categorical(y_val)

"Normalizam"
x_std, x_val_std = normalize(x, x_val)
# x_std, x_tests_std = normalize(x, x_tests)

"Incepem MLP ul"

"Initial folosim un model Sequential pentru ca dorim sa facem un stack de layere cu un singur input si un singur output"
"MLP ul meu are 3 hidden layere"
model = Sequential()

"Primul layer are 512 output neurons"
"Deoarece x_train are shape ul (8000, 40) vom folosi ca input (40, )"
"Se activeaza folosind Rectified Linear Unit si are un dropout de 0.4 pentru a reduce overfitting-ul"
model.add(Dense(512, input_shape=(40, )))
model.add(Activation('relu'))
model.add(Dropout(0.4))

"Layer ul 2 are tot 512 output neurons, aceasi activare si acelasi dropout"
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.4))

"Layerul final, cel de output are 2 output neurons deoarece avem 2 label uri in problema noastra."
"Am folosit activarea softmax fara dropout"
model.add(Dense(2))
model.add(Activation('softmax'))


model.summary()
"Am incercat sa mai optimizez Adam ul dar imi da un rezultat mai slab"
# opt = Adam(learning_rate=0.0001)

"Configuram modelul si incepem sa il antrenam"
"Am folosit binary_cossentropy pentru ca avem doar 2 clase."
"CU Adam am luat cea mai buna acuratete"
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

num_epochs = 50#Un numar prea mare deoarece dadea overfit. Puteam incerca si un early stop
num_batch_size = 64

history_mlp = model.fit(x_std, y_binary, batch_size=num_batch_size, validation_data=(x_val_std, y_val_binary), epochs=num_epochs, verbose=1)

"Acum incepe plotarea pentru acuratea si loss a mlp ului"
plt.plot(history_mlp.history['accuracy'])
plt.plot(history_mlp.history['val_accuracy'])
plt.title("Acuratetea MLP-ului")
plt.ylabel("Acuratete")
plt.xlabel("Numarul de epoci")
plt.legend(['train', 'validation'])
plt.show()

plt.plot(history_mlp.history['loss'])
plt.plot(history_mlp.history['val_loss'])
plt.title("Loss-ul MLP-ului")
plt.ylabel("Loss")
plt.xlabel("Numarul de epoci")
plt.legend(['train', 'validation'])
plt.show()

"Scoatem predictiile pentru a calcula acc_score, f1, recall, precision si confusion matrix"
y_pred = model.predict_classes(x_val_std)

accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
c_matrx = confusion_matrix(y_val, y_pred)

print(f"accuracy: {accuracy}, f1_score: {f1}, recall: {recall} si precision: {precision}")
print("Matricea de confuzie:")
print(c_matrx)

"Scoatem predictiile si le scriem in csv cu ajutorul pandas"
y_pred = model.predict_classes(x_tests)

df = pd.DataFrame({'name': file_names[1:], 'label': y_pred})
df.to_csv('predictions.csv', index=False)

"Modelul pentru o comparatie"
"Am folosit un svm cu singurul parametru schimbat fiind C = 10 si un kernel linear"
svm = SVC(kernel="linear", C=10)

svm.fit(x_std, y)

y_pred = svm.predict(x_val_std)


"Scoatem predictiile pentru a calcula acc_score, f1, recall, precision si confusion matrix"

accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
c_matrx = confusion_matrix(y_val, y_pred)

print("Pentru SVM:")
print(f"accuracy: {accuracy}, f1_score: {f1}, recall: {recall} si precision: {precision}")
print("Matricea de confuzie:")
print(c_matrx)