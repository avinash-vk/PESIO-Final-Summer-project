#ACCURACY THAT I GOT: 0.7791897277032904

#imports
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
import warnings
warnings.filterwarnings("ignore") #To disable version warnings thrown by python
 
#reading the csvfile
df = pd.read_csv("weather.csv")
df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','RISK_MM'],axis=1)
df = df.dropna(how='any')
#table modification
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

df = shuffle(df) #shuffling the rows
y = df.iloc[:, 18] #rain tommorow
X = df.iloc[:, 2:18]  #rest of the dataset


#LabelEncoding
le = LabelEncoder()
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

X['WindGustDir'] = le.fit_transform(X['WindGustDir'])
X['WindDir9am'] = le1.fit_transform(X['WindDir9am'])
X['WindDir3pm'] = le2.fit_transform(X['WindDir3pm'])
X['RainToday'] = le3.fit_transform(X['RainToday'])

#Onehotencoder
ohe = OneHotEncoder(categorical_features=[3])   #windgustdirection
ohe1 = OneHotEncoder(categorical_features=[19]) #9am
ohe2 = OneHotEncoder(categorical_features=[33]) #3pm
ohe3 = OneHotEncoder(categorical_features=[55]) #Raintoday

X = ohe.fit_transform(X).toarray()
X = X[:,1:]
X = ohe1.fit_transform(X).toarray()
X = X[:,1:]
X = ohe2.fit_transform(X).toarray()
X = X[:,1:]
X = ohe3.fit_transform(X).toarray()
X = X[:,1:]

#splitting the data into training & test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Standardization
ss  = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
y_train = (y_train == 'Yes')

#Callback class to control output during training
class mycb(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Starting",epoch)

#Creating the datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#Creating a binary classifier model and defining its layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(55,input_shape=(X_train.shape[1],), activation = 'relu'))
model.add(tf.keras.layers.Dense(55,activation = 'relu'))
model.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))

#defining the model and training the dataset
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 8, batch_size = 10,verbose=0, callbacks=[mycb()])

#Accuracy of the model output
print('Accuracy of the model:',model.evaluate(X_test, y_test,verbose=0)[1])


