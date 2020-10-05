import numpy as np
from flask import Flask, request, jsonify ,render_template,g
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from pandas.tseries.offsets import DateOffset
import csv


app = Flask(__name__,static_url_path='', 
            static_folder='static',
            template_folder='templates')

model = pickle.load(open('elearn.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    Log = request.form["file"]
    data = LoadPastDataset(Log)
    data1  = pd.read_csv(r'{0}'.format(Log),sep=",",usecols=(0,1,2,3))
    pred = model.predict(data)
    print(pred)
    data1['Predicitons'] = pred
    Pass = data1.loc[data1.Predicitons==0].drop(['Predicitons'], axis=1)
    Pass['Result'] = 'Pass'
    Fail = data1.loc[data1.Predicitons==1].drop(['Predicitons'], axis=1)
    Fail['Result'] = 'Fail'
    With = data1.loc[data1.Predicitons==2]
    Disc = data1.loc[data1.Predicitons==3]
    
    print(Pass)
    return render_template('index.html',tables=[Pass.to_html(classes='Pass'), Fail.to_html(classes='Fail')],titles = ['na'], P=len(Pass), F=len(Fail),
    T=len(data1),W=len(With),D=len(Disc),G =len(Disc) + len(Pass),  logName=Log)
    

from sklearn.preprocessing import StandardScaler

def LoadPastDataset(log):
    data  = pd.read_csv(r'{0}'.format(log),sep=",",usecols=(0,1,2,3))
    X = data.iloc[:,0:4].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    return X_train



if __name__ == '__main__':
    app.run(debug=True)