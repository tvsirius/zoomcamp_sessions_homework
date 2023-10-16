import pickle
import numpy as np


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)




customer = {"job": "retired", "duration": 445, "poutcome": "success"}

prediction = predict_single(customer, dv, model)

print('prediction: %.3f' % prediction)
