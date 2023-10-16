#!/usr/bin/env python
# coding: utf-8

import requests


url = "http://localhost:9696/predict"
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
requests.post(url, json=client).json()

response = requests.post(url, json=client).json()
print(response)

if response['churn'] == True:
    print('Will give credit')
else:
    print('No credit')