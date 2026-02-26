import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "ambient": 25,
    "coolant": 40,
    "u_d": 0.5,
    "u_q": 1.2,
    "motor_speed": 1500,
    "i_d": 2.1,
    "i_q": 3.5
}

response = requests.post(url, json=data)
print("Predicted PM Temperature:", response.json())
