import requests
import numpy as np 

dummy_input = np.zeros((1, 3, 224, 224), dtype = np.float32)

res = requests.post(
    "http://localhost:8000/predict",
    json = {"data": dummy_input.tolist()}
)


print("Status", res.status_code)
print("Output:", res.text)


