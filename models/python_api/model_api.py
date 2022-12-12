""" Fastapi service to compute the ML temperature prediction models. """

import torch
import uvicorn
from fastapi import FastAPI

from models import LSTM, CustomRandomForestRegressor

with open("./outside_model/model_v3","rb") as o:
    outside_model: LSTM = torch.load(o, encoding='bytes')

print("Outside Model:")
print(outside_model)
outside_model.set_device()


print("Inside Model:")
with open("./inside_model/model_v3","rb") as o:
    inside_model: CustomRandomForestRegressor = pickle.load(o, encoding='bytes')
print(inside_model)


app = FastAPI()

@app.post("/single")
async def single_pred(msg):
    """ Takes in a single temperature as input and predicts the posteriors. """
    print(str(msg))
    return msg

@app.post("/batch")
async def batch_pred(msg):
    """ Takes in a batch of temperatures as input and predicts the posteriors. """
    print(str(msg))
    return msg

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=9090, log_level="info")
