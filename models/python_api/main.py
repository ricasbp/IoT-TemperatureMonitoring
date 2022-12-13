""" Fastapi service to compute the ML temperature prediction models. """

import pickle
from typing import List
import __main__

import torch
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from models import LSTM, CustomRandomForestRegressor

class SingleRequest(BaseModel):
    """ Base Model for Single temperature Requests. """

    current_temperature_outside: float
    current_temperature_inside: float = None
    current_datetime: str

class BatchRequest(BaseModel):
    """ Base Model for Batched temperature Requests. """

    temperature_sequence_outside: List[float]
    current_temperature_inside: float = None
    current_datetime: str

setattr(__main__, "LSTM", LSTM)
setattr(__main__, "CustomRandomForestRegressor", CustomRandomForestRegressor)

with open("./outside_model/model_v3","rb") as o:
    outside_model: LSTM = \
        torch.load("./outside_model/model_v3", encoding='bytes', map_location="cpu")

print("Outside Model:")
print(outside_model)
outside_model.set_device("cpu")

print("Inside Model:")
with open("./inside_model/model_v3","rb") as o:
    inside_model: CustomRandomForestRegressor = pickle.load(o, encoding='bytes')
print(inside_model)


app = FastAPI()

@app.post("/single")
async def single_prediction(body: SingleRequest):
    """ Takes in a single inside and outside temperature as input and predicts the posteriors. """

    curr_temp_out = body.current_temperature_outside

    with_inside = True
    if body.current_temperature_inside is None:
        with_inside = False
    else:
        curr_temp_in = body.current_temperature_inside

    outside_preds = outside_model.predict(np.array(curr_temp_out))
    if with_inside:
        inside_preds = inside_model.total_predict(curr_temp_out, outside_preds, curr_temp_in)

    outside_preds = [ float(o[0]) for o in outside_preds ]
    if with_inside:
        inside_preds = [ float(o[0]) for o in inside_preds ]
        return JSONResponse(content=jsonable_encoder(
            {
                "OutsidePreds": outside_preds,
                "InsidePreds": inside_preds,
                "current_temperature_outside": curr_temp_out,
                "current_temperature_inside": curr_temp_in,
                "current_datetime": body.current_datetime
            }
        ))
    else:
        return JSONResponse(content=jsonable_encoder(
            {
                "OutsidePreds": outside_preds,
                "current_temperature_outside": curr_temp_out,
                "current_datetime": body.current_datetime
            }
        ))

@app.post("/batch")
async def batch_prediction(body: BatchRequest):
    """ Takes in a batch of outside temperatures and a single inside temperature
    as input and predicts the posteriors. """

    temp_seq_out = body.temperature_sequence_outside

    with_inside = True
    if body.current_temperature_inside is None:
        with_inside = False
    else:
        curr_temp_in = body.current_temperature_inside

    outside_preds = outside_model.predict(np.array(temp_seq_out))
    if with_inside:
        inside_preds = inside_model.total_predict(
                            temp_seq_out[-1],
                            outside_preds,
                            curr_temp_in
                        )

    outside_preds = [ float(o[0]) for o in outside_preds ]
    if with_inside:
        inside_preds = [ float(o[0]) for o in inside_preds ]
        return JSONResponse(content=jsonable_encoder(
            {
                "OutsidePreds": outside_preds,
                "InsidePreds": inside_preds,
                "temperature_sequence_outside": temp_seq_out,
                "current_temperature_inside": curr_temp_in,
                "current_datetime": body.current_datetime
            }
        ))
    else:
        return JSONResponse(content=jsonable_encoder(
            {
                "OutsidePreds": outside_preds,
                "temperature_sequence_outside": temp_seq_out,
                "current_datetime": body.current_datetime
            }
        ))

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=9090, log_level="info")
