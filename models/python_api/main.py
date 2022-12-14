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
    current_temperature_inside: float

class BatchRequest(BaseModel):
    """ Base Model for Batched temperature Requests. """

    temperature_sequence_outside: List[float]
    current_temperature_inside: float

setattr(__main__, "LSTM", LSTM)
setattr(__main__, "CustomRandomForestRegressor", CustomRandomForestRegressor)

with open("./outside_model/model_v3.1-torch","rb") as o:
    outside_model: LSTM = \
        torch.load(o, encoding='bytes', map_location="cpu")

print("Outside Model:")
print(outside_model)
outside_model.set_device("cpu")
print(outside_model.predict(np.array([
	24.25,
	24.187,
	15.367,
	14.383,
	24.5,
	13.567,
	24.25,
	13.156,
	24.562,
	24.562,
	12.922,
	24.312,
	12.189,
	24.0,
	12.283,
	24.25,
	12.072,
	23.937,
	11.411,
	23.875,
	11.5,
	23.875,
	11.656,
	11.528,
	23.812,
	11.144,
	23.875,
	11.306,
	23.687,
	23.562,
	11.028,
	10.111,
	23.375,
	9.1,
	23.5,
	8.772,
	23.625,
	9.106,
	23.625,
	10.1,
	23.625,
	23.625,
	9.056,
	9.483,
	23.562,
	23.562,
	8.922,
	8.511000000000001,
	23.562,
	8.078,
	23.562,
	7.9,
	23.437,
	23.437,
	7.494,
	23.5,
	7.45,
	7.794,
	23.375,
	23.5,
	8.161,
	23.437,
	8.111,
	23.375,
	8.328,
	8.45,
	23.562,
	23.812,
	8.667,
	23.937,
	8.95,
	9.728,
	24.062,
	11.511,
	24.062,
	24.062,
	13.528,
	14.489,
	24.25,
	24.5,
	15.25,
	16.044,
	24.625,
	17.344,
	24.625,
	18.489,
	24.812,
	18.033,
	24.875,
	17.878,
	24.437,
	18.678,
	24.5,
	17.383,
	24.25,
	17.194000000000003,
	24.437,
	24.687,
	16.910999999999998,
	24.562,
	15.161,
	24.375,
	13.678,
	12.922,
	24.437,
	12.806,
	24.25,
	24.25,
	13.361,
	24.062,
	12.956,
	23.937,
	12.839,
	24.0,
	12.683,
	24.0,
	12.578,
	12.606,
	23.937,
	12.717,
	23.875,
	12.55,
	23.937,
	12.233,
	23.687,
	12.7,
	23.562,
	12.906,
	23.562,
	12.422,
	23.437,
	11.856,
	23.562,
	11.005999999999998,
	23.625,
	10.811,
	23.625,
	10.239,
	23.625,
	10.006,
	23.625,
	9.833,
	23.687,
	10.072,
	23.625,
	23.562
])))

print("Inside Model:")
with open("./inside_model/model_v3","rb") as o:
    inside_model: CustomRandomForestRegressor = pickle.load(o, encoding='bytes')
print(inside_model)


app = FastAPI()

@app.post("/single")
async def single_prediction(body: SingleRequest):
    """ Takes in a single inside and outside temperature as input and predicts the posteriors. """

    curr_temp_out = body.current_temperature_outside
    curr_temp_in = body.current_temperature_inside

    outside_preds = outside_model.predict(np.array(curr_temp_out))
    inside_preds = inside_model.total_predict(curr_temp_out, outside_preds, curr_temp_in)

    outside_preds = [ float(o[0]) for o in outside_preds ]
    inside_preds = [ float(o[0]) for o in inside_preds ]
    return JSONResponse(content=jsonable_encoder(
        {
            "OutsidePreds": outside_preds,
            "InsidePreds": inside_preds
        }
    ))

@app.post("/batch")
async def batch_prediction(body: BatchRequest):
    """ Takes in a batch of outside temperatures and a single inside temperature
    as input and predicts the posteriors. """

    temp_seq_out = body.temperature_sequence_outside
    curr_temp_in = body.current_temperature_inside

    outside_preds = outside_model.predict(np.array(temp_seq_out))
    inside_preds = inside_model.total_predict(
                        temp_seq_out[-1],
                        outside_preds,
                        curr_temp_in
                    )

    outside_preds = [ float(o[0]) for o in outside_preds ]
    inside_preds = [ float(o[0]) for o in inside_preds ]
    return JSONResponse(content=jsonable_encoder(
        {
            "OutsidePreds": outside_preds,
            "InsidePreds": inside_preds
        }
    ))

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=9090, log_level="info")
