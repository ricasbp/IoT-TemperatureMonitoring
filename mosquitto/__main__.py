""" Mosquitto Sensor Script """

import time
from datetime import datetime

import pandas as pd
import paho.mqtt.client as mqtt

# This is the Publisher


#https://www.ev3dev.org/docs/tutorials/sending-and-receiving-messages-with-mqtt/

client = mqtt.Client()
client.connect('34.175.93.216',1883,60)
client.publish("idc/fc55309/truncate", 0)
time.sleep(1)
# reading from the file

df = pd.read_csv("./data/datasets/online.data")

df.sort_values(["date"])

df_in = df.loc[df["label"] == "inside"]
df_in = df_in.drop(["label"], axis=1)

df_out = df.loc[df["label"] == "outside"]
df_out = df_out.drop(["label"], axis=1)


ix_in = 0
ix_out = 0
while ix_in < len(df_in) or ix_out < len(df_out):

    if ix_in < len(df_in):
        in_val = df_in.to_numpy()[ix_in]
        in_date = in_val[0]
        in_temp = in_val[1]
        req_str_in = f"{in_date},{in_temp},inside"
        print(req_str_in)
    else:
        in_date = "2999-12-30 12:30:00"

    if ix_out < len(df_out):
        out_val = df_out.to_numpy()[ix_out]
        out_date = out_val[0]
        out_temp = out_val[1]
        req_str_out = f"{out_date},{out_temp},outside"
        print(req_str_out)
    else:
        out_date = "2999-12-30 12:30:00"




    if datetime.strptime(in_date, f"%Y-%m-%d %H:%M:%S") == datetime.strptime(out_date, f"%Y-%m-%d %H:%M:%S"):
        client.publish("idc/fc55309/insert", req_str_out)
        client.publish("idc/fc55309/insert", req_str_in)
        ix_in += 1
        ix_out += 1
    elif datetime.strptime(in_date, f"%Y-%m-%d %H:%M:%S") >= datetime.strptime(out_date, f"%Y-%m-%d %H:%M:%S"):
        client.publish("idc/fc55309/insert", req_str_out)
        ix_out += 1
    elif datetime.strptime(in_date, f"%Y-%m-%d %H:%M:%S") <= datetime.strptime(out_date, f"%Y-%m-%d %H:%M:%S"):
        client.publish("idc/fc55309/insert", req_str_in)
        ix_in += 1
    else:
        raise Exception()
    time.sleep(3)
client.disconnect()