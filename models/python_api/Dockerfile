FROM continuumio/miniconda3:4.10.3-alpine

RUN apk update
RUN apk upgrade


RUN conda install numpy
RUN conda install -c pytorch pytorch torchvision torchaudio
RUN pip install -U scikit-learn==1.1.2

RUN pip install fastapi
RUN pip install uvicorn[standard]


WORKDIR /app

ADD main.py main.py
ADD models.py models.py
ADD inside_model inside_model
ADD outside_model outside_model

ENV PYTHONUNBUFFERED=0
ENTRYPOINT [ "uvicorn", "--host", "0.0.0.0","main:app", "--port", "9090" ]
