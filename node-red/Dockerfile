FROM nodered/node-red:2.2.3-14

USER root



RUN apk update


# Install python/pip
ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 python3-dev && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools

RUN echo "http://dl-8.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
RUN apk --no-cache --update-cache add gcc gfortran build-base wget freetype-dev libpng-dev openblas-dev
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

# Install deps
RUN pip3 install numpy
#RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install torch torchvision torchaudio


RUN pip3 install fastapi uvicorn[standard]


USER node-red
#EXPOSE 1880
