# System Architechure 
![image](https://github.com/ricasbp/IoT-TemperatureMonitoring/assets/59062659/60a17498-d8cd-451e-a87b-aae9751b8bc2)

# Description

The project involved developing an IoT application for monitoring and analyzing room temperature, using sensors inside and outside a room. The first stage required implementing the IoT architecture, analyzing historical temperature data, and creating a dashboard for statistical insights. In the second stage, machine learning models were trained to predict temperatures, and a live dashboard was created to display real-time and predicted temperatures, with evaluations of thermal comfort deviations. The project utilized MQTT for data communication, Node-RED for data processing and dashboard creation, and was deployed on Google Cloud Platform.


# How to Deploy and use

## 1. Use Docker containers to build and deploy the application's containers

`docker-compose up` or, in detached mode `docker-compose up -d`

## 2. Use Guest or Add User

### 2.1. Login with user "guest" and password "guest"

### 2.2. Add new User
In the ./data/node_red folder, open config.users.json and add your desired user.
To generate hash - `sh ./utils/hash_pwd.sh`
