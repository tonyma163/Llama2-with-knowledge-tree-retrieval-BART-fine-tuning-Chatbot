# NLP

## Team
Ma Man To Tony 23035732g
Lam Shu Yu Jack 23026067g
Tam Wai Yin Leonidas 23064207g

## System Structure
The system contain the server and web application.

User could interact with the system from the web application through the local api endpoint hosted on the server.

The server contain loading the required models, loading the required knowledge_set file, and host the api endpoint for communication.

Please ensure you have download the required environmental dependencies and packages.

## Prerequisite
NodeJs installed
pip installed
python installed
knowledge_set in the server directory

## Install Required Packages
```
pip install transformers ctransformers accelerate datasets peft trl jieba fastapi uvicorn
```

## Install Webapp Packages
```
cd app
npm i
```

## Run Server at localhost:8000
```
uvicorn server:app --reload
```

## Run Webapp at localhost:3000
```
cd app
npm run dev
```
