# NLP

## Prerequisite
NodeJs installed
pip installed
python installed

## Install Required Packages
```
pip install transformers bitsandbytes accelerate datasets peft trl fastapi uvicorn
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
