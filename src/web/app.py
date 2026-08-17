from fastapi import FastAPI
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from web.schemas import *
from configuration.config import *
from runner.predict import Predict
from web.service import TitlePredict
# ["辣西西里传统意面盒装500g*1盒","500ML百事可乐果缤纷蓝莓石榴"]
#lsof -ti :8000 | xargs kill -9杀进程指令
app=FastAPI()
# model,device,tokenizer
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=AutoModelForSequenceClassification.from_pretrained(MODELS_DIR/"best").to(device)
tokenizer=AutoTokenizer.from_pretrained(MODELS_DIR/"best")
predict=Predict(model,device,tokenizer)
titlepredict=TitlePredict(predict)
@app.post("/predict")
def predict(title:Title)->Category:
    cate=titlepredict.predict(title.title)
    return Category(cate=cate)

