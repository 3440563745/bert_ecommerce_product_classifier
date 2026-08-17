import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from configuration.config import *
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
class Predict:
    def __init__(self,model,device,tokenizer):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer

    def predict(self,text:str|list):
        flag=isinstance(text,str)
        if flag:
            text=list(text)
        inputs=self.tokenizer(text,padding=True,truncation=True,return_tensors="pt")
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        outputs=self.model(**inputs)
        logits=outputs.logits
        predictions=torch.argmax(logits,dim=-1).tolist()
        predictions=[self.model.config.id2label[prediction] for prediction in predictions]
        if flag:
            return predictions[0]
        else:
            return predictions
def predict():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(MODELS_DIR / "best")
    print(f"加载模型: {model_path}")
    model=AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    print("记载tokenizaer结束")
    predict_obj=Predict(model,device,tokenizer)
    text=["劲度kn95口罩儿童防雾霾防飞沫5只装","唯本考拉A梦CS-2德绒儿童套装","坚宝魔法防尘1600ML咖啡壶KB-790"]
    result=predict_obj.predict(text)
    print(result)
if __name__=="__main__":
    predict()