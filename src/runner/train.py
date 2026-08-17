from dataclasses import dataclass

import torch
import time
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding
from configuration.config import *
from preprocess.dataset import get_dataloader, get_dataset


# class TrainingConfig:
#     def __init__(self,epochs,batch_size,learning_rate,output_dir):
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.output_dir = output_dir
#可以用classdata装饰器简化上面操作
@dataclass
class TrainingConfig:
    epochs:int=10
    batch_size:int=16
    learning_rate:float=5e-5
    output_dir:str="./models"
    save_steps:int=30
    logs_dir:str="./logs"
    stop_metric:str="loss"
#也就是一步可以抵最上面的那一个的两步动作

class trainer:
    def __init__(self,compute_metrics,device,model,train_dataset,valid_dataset,training_config,collate_fn):
        self.training_config = training_config
        self.device=device
        self.model=model.to(device)
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.compute_metrics=compute_metrics
        self.optimizer=torch.optim.Adam(model.parameters(),lr=training_config.learning_rate)
        self.collate_fn=collate_fn
        self.step=1
        self.writer=SummaryWriter(log_dir=str(Path(self.training_config.logs_dir)/time.strftime("%Y-%m-%d-%H_%M_%S")))
        self.min_loss=float("inf")
        self.best_stop_score=-float("inf")
        self.stop_counter=0
        self.stop_counter_max=3
    def _get_dataloader(self,dataset):
        dataset.set_format(type="torch")
        return DataLoader(dataset=dataset,
                          batch_size=self.training_config.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_fn)
    def train(self):
        dataloader=self._get_dataloader(dataset=self.train_dataset)

        for epoch in range(1,self.training_config.epochs+1):
            for batch in tqdm(dataloader,desc=f"epoch:{epoch}",position=0):
                loss=self.train_one_step(batch)
                if self.step %self.training_config.save_steps==0:
                    tqdm.write(f"epoch:{epoch},step:{self.step},loss:{loss}")
                    self.writer.add_scalar("loss",loss,self.step)
                    results=self.evaluate()
                    metrics_str="|".join([f"{k}:{v:.4f}" for k,v in results.items()])
                    tqdm.write(f"Evaluation:{metrics_str}")

                    if self._should_stop(results):
                        tqdm.write("早停")
                        return
                    # if loss<self.min_loss:
                    #     tqdm.write("保存模型")
                    #     self.min_loss=loss
                    #     self.model.save_pretrained(self.training_config.output_dir)
                self.step+=1

        # print("training...")
    def _should_stop(self,metrics):
        score=-metrics[self.training_config.stop_metric] if self.training_config.stop_metric=="loss" else metrics[self.training_config.stop_metric]
        if score >self.best_stop_score:
            tqdm.write("保存模型")
            self.best_stop_score=score
            self.stop_counter=0
            self.model.save_pretrained(self.training_config.output_dir)
            return False
        else:
            self.stop_counter+=1
            if self.stop_counter>=self.stop_counter_max:
                return True
            else:
                return False
    def train_one_step(self,inputs):
        self.model.train()
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        outputs=self.model(**inputs)
        loss=outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    def evaluate(self):
        dataloader=self._get_dataloader(dataset=self.valid_dataset)
        self.model.eval()
        total_loss=0
        all_prediction=[]
        all_labels=[]
        for batch in dataloader:
            inputs={k:v.to(self.device) for k,v in batch.items()}
            outputs=self.model(**inputs)
            loss=outputs.loss
            total_loss+=loss.item()
            logits=outputs.logits #n,l
            predictions=torch.argmax(logits,dim=-1)
            all_prediction.extend(predictions.tolist())
            labels=inputs["labels"].tolist()
            all_labels.extend(labels)
        loss=total_loss/len(dataloader)
        metrics=self.compute_metrics(all_prediction,all_labels)
        return {"loss":loss,**metrics}


if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(MODELS_DIR/"labels.txt","r",encoding="utf-8") as f:
        labels=[line.strip() for line in f.readlines()]
    print(labels)
    lable2id={label:index for index,label in enumerate(labels)}
    id2label={index:label for index,label in enumerate(labels)}
    model=AutoModelForSequenceClassification.from_pretrained("bert-base-chinese",
                                                             num_labels=len(labels),
                                                             id2label=id2label,
                                                             label2id=lable2id)
    # print(model)str(MODELS_DIR)PRE_TRAINED_DIR/"bert-base-chinese"
    # model.save_pretrained(MODELS_DIR)
    def compute_metrics(all_predictions, all_labels)->dict:
        accuracy=accuracy_score(all_labels,all_predictions)
        f1=f1_score(all_labels,all_predictions,average="weighted")
        return {"accuracy":accuracy,"f1":f1}

    tokenzier=AutoTokenizer.from_pretrained("bert-base-chinese")
    # train_dataset=get_dataset("train").select(range(2000))
    # valid_dataset=get_dataset("valid").select(range(300))
    train_dataset=get_dataset("train")
    valid_dataset=get_dataset("valid")
    training_config=TrainingConfig(output_dir=MODELS_DIR,logs_dir=LOG_DIR)
    collate_fn=DataCollatorWithPadding(tokenizer=tokenzier,padding=True,return_tensors="pt")


    trainer=trainer(compute_metrics,device,model,train_dataset,valid_dataset,training_config,collate_fn)
    trainer.train()

