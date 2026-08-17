import torch
from sklearn.metrics import accuracy_score,f1_score
from configuration.config import *

from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding

from preprocess.dataset import get_dataset
from .train import trainer

def evaluate():
    # def __init__(self, compute_metrics, device, model,
    #              train_dataset, valid_dataset, training_config,
    #              collate_fn):
    def compute_metrics(all_predictions, all_labels) -> dict:
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted")
        return {"accuracy": accuracy, "f1": f1}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(MODELS_DIR / "best")
    model=AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    dataset=get_dataset("test")
    collate_fn=DataCollatorWithPadding(tokenizer,padding=True,return_tensors="pt")
    # ✅ 正确：按照定义顺序传递所有参数
    train = trainer(
        compute_metrics,  # 位置参数
        device,  # 位置参数
        model,  # 位置参数
        # train_dataset,  # 位置参数 - train_dataset
        dataset,  # 位置参数 - valid_dataset
        # training_config,  # 位置参数 - training_config
        collate_fn  # 位置参数 - collate_fn
    )
    metrics=train.evaluate()
    print(metrics)
if __name__=="__main__":
    evaluate()
