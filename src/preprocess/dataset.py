from torch.utils.data import DataLoader
from datasets import load_from_disk
from configuration.config import *
from transformers import DataCollatorWithPadding,AutoTokenizer
def get_dataset(dtype="train"):
    path=str(PROCESSED_DATA_DIR/dtype)
    dataset=load_from_disk(path)
    return dataset
def get_dataloader(tokenizer,dtype="train"):
    path=str(PROCESSED_DATA_DIR/dtype)
    dataset=load_from_disk(dataset_path=path)
    # print(dataset)
    # print(dataset[1])
    dataset.set_format("torch")
    # print(dataset[:3])
    #弄为dataloader
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer,padding=True,return_tensors="pt")
    return DataLoader(dataset,batch_size=16,shuffle=True,collate_fn=collate_fn)
if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_DIR / "bert-base-chinese")
    train_data_loader=get_dataloader(tokenizer,"train")
    for batch in train_data_loader:
        for k,v in batch.items():
            print(k,v)
        break


