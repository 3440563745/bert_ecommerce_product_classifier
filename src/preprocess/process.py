from token import tok_name

from datasets import load_dataset, ClassLabel
from configuration.config import *
from transformers import AutoTokenizer
def process():
    # 加载数据集
    dataset_dict = load_dataset(path="csv", data_files={
        "train": str(RAW_DATA_DIR / "train.txt"),
        "test": str(RAW_DATA_DIR / "test.txt"),
        "valid": str(RAW_DATA_DIR / "valid.txt"),
    }, delimiter="\t")
    # print(dataset_dict)
    # print(dataset_dict["train"][0])
    # 数据清洗
    dataset_dict=dataset_dict.filter(lambda x: x["label"] is not None and x["text_a"] is not None)
    # print(dataset_dict["train"]["label"])
    # 保存labels
    labels = sorted(set(dataset_dict["train"]["label"]))
    labels_dict = {label: i for i, label in enumerate(labels)}

    def trans(label):
        label["label"] = labels_dict[label["label"]]
        return label

    dataset_dict = dataset_dict.map(trans, batch_size=True)
    # print(dataset_dict["train"][:4])
    dataset_dict = dataset_dict.cast_column('label', ClassLabel(names=labels))
    # print(labels)
    # print(dataset_dict["train"][:4])
    with open(MODELS_DIR / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(labels))
    # tokenizer为id列表 input_ids,attention_mask,labels
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_DIR / "bert-base-chinese")

    def encode(batch):
        outputs = tokenizer(batch["text_a"], truncation=True, return_token_type_ids=False)
        outputs["labels"] = batch["label"]
        return outputs

    dataset_dict = dataset_dict.map(encode, batched=True, remove_columns=["text_a", "label"])
    # dataset_dict.set_format("pt")
    # print(dataset_dict)
    # print(dataset_dict["train"][0:3])
    # 保存数据到文件
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)
if __name__ == "__main__":
    process()


