import argparse

from preprocess.process import process
from runner.evaluate import evaluate
from runner.predict import predict
from runner.train import train

parse=argparse.ArgumentParser(description="商品标题预测")

parse.add_argument("action",
                   choices=["process","train","predict","evaluate","service"],
                   help="可选参数：process,train,predict,evaluate,service"
                   )
args=parse.parse_args()
# print(args)
if args.action=="process":
   process()
elif args.action=="train":
   train()
elif args.action=="predict":
   predict()
else:
   evaluate()
