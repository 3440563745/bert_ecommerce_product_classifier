from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello_world():
    print("hello world")
    return {"message": "hello world"}  # ✅ 添加返回值

@app.get("/test")
def test():
    return {"status": "ok", "message": "测试成功"}