from pydantic import BaseModel

class Title(BaseModel):
    title:str|list

class Category(BaseModel):
    cate:str|list