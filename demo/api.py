from fastapi import FastAPI
import pandas as pd

from pydantic import BaseModel
from typing import Optional

from diamonds.data import preprocess_data, load_model

import asyncio
import time

class Diamands(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    price: Optional[float] = None 
    x: float
    y: float
    z: float

api = FastAPI()


@api.get("/")
def read_root():
    """
    Root endpoint for the API.
    """
    return {"Hello": "World"}   


@api.get("/hello-async")
async def read_hello_sync(name: str, times : int = 1):
    """
    Endpoint to greet a user by name.
    """
    await asyncio.sleep(0.1)
    return {name: times * "Hello "}


@api.get("/hello")
def read_hello(name: str, times : int = 1):
    """
    Endpoint to greet a user by name.
    """
    
    time.sleep(0.1)
    return {name: times * "Hello "}

@api.get("/diamond_price")
def read_diamond_price( carat: float
                       , cut: str
                       , color: str
                       , clarity: str
                       , depth: float
                       , table: float
                       , x: float
                       , y: float
                       , z: float) -> dict[str, float]:

    diamond_df = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })
    
    pipeline = load_model("pipeline")
    prediction = pipeline.predict(diamond_df)
    return {"predicted_price": prediction[0]}

@api.post("/diamonds")
def post_diamonds_price(diamonds: list[Diamands]) -> list[Diamands]:
    """
    Endpoint to receive diamond data.
    """
    diamond_df  = pd.DataFrame([diamond.model_dump() for diamond in diamonds])

    pipeline = load_model("pipeline")
    prediction = pipeline.predict(diamond_df)
    for diamond, price in zip(diamonds, prediction):
        diamond.price = price
    return diamonds