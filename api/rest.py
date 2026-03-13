from fastapi import FastAPI
import pandas as pd

from diamonds.model import load_model 
from data_model import DiamondResponse, Diamond

import time
import asyncio


app = FastAPI(title="Diamond API", 
              description="API for Diamond application", 
              version="0.1.0")


pipeline = load_model("pipeline", "local")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Diamond API!"}


@app.post("/predict_one"#, response_model=DiamondResponse
          )
def predict_price(diamond: Diamond):

    diamond_df = pd.DataFrame([diamond.model_dump()])
    prediction = pipeline.predict(diamond_df) # Table 
    diamond_response = DiamondResponse(**diamond.model_dump(),
                                       price=prediction[0])
    return diamond_response


@app.post("/diamonds")
def post_diamonds_price(diamonds: list[Diamond]) -> list[DiamondResponse]:
    """
    Endpoint to receive diamond data.
    """
    diamond_df  = pd.DataFrame([diamond.model_dump() for diamond in diamonds])

    pipeline = load_model("pipeline")
    prediction = pipeline.predict(diamond_df)
    diamond_responses = [DiamondResponse(**diamond.model_dump(), price=price) for diamond, price in zip(diamonds, prediction)]
    return diamond_responses

counter, counter_async = 0, 0

@app.get("/hello")
def hello(name: str):
    # Requete Api a un autre service 
    time.sleep(0.1)
    
    global counter
    counter += 1
    print(f"Request {counter}")
    return {"message": f"Hello, {name}!"}


@app.get("/hello-async")
async def hello_async(name: str):
    # Requete Api a un autre service 
    await asyncio.sleep(0.1)
    
    global counter_async
    counter_async += 1
    print(f"Request {counter_async}")
    return {"message": f"Hello, {name}!"}
