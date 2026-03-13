from pydantic import BaseModel
from typing import List, Optional

class Diamond(BaseModel):
    carat: float
    color: str
    clarity: str
    cut: str
    depth : float
    table : float
    x : float
    y : float
    z : float
    
class DiamondResponse(Diamond):
    price: float
    
    
