from pydantic import BaseModel
from typing import List

class RiskPredictionInput(BaseModel):
    Latitude_Degrees: float
    Longitude_Degrees: float
    Distance_to_Shore: float
    Depth_m: float
    Temperature_Mean: float
    SSTA: float
    TSA: float
    Turbidity: float
    Date_Year: int
    Date_Month: int
    Date_Day: int
    Ocean_Name: str
    Exposure: str
    Realm_Name: str
    Country_Name: str

class RiskPredictionOutput(BaseModel):
    predicted_category: str
    confidence: float

class ChatInput(BaseModel):
    question: str

class ChatOutput(BaseModel):
    answer: str
    sources: List[str]

class ImagePredictionOutput(BaseModel):
    prediction: str
    confidence: float