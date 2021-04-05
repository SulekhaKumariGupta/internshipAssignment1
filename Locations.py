from pydantic import BaseModel


class Location(BaseModel):
    Latitude: float
    Longitude: float
