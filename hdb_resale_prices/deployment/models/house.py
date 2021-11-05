from pydantic import BaseModel

class House(BaseModel):
    postal_code : str
    floor: float
    floor_area_sqm: float
    remaining_lease: float
    flat_type: str
    flat_model: str
    town: str