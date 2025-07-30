from pydantic import BaseModel, conlist, field_validator,Field
from typing import List


class IrisFeatures(BaseModel):
    data: List[List[float]] = Field(..., min_items=1)

    @field_validator('data')
    def check_inner_list_length(cls, v):
        if not all(len(inner) == 4 for inner in v):
            raise ValueError('Each inner list must have exactly 4 floats')
        return v