from pydantic import BaseModel, ConfigDict, Field, field_validator
import math
import re

class NWSForecastSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    # NWS uses lat/long points. Limit the range to US territories only.
    latitude: float = Field(..., ge=18.0, le=72.0)
    longitude: float = Field(..., ge=-170.0, le=-65.0)
    user_agent: str = Field(
        default="(my-mcp-app, contact@me.com)",
        pattern=r"^\(.*\)$"
    )

    @field_validator("latitude", "longitude")
    @classmethod
    def validate_us_bounds(cls, v: float) -> float:
        # Real Value: This prevents the agent from wasting API credits/bandwidth
        # on coordinates outside NWS coverage (e.g., fetching weather for Paris).
        if not math.isfinite(v):
            raise ValueError("coordinates must be finite")
        return round(v, 4)
