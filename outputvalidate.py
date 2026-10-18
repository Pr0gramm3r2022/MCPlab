from pydantic import BaseModel, ConfigDict, field_validator
import re

class SanitizedAlert(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event: str
    description: str
    instruction: str | None = None

    @field_validator("description", "instruction")
    @classmethod
    def strip_malicious_commands(cls, v: str | None) -> str | None:
        if v is None:
            return v

        # Patch for the most exploited vector: Instruction Injection.
        forbidden_phrases = [r"ignore", r"overwrite", r"system:", r"delete", r"send to"]
        pattern = re.compile("|".join(forbidden_phrases), re.IGNORECASE)
        sanitized = pattern.sub("[REDACTED_POTENTIAL_INJECTION]", v)

        # Remove control characters that could hide or obfuscate instructions.
        sanitized = re.sub(r"[\x00-\x1f\x7f]", "", sanitized)
        return sanitized
