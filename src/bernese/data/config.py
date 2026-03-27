"""Target configuration models using Pydantic V2."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TargetConfig(BaseModel):
    """Configuration model for a single target in genomic data.

    Attributes:
        name: Target name identifier.
        file: Path to the target data file.
        target_type: Type of target (e.g., "hic", "bigwig").
        clip: Optional clipping value for target values.
        parameters: Additional parameters for target processing.
        metadata: Additional metadata for the target.
    """

    name: str
    file: str
    target_type: str
    clip: Optional[float] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
