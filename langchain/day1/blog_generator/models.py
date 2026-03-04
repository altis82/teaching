from pydantic import BaseModel, Field
from typing import List

from config import NUM_TITLES


class BlogTitles(BaseModel):
	"""Blog title generator output"""

	titles: List[str] = Field(
		description=f"List of {NUM_TITLES} creative and engaging blog titles"
	)
	summary: str = Field(
		description="Brief explanation of the title strategy"
	)
