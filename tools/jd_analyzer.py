"""JD Analyzer — parses job descriptions into structured requirements.

Single responsibility: Accept raw job description text, extract required
skills, experience requirements, education requirements, and preferred
skills into a structured dict that agents can consume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools.skill_normalizer import normalize_skill_list
from utils.observability import instrument_tool_methods

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class ExtractedJD(BaseModel):
    """Structured data extracted from a job description by the LLM."""

    job_title: str = Field(description="The job title.")
    required_skills: list[str] = Field(description="Must-have skills.")
    preferred_skills: list[str] = Field(description="Nice-to-have skills.")
    min_experience_years: float = Field(description="Minimum years of experience required.")
    max_experience_years: float | None = Field(
        default=None, description="Maximum years of experience (null if no max)."
    )
    education_requirements: dict = Field(
        description="Education requirements: degree_level, field_of_study, alternatives."
    )
    summary: str = Field(description="Brief 1-2 sentence summary of the role.")


JD_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
      (
          "system",
          (
              "You are a precise job description analyzer. Extract structured "
              "requirements from the JD text below. Return ONLY information explicitly "
              "stated in the JD. Do NOT infer or fabricate requirements.\n\n"
              "IMPORTANT: required_skills and preferred_skills must be atomic skill "
              "names (for example [Python, SQL, LangChain]), never category headings "
              "or complete bullet sentences.\n\n"
              "Return your response as a JSON object matching the ExtractedJD schema."
          ),
      ),
    ("human", "{jd_text}"),
])


@instrument_tool_methods
class JDAnalyzer:
    """Analyzes job descriptions and extracts structured requirements using the LLM."""

    def __init__(self, llm: ChatOllama) -> None:
        """Initialize the JD Analyzer.

        Args:
            llm: The Ollama LLM instance for structured extraction.
        """
        self.llm = llm

    def analyze(self, jd_text: str) -> dict:
        """Analyze a job description and return structured requirements.

        Args:
            jd_text: Raw job description text.

        Returns:
            Dict with job_title, required_skills, preferred_skills,
            min_experience_years, max_experience_years,
            education_requirements, and summary.
        """
        structured_llm = self.llm.with_structured_output(ExtractedJD)
        chain = JD_ANALYSIS_PROMPT | structured_llm
        result = chain.invoke({"jd_text": jd_text})
        extracted = result.model_dump()
        extracted["required_skills"] = normalize_skill_list(extracted["required_skills"])
        extracted["preferred_skills"] = normalize_skill_list(extracted["preferred_skills"])
        return extracted

    def analyze_file(self, file_path: str) -> dict:
        """Analyze a JD file and return structured requirements.

        Args:
            file_path: Path to the JD file (currently supports .txt).

        Returns:
            Dict with raw_text and all extracted structured fields.
        """
        path = Path(file_path)
        raw_text = path.read_text(encoding="utf-8")
        extracted = self.analyze(raw_text)
        extracted["raw_text"] = raw_text
        return extracted


if __name__ == "__main__":
    import sys

    from utils.llm_factory import get_llm

    logging.basicConfig(level=logging.INFO)
    llm = get_llm()
    analyzer = JDAnalyzer(llm)

    sample_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_jd.txt"
    result = analyzer.analyze_file(sample_path)
    print("\n=== Parsed JD ===")
    for key, value in result.items():
        if key != "raw_text":
            print(f"  {key}: {value}")
