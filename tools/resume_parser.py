"""Resume Parser — extracts text and structured data from resume files.

Single responsibility: Accept raw resume text, perform structured field
extraction using the LLM, and return a dict with candidate_name, skills,
experience_years, education, certifications, and past_roles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools.skill_normalizer import extract_skills_from_text, normalize_skill_list
from utils.observability import instrument_tool_methods

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class ExtractedResume(BaseModel):
    """Structured data extracted from a resume by the LLM."""

    candidate_name: str = Field(description="Full name of the candidate.")
    skills: list[str] = Field(description="Technical and soft skills.")
    experience_years: float = Field(description="Total years of professional experience.")
    education: list[dict] = Field(
        description="Education entries with degree, institution, and year."
    )
    certifications: list[str] = Field(
        default_factory=list, description="Professional certifications."
    )
    past_roles: list[dict] = Field(
        default_factory=list,
        description="Previous roles with title, company, and dates.",
    )


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
      (
          "system",
          (
              "You are a precise resume parser. Extract structured data from the "
              "resume text below. Return ONLY facts explicitly stated in the resume. "
              "Do NOT infer, assume, or fabricate any information. If a field is not "
              "mentioned, use an empty list or 0 as appropriate.\n\n"
              "Return your response as a JSON object matching the ExtractedResume schema."
          ),
      ),
    ("human", "{resume_text}"),
])


@instrument_tool_methods
class ResumeParser:
    """Parses resume text and extracts structured candidate data using the LLM."""

    def __init__(self, llm: ChatOllama) -> None:
        """Initialize the Resume Parser.

        Args:
            llm: The Ollama LLM instance for structured extraction.
        """
        self.llm = llm

    def parse_text(self, resume_text: str) -> dict:
        """Parse raw resume text into structured data.

        Args:
            resume_text: The raw resume content as a string.

        Returns:
            Dict with candidate_name, skills, experience_years, education,
            certifications, and past_roles.
        """
        structured_llm = self.llm.with_structured_output(ExtractedResume)
        chain = EXTRACTION_PROMPT | structured_llm
        result = chain.invoke({"resume_text": resume_text})
        extracted = result.model_dump()
        # Keep the model's structured extraction, but supplement it with
        # skills visibly present in the raw document. This prevents a compact
        # local model from silently reducing two distinct resumes to the same
        # four generic skills.
        extracted["skills"] = normalize_skill_list(
            extracted["skills"] + extract_skills_from_text(resume_text)
        )
        return extracted

    def parse_file(self, file_path: str) -> dict:
        """Parse a resume file and return structured data.

        For Phase 2, only .txt files are supported. PDF/DOCX support
        will be added in Phase 5 with Streamlit file upload wiring.

        Args:
            file_path: Path to the resume file.

        Returns:
            Dict with raw_text and all extracted structured fields.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            raw_text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            raw_text = self._extract_text_from_pdf(file_path)
        elif suffix in (".docx", ".doc"):
            raw_text = self._extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        extracted = self.parse_text(raw_text)
        extracted["raw_text"] = raw_text
        return extracted

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract raw text from a PDF resume.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text content.
        """
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract raw text from a DOCX resume.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            Extracted text content.
        """
        from docx import Document

        document = Document(file_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)


if __name__ == "__main__":
    import sys

    from utils.llm_factory import get_llm

    logging.basicConfig(level=logging.INFO)
    llm = get_llm()
    parser = ResumeParser(llm)

    sample_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resumes/sarah_chen.txt"
    result = parser.parse_file(sample_path)
    print("\n=== Parsed Resume ===")
    for key, value in result.items():
        if key != "raw_text":
            print(f"  {key}: {value}")
