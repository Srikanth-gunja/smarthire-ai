from agents.candidate_matching_agent import CandidateMatchingAgent
from tools.skill_normalizer import normalize_skill_list


def test_normalizes_grouped_jd_skill_categories():
    skills = normalize_skill_list([
        "Core Languages: Programming: Python (Advanced), SQL, JavaScript, Java.",
        "Generative AI: LangChain, LangGraph, Hugging Face, FAISS.",
    ])
    assert {"Python", "SQL", "JavaScript", "Java", "LangChain", "LangGraph", "FAISS"} <= set(skills)


def test_grouped_jd_skills_match_atomic_resume_skills():
    agent = CandidateMatchingAgent(llm=None)
    matched, missing = agent._compute_skills_match(
        ["Python", "SQL", "LangChain", "LangGraph", "Hugging Face", "FAISS"],
        ["Programming: Python (Advanced), SQL", "Generative AI: LangChain, LangGraph, Hugging Face, FAISS"],
    )
    assert len(matched) == 6
    assert missing == []
