import os
import re
import json
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq

# === Load API Key from .env ===
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Utility: Fallback regex-based JSON extractor ===
def extract_json_from_response(response_text):
    try:
        response_text = response_text.strip().strip("`")
        match = re.search(r"\[\s*{.*?}\s*]", response_text, re.DOTALL)
        if match:
            json_text = match.group()
            parsed = json.loads(json_text)
            if isinstance(parsed, list):
                return parsed
            raise OutputParserException("❌ Extracted data is not a JSON list.")
        else:
            raise OutputParserException("❌ No JSON array found in the response.")
    except json.JSONDecodeError as e:
        raise OutputParserException(f"❌ Failed to decode JSON: {e}")


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}

    ### INSTRUCTION:
    You are an AI job parser. From the provided job listings text, extract job data into the following JSON format.

    Include:
    - "role": Job title
    - "experience": Mentioned years or seniority level
    - "skills": A list of relevant technical or soft skills — either mentioned explicitly OR logically inferred from the role/description
    - "description": One-liner summary of the job

    Even if skills are not explicitly mentioned, infer them using your knowledge of similar roles. For example:
    - "Data Scientist" → ["Python", "Statistics", "Machine Learning"]
    - "Frontend Engineer" → ["JavaScript", "React", "HTML", "CSS"]

    Return **only the JSON array** — no backticks, no markdown, no explanation.

    ### OUTPUT FORMAT:

    [
        {{
            "role": "string",
            "experience": "string",
            "skills": ["string", ...],
            "description": "string"
        }},
        ...
    ]
    """
)


        truncated_text = cleaned_text[:3000]
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": truncated_text})

        print("🔍 LLM Raw Response:\n", str(res))

        try:
            parser = JsonOutputParser()
            raw_content = res.content.strip().strip("`")
            return parser.parse(raw_content)
        except OutputParserException:
            print("⚠️ OutputParserException: Falling back to regex-based JSON extraction.")
            return extract_json_from_response(res.content)

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Aroon, a Business Development Executive at AtliQ, a top-tier AI & software consulting firm.
            Craft a professional cold email for the client, referencing the job role above and incorporating relevant links below
            to showcase AtliQ’s work.

            Most importantly, do not include any preamble or greeting lines. Just return the email content.

            ### RELEVANT PORTFOLIO LINKS:
            {link_list}

            ### COLD EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            "link_list": str(links)
        })

        return res.content.strip().strip("`")  # Just return the cleaned content, no need to parse here
