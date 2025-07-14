import os
import re
import json
import uuid
import pandas as pd
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# === Load environment variables ===
load_dotenv()  # Loads from .env in the same directory
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Please check your .env file.")

# === Add USER_AGENT to avoid warning ===
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# === Fallback JSON extractor ===
def extract_json_from_response(response_text):
    match = re.search(r"\[\s*{.*?}\s*]\s*", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise OutputParserException(f"❌ Failed to decode JSON: {e}")
    raise OutputParserException("❌ No JSON array found in the response.")

# === Initialize LLM ===
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# === Load Job Description from Web ===
loader = WebBaseLoader("https://jobs.nike.com/job/R-33460")  # Replace with your target job URL
page_data = loader.load().pop().page_content

# === Extract Job Info from LLM ===
prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}

    ### INSTRUCTION:
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing
    the following keys: 'role', 'experience', 'skills', and 'description'.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
    """
)
chain_extract = prompt_extract | llm
res = chain_extract.invoke({'page_data': page_data})

# === Debug raw response ===
print("🔍 RAW LLM Response Start")
print(res.content)
print("🔍 RAW LLM Response End")

# === Try parsing response into JSON ===
try:
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)
except OutputParserException:
    print("⚠️ OutputParserException: Falling back to regex-based JSON extraction.")
    json_res = extract_json_from_response(res.content)

# === Load Portfolio CSV ===
df = pd.read_csv("resource/my_portfolio.csv")

# === Initialize Vector Store ===
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient("app/vectorstore")
collection = client.get_or_create_collection(
    name="portfolio",
    embedding_function=embedding_function
)

# === Add Portfolio to Vector DB (if empty) ===
if not collection.count():
    for _, row in df.iterrows():
        techstack = str(row["Techstack"])
        link = str(row["Links"])
        collection.add(
            documents=[techstack],
            metadatas=[{"links": link}],
            ids=[str(uuid.uuid4())]
        )

# === Query Matching Portfolio Links ===
job = json_res[0]
links = collection.query(query_texts=job["skills"], n_results=2).get("metadatas", [])
relevant_links = [meta["links"] for meta in links if "links" in meta]

# === Generate Cold Email ===
# === Generate Cold Email ===
prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}

    ### INSTRUCTION:
    You are Aroon, a business development executive at AtliQ. AtliQ is an AI & Software company enabling
    the seamless integration of business processes through automated tools.
    Over our experience, we have empowered numerous enterprises with tailored solutions for
    process optimization, cost reduction, and heightened overall efficiency.
    Your job is to write a cold email to the client regarding the job mentioned above and assist them
    in fulfilling their needs.
    Also add the most relevant ones from the following links to showcase AtliQ's portfolio.
    Remember you are Aroon, BDE at AtliQ.
    Do not provide a preamble.
    ### RELEVANT PORTFOLIO LINKS:
    {link_list}

    ### EMAIL (NO PREAMBLE):
    """
)
chain_email = prompt_email | llm


res = chain_email.invoke({
    "job_description": str(job),
    "link_list": "\n".join(relevant_links)
})

# === Output the Result ===
print("---- COLD EMAIL ----\n")
print(res.content)
