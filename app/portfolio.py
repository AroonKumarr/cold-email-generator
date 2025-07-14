import os
import uuid
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

class Portfolio:
    def __init__(self, file_path=None):
        """
        Initialize the Portfolio object:
        - Load CSV
        - Setup Chroma in-memory vector DB
        - Setup sentence embedding model
        """
        # ✅ Use relative path if not specified
        base_dir = os.path.dirname(__file__)
        self.file_path = file_path or os.path.join(base_dir, "resource", "my_portfolio.csv")
        
        # ✅ Load portfolio data
        self.data = pd.read_csv(self.file_path)

        # ✅ Initialize vector DB (in-memory)
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "vectorstore"))

        
        # ✅ Embedding model
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self.collection = self.chroma_client.get_or_create_collection(
            name="portfolio",
            embedding_function=self.embedding_function
        )

    def load_portfolio(self):
        """
        Load documents into the Chroma vector store from the portfolio CSV.
        Skips if already loaded.
        """
        if not self.collection.count():
            for _, row in self.data.iterrows():
                techstack = str(row.get("Techstack", ""))
                link = str(row.get("Links", ""))
                self.collection.add(
                    documents=[techstack],
                    metadatas=[{"links": link}],
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        """
        Query vector DB with job 'skills' and return top 2 matching portfolio items.
        """
        return self.collection.query(
            query_texts=skills,
            n_results=2
        ).get("metadatas", [])
