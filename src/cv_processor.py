import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from dotenv import load_dotenv
import pickle

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class CVProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.7
        )
        self.cv_database = {}
        self.vector_store = None
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing CV data and vector store"""
        try:
            if os.path.exists("cv_database.json"):
                with open("cv_database.json", "r") as f:
                    self.cv_database = json.load(f)
            
            if os.path.exists("vector_store.pkl"):
                with open("vector_store.pkl", "rb") as f:
                    self.vector_store = pickle.load(f)
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
    def save_data(self):
        """Save CV data and vector store"""
        try:
            with open("cv_database.json", "w") as f:
                json.dump(self.cv_database, f, indent=2)
            
            if self.vector_store:
                with open("vector_store.pkl", "wb") as f:
                    pickle.dump(self.vector_store, f)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def categorize_cv(self, cv_text: str) -> Dict[str, Any]:
        """Use AI to categorize and extract information from CV"""
        prompt = f"""
        Analyze the following CV/Resume text and extract the following information:
        1. Full Name
        2. Email
        3. Phone
        4. Job Category/Field (e.g., Software Engineer, Data Scientist, Marketing, etc.)
        5. Experience Level (Junior, Mid-level, Senior, Executive)
        6. Skills (top 5-10 skills)
        7. Education Level (High School, Bachelor's, Master's, PhD, etc.)
        8. Years of Experience (estimate if not explicitly mentioned)
        
        CV Text:
        {cv_text[:2000]}...
        
        Please return the information in the following JSON format:
        {{
            "name": "Full Name",
            "email": "email@example.com",
            "phone": "phone number",
            "category": "Job Category",
            "experience_level": "Experience Level",
            "skills": ["skill1", "skill2", "skill3"],
            "education": "Education Level",
            "years_experience": "number"
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Try to extract JSON from the response
            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return {
                    "name": "Unknown",
                    "email": "Unknown",
                    "phone": "Unknown",
                    "category": "General",
                    "experience_level": "Unknown",
                    "skills": [],
                    "education": "Unknown",
                    "years_experience": "0"
                }
        except Exception as e:
            print(f"Error categorizing CV: {e}")
            return {
                "name": "Unknown",
                "email": "Unknown",
                "phone": "Unknown",
                "category": "General",
                "experience_level": "Unknown",
                "skills": [],
                "education": "Unknown",
                "years_experience": "0"
            }
    
    def add_cv(self, pdf_file, filename: str) -> str:
        """Add a new CV to the database"""
        # Generate unique ID
        cv_id = str(uuid.uuid4())
        
        # Extract text
        cv_text = self.extract_text_from_pdf(pdf_file)
        
        if not cv_text.strip():
            raise ValueError("Could not extract text from PDF")
        
        # Categorize CV using AI
        cv_info = self.categorize_cv(cv_text)
        
        # Create text chunks for vector storage
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(cv_text)
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "cv_id": cv_id,
                    "filename": filename,
                    "chunk_id": i,
                    "category": cv_info["category"],
                    "experience_level": cv_info["experience_level"],
                    "name": cv_info["name"]
                }
            )
            documents.append(doc)
        
        # Add to vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            new_vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.merge_from(new_vector_store)
        
        # Store CV information
        self.cv_database[cv_id] = {
            "id": cv_id,
            "filename": filename,
            "upload_date": datetime.now().isoformat(),
            "text": cv_text,
            "info": cv_info
        }
        
        # Save data
        self.save_data()
        
        return cv_id
    
    def search_cvs_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Search CVs by category"""
        matching_cvs = []
        for cv_id, cv_data in self.cv_database.items():
            if cv_data["info"]["category"].lower() == category.lower():
                matching_cvs.append(cv_data)
        
        return matching_cvs[:limit]
    
    def search_cvs_by_skills(self, required_skills: List[str], limit: int = 10) -> List[Dict]:
        """Search CVs by required skills"""
        matching_cvs = []
        for cv_id, cv_data in self.cv_database.items():
            cv_skills = [skill.lower() for skill in cv_data["info"]["skills"]]
            required_skills_lower = [skill.lower() for skill in required_skills]
            
            # Check if any required skill matches
            if any(req_skill in cv_skills for req_skill in required_skills_lower):
                cv_data["skill_match_score"] = sum(1 for req_skill in required_skills_lower if req_skill in cv_skills)
                matching_cvs.append(cv_data)
        
        # Sort by skill match score
        matching_cvs.sort(key=lambda x: x.get("skill_match_score", 0), reverse=True)
        return matching_cvs[:limit]
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories from stored CVs"""
        categories = set()
        for cv_data in self.cv_database.values():
            categories.add(cv_data["info"]["category"])
        return sorted(list(categories))
    
    def get_cv_by_id(self, cv_id: str) -> Dict:
        """Get CV by ID"""
        return self.cv_database.get(cv_id)
    
    def semantic_search(self, query: str, category_filter: str = None, limit: int = 5) -> List[Dict]:
        """Perform semantic search on CVs"""
        if not self.vector_store:
            return []
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(query, k=limit*3)  # Get more to filter
        
        # Filter by category if specified
        if category_filter:
            docs = [doc for doc in docs if doc.metadata.get("category", "").lower() == category_filter.lower()]
        
        # Group by CV ID and get unique CVs
        cv_ids = []
        unique_docs = []
        for doc in docs:
            cv_id = doc.metadata.get("cv_id")
            if cv_id not in cv_ids:
                cv_ids.append(cv_id)
                unique_docs.append(doc)
                if len(cv_ids) >= limit:
                    break
        
        # Get full CV data
        results = []
        for doc in unique_docs:
            cv_id = doc.metadata.get("cv_id")
            if cv_id in self.cv_database:
                cv_data = self.cv_database[cv_id].copy()
                cv_data["relevance_score"] = doc.metadata.get("score", 0)
                cv_data["matched_content"] = doc.page_content[:200] + "..."
                results.append(cv_data)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        if not self.cv_database:
            return {"total_cvs": 0, "categories": {}, "experience_levels": {}}
        
        categories = {}
        experience_levels = {}
        
        for cv_data in self.cv_database.values():
            # Count categories
            category = cv_data["info"]["category"]
            categories[category] = categories.get(category, 0) + 1
            
            # Count experience levels
            exp_level = cv_data["info"]["experience_level"]
            experience_levels[exp_level] = experience_levels.get(exp_level, 0) + 1
        
        return {
            "total_cvs": len(self.cv_database),
            "categories": categories,
            "experience_levels": experience_levels
        }
