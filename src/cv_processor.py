import os
import uuid
import json
import re
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
        You are an expert HR analyst. Analyze the following CV/Resume text and extract key information.
        
        CV Text:
        {cv_text[:3000]}
        
        Extract the following information and respond ONLY with a valid JSON object:
        
        {{
            "name": "Full Name (extract from CV)",
            "email": "email@example.com (extract actual email or 'Not Found')",
            "phone": "phone number (extract actual phone or 'Not Found')",
            "category": "Job Category (e.g., Software Engineer, Data Scientist, Marketing Manager, Sales Representative, Project Manager, etc.)",
            "experience_level": "Junior or Mid-level or Senior or Executive",
            "skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
            "education": "High School or Bachelor's or Master's or PhD or Other",
            "years_experience": "estimate number as string"
        }}
        
        Important: Return ONLY the JSON object, no other text.
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Print response for debugging
            print(f"AI Response: {content[:500]}...")
            
            # Try to extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                cv_info = json.loads(json_str)
                
                # Validate and clean the extracted information
                return {
                    "name": cv_info.get("name", "Unknown"),
                    "email": cv_info.get("email", "Not Found"),
                    "phone": cv_info.get("phone", "Not Found"),
                    "category": cv_info.get("category", "General"),
                    "experience_level": cv_info.get("experience_level", "Unknown"),
                    "skills": cv_info.get("skills", [])[:10],  # Limit to 10 skills
                    "education": cv_info.get("education", "Unknown"),
                    "years_experience": str(cv_info.get("years_experience", "0"))
                }
            else:
                print("No valid JSON found in response")
                return self._fallback_extraction(cv_text)
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return self._fallback_extraction(cv_text)
        except Exception as e:
            print(f"Error categorizing CV: {e}")
            return self._fallback_extraction(cv_text)
    
    def _fallback_extraction(self, cv_text: str) -> Dict[str, Any]:
        """Fallback method for basic information extraction using simple text processing"""
        
        # Basic regex patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?\d{1,4}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        
        # Extract basic information
        name = "Unknown"
        email = "Not Found"
        phone = "Not Found"
        
        # Try to find email
        email_match = re.search(email_pattern, cv_text)
        if email_match:
            email = email_match.group()
        
        # Try to find phone
        phone_match = re.search(phone_pattern, cv_text)
        if phone_match:
            phone = phone_match.group()
        
        # Try to extract name (usually at the beginning)
        lines = cv_text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 2 and len(line) < 50 and not '@' in line and not any(char.isdigit() for char in line):
                if not line.lower().startswith(('resume', 'cv', 'curriculum')):
                    name = line
                    break
        
        # Simple category detection
        category = "General"
        cv_lower = cv_text.lower()
        if any(word in cv_lower for word in ['python', 'java', 'javascript', 'programming', 'developer', 'software']):
            category = "Software Engineer"
        elif any(word in cv_lower for word in ['data scientist', 'machine learning', 'data analysis', 'analytics']):
            category = "Data Scientist"
        elif any(word in cv_lower for word in ['marketing', 'digital marketing', 'social media']):
            category = "Marketing"
        elif any(word in cv_lower for word in ['sales', 'business development', 'account manager']):
            category = "Sales"
        elif any(word in cv_lower for word in ['project manager', 'scrum master', 'agile']):
            category = "Project Manager"
        
        # Simple skills extraction
        skills = []
        skill_keywords = ['python', 'java', 'javascript', 'react', 'node.js', 'sql', 'machine learning', 
                         'data analysis', 'excel', 'powerpoint', 'project management', 'agile', 'scrum']
        
        for skill in skill_keywords:
            if skill.lower() in cv_lower:
                skills.append(skill.title())
        
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "category": category,
            "experience_level": "Unknown",
            "skills": skills[:5],
            "education": "Unknown",
            "years_experience": "0"
        }
    
    def _create_enhanced_chunks(self, cv_text: str, cv_id: str, filename: str, cv_info: dict) -> list:
        """
        Create enhanced text chunks with better section identification
        """
        chunks = []
        
        # Define section patterns
        section_patterns = {
            'experience': [
                r'(?i)(work\s+experience|professional\s+experience|employment\s+history|career\s+history)',
                r'(?i)(projects?|project\s+experience)',
                r'(?i)(internship|internships)'
            ],
            'education': [
                r'(?i)(education|academic\s+background|qualifications)',
                r'(?i)(university|college|school|degree)'
            ],
            'skills': [
                r'(?i)(skills?|technical\s+skills?|core\s+competencies)',
                r'(?i)(programming|technologies|tools)'
            ],
            'contact': [
                r'(?i)(contact|personal\s+details?|profile)',
                r'(?i)(email|phone|address)'
            ]
        }
        
        # Split text into sections based on common CV patterns
        lines = cv_text.split('\n')
        current_section = 'general'
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line indicates a new section
            detected_section = None
            for section_name, patterns in section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line):
                        detected_section = section_name
                        break
                if detected_section:
                    break
            
            # If we detected a new section, save the current content
            if detected_section and detected_section != current_section and current_content:
                chunk_content = '\n'.join(current_content)
                if len(chunk_content.strip()) > 50:  # Only include substantial content
                    chunks.append({
                        "content": chunk_content,
                        "metadata": {
                            "cv_id": cv_id,
                            "filename": filename,
                            "section": current_section,
                            "category": cv_info["category"],
                            "experience_level": cv_info["experience_level"],
                            "name": cv_info["name"],
                            "chunk_type": "section"
                        }
                    })
                current_content = []
                current_section = detected_section if detected_section else current_section
            
            current_content.append(line)
        
        # Add the last section
        if current_content:
            chunk_content = '\n'.join(current_content)
            if len(chunk_content.strip()) > 50:
                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "cv_id": cv_id,
                        "filename": filename,
                        "section": current_section,
                        "category": cv_info["category"],
                        "experience_level": cv_info["experience_level"],
                        "name": cv_info["name"],
                        "chunk_type": "section"
                    }
                })
        
        # Also create overlapping chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=150
        )
        overlap_chunks = text_splitter.split_text(cv_text)
        
        for i, chunk in enumerate(overlap_chunks):
            chunks.append({
                "content": chunk,
                "metadata": {
                    "cv_id": cv_id,
                    "filename": filename,
                    "section": "overlap",
                    "chunk_id": i,
                    "category": cv_info["category"],
                    "experience_level": cv_info["experience_level"],
                    "name": cv_info["name"],
                    "chunk_type": "overlap"
                }
            })
        
        return chunks

    def add_cv(self, pdf_file, filename: str) -> str:
        """Add a new CV to the database"""
        # Generate unique ID
        cv_id = str(uuid.uuid4())
        
        # Create CV files directory if it doesn't exist
        cv_files_dir = "cv_files"
        if not os.path.exists(cv_files_dir):
            os.makedirs(cv_files_dir)
        
        # Save the original PDF file
        pdf_filename = f"{cv_id}.pdf"
        pdf_path = os.path.join(cv_files_dir, pdf_filename)
        
        # Save PDF file to disk
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.read())
            # Reset file pointer again for text extraction
            pdf_file.seek(0)
        except Exception as e:
            print(f"Error saving PDF file: {e}")
        
        # Extract text
        cv_text = self.extract_text_from_pdf(pdf_file)
        
        if not cv_text.strip():
            raise ValueError("Could not extract text from PDF")
        
        # Categorize CV using AI
        cv_info = self.categorize_cv(cv_text)
        
        # Create enhanced text chunks with section identification
        enhanced_chunks = self._create_enhanced_chunks(cv_text, cv_id, filename, cv_info)
        
        # Create documents with metadata
        documents = []
        for chunk_data in enhanced_chunks:
            doc = Document(
                page_content=chunk_data["content"],
                metadata=chunk_data["metadata"]
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
            "pdf_path": pdf_path,
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
    
    def get_cv_pdf_path(self, cv_id: str) -> str:
        """Get PDF file path for a CV"""
        cv_data = self.cv_database.get(cv_id)
        if cv_data and "pdf_path" in cv_data:
            return cv_data["pdf_path"]
        return None
    
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
    
    def create_cv_chat_chain(self, cv_id: str):
        """Create a conversational chain for a specific CV"""
        cv_data = self.get_cv_by_id(cv_id)
        if not cv_data:
            return None
        
        # Get CV-specific documents from vector store
        cv_documents = []
        if self.vector_store:
            # Search for documents belonging to this CV
            all_docs = self.vector_store.similarity_search("", k=1000)  # Get many docs
            cv_documents = [doc for doc in all_docs if doc.metadata.get("cv_id") == cv_id]
        
        if not cv_documents:
            # Create documents from CV text if not found in vector store
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(cv_data['text'])
            
            cv_documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "cv_id": cv_id,
                        "filename": cv_data['filename'],
                        "chunk_id": i,
                        "category": cv_data['info']['category'],
                        "name": cv_data['info']['name']
                    }
                )
                cv_documents.append(doc)
        
        # Create a temporary vector store for this CV only
        if cv_documents:
            cv_vector_store = FAISS.from_documents(cv_documents, self.embeddings)
            
            # Create memory for conversation
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"  # Specify which output key to store in memory
            )
            
            # Create conversational chain with improved retrieval
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=cv_vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}  # Get more documents for better context
                ),
                memory=memory,
                return_source_documents=True,
                output_key="answer"  # Explicitly set output key to avoid confusion
            )
            
            return conversation_chain
        
        return None

    def chat_with_cv(self, cv_id: str, question: str, chat_chain=None):
        """Chat with a specific CV"""
        try:
            if not chat_chain:
                chat_chain = self.create_cv_chat_chain(cv_id)
            
            if not chat_chain:
                return {
                    "answer": "I apologize, but I couldn't create a chat interface for this CV. Please try again or contact support.",
                    "source_documents": []
                }, "Error: Could not create chat interface for this CV"
            
            cv_data = self.get_cv_by_id(cv_id)
            cv_name = cv_data['info']['name'] if cv_data else "Unknown"
            
            # Enhanced prompt with CV context and specific instructions
            enhanced_question = f"""
You are analyzing the CV/Resume of {cv_name}. Please answer the following question accurately and specifically.

IMPORTANT INSTRUCTIONS:
1. Only use information that is explicitly mentioned in the CV
2. When discussing projects, work experience, or education, be very careful to:
   - Only mention details that belong to the specific item being asked about
   - Do not mix information from different projects, jobs, or educational experiences
   - If multiple projects/jobs exist, clearly distinguish between them
   - If asked about a specific project, only provide details about that project
3. If information is not available in the CV, clearly state "This information is not found in the CV"

Question: {question}

Please provide a specific and accurate answer based only on the CV content."""
            
            response = chat_chain({"question": enhanced_question})
            
            # Ensure proper response format
            if isinstance(response, dict):
                answer = response.get("answer", "No response generated")
                source_docs = response.get("source_documents", [])
                
                return {
                    "answer": answer,
                    "source_documents": source_docs
                }, None
            else:
                return {
                    "answer": str(response),
                    "source_documents": []
                }, None
                
        except Exception as e:
            error_msg = f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."
            return {
                "answer": error_msg,
                "source_documents": []
            }, f"Error: {str(e)}"
    
    def search_cv_content(self, cv_id: str, query: str, limit: int = 3):
        """Search within a specific CV's content with improved context"""
        if not self.vector_store:
            return []
        
        # Get all documents with similarity scores
        all_docs = self.vector_store.similarity_search_with_score(query, k=50)
        
        # Filter for this specific CV
        cv_docs = [(doc, score) for doc, score in all_docs if doc.metadata.get("cv_id") == cv_id]
        
        # Prioritize section-based chunks over overlap chunks
        section_docs = [(doc, score) for doc, score in cv_docs if doc.metadata.get("chunk_type") == "section"]
        overlap_docs = [(doc, score) for doc, score in cv_docs if doc.metadata.get("chunk_type") == "overlap"]
        
        # Combine with priority to sections
        prioritized_docs = section_docs + overlap_docs
        
        # Sort by similarity score (lower is better for FAISS)
        prioritized_docs.sort(key=lambda x: x[1])
        
        # Return top results (just the documents, not scores)
        return [doc for doc, score in prioritized_docs[:limit]]
