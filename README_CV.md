# CV Management System with AI

An intelligent CV/Resume management system that uses generative AI to analyze, categorize, and search through CVs. Built with Streamlit, LangChain, and Groq API.

## Features

### 🔍 **Intelligent CV Processing**

- **Automatic CV Analysis**: Uses AI to extract key information from CVs including:
  - Name, Email, Phone
  - Job Category/Field
  - Experience Level (Junior, Mid-level, Senior, Executive)
  - Skills and Technologies
  - Education Level
  - Years of Experience

### 📊 **Advanced Search & Filtering**

- **Semantic Search**: Natural language queries to find relevant CVs
- **Category Filtering**: Filter CVs by job categories (Software Engineer, Data Scientist, etc.)
- **Skills-based Search**: Find CVs with specific technical skills
- **Experience Level Filtering**: Filter by experience levels

### 💾 **Smart Database Management**

- **Unique ID System**: Each CV gets a unique identifier for easy tracking
- **Vector Database**: Uses FAISS for efficient similarity search
- **Persistent Storage**: CVs and metadata are saved locally
- **Metadata Extraction**: Structured information extraction from unstructured CV text

### 📈 **Analytics Dashboard**

- **Database Statistics**: Overview of total CVs, categories, experience levels
- **Visual Charts**: Interactive charts showing distribution of CVs
- **Category Distribution**: Pie charts showing CV categories
- **Experience Level Analysis**: Bar charts showing experience distribution

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ASHAN2021/Infromation_Retrival_System.git
   cd Infromation_Retrival_System
   ```

2. **Create and activate conda environment**

   ```bash
   conda create -n genai python=3.8
   conda activate genai
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### Running the CV Management System

```bash
conda activate genai
streamlit run cv_app.py
```

### Running the Original Information Retrieval System

```bash
conda activate genai
streamlit run app.py
```

## How to Use

### 1. **Upload CVs**

- Navigate to "Upload CVs" page
- Select multiple PDF files (CV/Resume files)
- Click "Process CVs" to analyze and store them

### 2. **Search CVs**

- **Semantic Search**: Enter natural language queries like "Python developer with machine learning experience"
- **Category Filter**: Select from automatically detected job categories
- **Skills Filter**: Enter comma-separated skills to find matching CVs

### 3. **View Analytics**

- Check database statistics
- View distribution charts
- Monitor CV collection growth

### 4. **Browse All CVs**

- View all stored CVs
- Apply filters by category and experience level
- Sort by different criteria

## System Architecture

### **AI Components**

- **LLM**: Groq's Llama3-8b-8192 for CV analysis and categorization
- **Embeddings**: HuggingFace's all-MiniLM-L6-v2 for semantic search
- **Vector Store**: FAISS for efficient similarity search

### **Data Storage**

- **CV Database**: JSON file storing CV metadata and full text
- **Vector Store**: Pickle file storing vector embeddings
- **Unique IDs**: UUID4 for each CV

### **CV Analysis Pipeline**

1. **PDF Text Extraction**: Extract text from PDF files
2. **AI Analysis**: Use LLM to categorize and extract structured information
3. **Text Chunking**: Split text into chunks for vector storage
4. **Vector Embedding**: Create embeddings for semantic search
5. **Metadata Storage**: Store structured information and full text

## API Reference

### CVProcessor Class

#### Key Methods:

- `add_cv(pdf_file, filename)`: Add new CV to database
- `search_cvs_by_category(category)`: Search by job category
- `search_cvs_by_skills(skills_list)`: Search by skills
- `semantic_search(query, category_filter)`: Natural language search
- `get_statistics()`: Get database statistics
- `get_all_categories()`: Get all job categories

## File Structure

```
Infromation_Retrival_System/
├── src/
│   ├── __init__.py
│   ├── helper.py              # Original helper functions
│   └── cv_processor.py        # CV processing logic
├── cv_app.py                  # New CV management Streamlit app
├── app.py                     # Original information retrieval app
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── .gitignore                # Git ignore file
├── cv_database.json          # CV metadata storage (auto-created)
├── vector_store.pkl          # Vector embeddings storage (auto-created)
└── README.md                 # This file
```

## Environment Variables

Create a `.env` file with the following:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for original app)
COHERE_API_KEY=your_cohere_api_key_here
```

## Data Privacy & Security

- ✅ All data stored locally
- ✅ No data sent to external services except for AI analysis
- ✅ API keys stored in environment variables
- ✅ `.env` file added to `.gitignore` for security

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Make sure you're in the `genai` conda environment
2. **API Key Error**: Check your `.env` file and Groq API key
3. **PDF Processing Error**: Ensure PDF files are not corrupted or password-protected
4. **Memory Issues**: For large CV collections, consider chunking the processing

### Performance Tips:

- Process CVs in smaller batches (5-10 at a time)
- Restart the app periodically for large datasets
- Monitor memory usage when processing many CVs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Groq** for fast LLM inference
- **LangChain** for LLM orchestration
- **Streamlit** for the web interface
- **FAISS** for vector similarity search
- **HuggingFace** for embeddings model
