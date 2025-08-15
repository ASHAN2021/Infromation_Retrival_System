import streamlit as st
import pandas as pd
from src.cv_processor import CVProcessor
import plotly.express as px
import plotly.graph_objects as go
import base64
import os

def get_pdf_download_link(cv_processor, cv_id, display_text="View CV PDF"):
    """Generate a download link for the CV PDF"""
    pdf_path = cv_processor.get_cv_pdf_path(cv_id)
    
    if pdf_path and os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            cv_data = cv_processor.get_cv_by_id(cv_id)
            filename = cv_data['filename'] if cv_data else f"cv_{cv_id[:8]}.pdf"
            
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" target="_blank">{display_text}</a>'
            return href
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            return None
    else:
        st.warning("PDF file not found")
        return None

def display_pdf_viewer(cv_processor, cv_id):
    """Display PDF in an embedded viewer"""
    pdf_path = cv_processor.get_cv_pdf_path(cv_id)
    
    if pdf_path and os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            
            # Embed PDF viewer
            pdf_display = f"""
            <iframe src="data:application/pdf;base64,{b64_pdf}" 
                    width="100%" height="800px" type="application/pdf">
                <p>Unable to display PDF file. 
                <a href="data:application/pdf;base64,{b64_pdf}" download>Download instead</a></p>
            </iframe>
            """
            
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying PDF: {e}")
    else:
        st.warning("PDF file not found")

# Initialize CV processor
@st.cache_resource
def get_cv_processor():
    return CVProcessor()

def display_cv_card(cv_data):
    """Display a CV in a card format"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"👤 {cv_data['info']['name']}")
            st.write(f"**Category:** {cv_data['info']['category']}")
            st.write(f"**Experience:** {cv_data['info']['experience_level']} ({cv_data['info']['years_experience']} years)")
            st.write(f"**Education:** {cv_data['info']['education']}")
            
            if cv_data['info']['skills']:
                skills_text = ", ".join(cv_data['info']['skills'][:5])
                st.write(f"**Skills:** {skills_text}")
            
            if 'matched_content' in cv_data:
                st.write(f"**Relevant Content:** {cv_data['matched_content']}")
        
        with col2:
            st.write(f"**ID:** `{cv_data['id'][:8]}...`")
            st.write(f"**Uploaded:** {cv_data['upload_date'][:10]}")
            if 'relevance_score' in cv_data:
                st.write(f"**Relevance:** {cv_data['relevance_score']:.2f}")
            
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                if st.button(f"View Details", key=f"view_{cv_data['id']}"):
                    st.session_state.selected_cv = cv_data['id']
            
            with col2b:
                if st.button(f"View PDF", key=f"pdf_{cv_data['id']}"):
                    st.session_state.view_pdf = cv_data['id']
            
            with col2c:
                if st.button(f"Chat", key=f"chat_{cv_data['id']}"):
                    st.session_state.chat_cv = cv_data['id']
        
        st.divider()

def show_cv_chat(cv_processor, cv_id):
    """Show chat interface for a specific CV"""
    cv_data = cv_processor.get_cv_by_id(cv_id)
    if not cv_data:
        st.error("CV not found!")
        return
    
    st.header(f"💬 Chat with {cv_data['info']['name']}'s CV")
    
    # Initialize chat chain in session state
    if f"chat_chain_{cv_id}" not in st.session_state:
        with st.spinner("Initializing chat interface..."):
            chat_chain = cv_processor.create_cv_chat_chain(cv_id)
            st.session_state[f"chat_chain_{cv_id}"] = chat_chain
    
    # Initialize chat history
    if f"chat_history_{cv_id}" not in st.session_state:
        st.session_state[f"chat_history_{cv_id}"] = []
    
    # CV Information Sidebar
    with st.sidebar:
        st.subheader("CV Information")
        info = cv_data['info']
        st.write(f"**Name:** {info['name']}")
        st.write(f"**Category:** {info['category']}")
        st.write(f"**Experience:** {info['experience_level']}")
        st.write(f"**Years:** {info['years_experience']}")
        st.write(f"**Education:** {info['education']}")
        
        if info['skills']:
            st.write("**Skills:**")
            for skill in info['skills'][:5]:
                st.write(f"• {skill}")
        
        if st.button("🔍 Search CV Content"):
            st.session_state.show_cv_search = True
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            history = st.session_state[f"chat_history_{cv_id}"]
            
            if not history:
                st.info("💡 Ask questions about this CV! Examples:\n- What are the candidate's key skills?\n- What experience does this person have?\n- What projects has this candidate worked on?\n- Is this candidate suitable for a Python developer role?")
            
            for i, (question, answer, sources) in enumerate(history):
                with st.expander(f"Q{i+1}: {question[:50]}...", expanded=True):
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")
                    st.divider()
        
        # Question input
        st.subheader("Ask a Question")
        
        # Suggested questions
        suggested_questions = [
            "What are the key skills mentioned in this CV?",
            "What is the candidate's work experience?",
            "What education background does this person have?",
            "What projects has this candidate worked on?",
            "Is this candidate suitable for a senior developer role?",
            "What programming languages does this person know?",
            "What are the candidate's achievements?",
        ]
        
        # Suggested questions as clickable buttons
        st.write("**Quick questions:**")
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"quick_q_{cv_id}_{i}", use_container_width=True):
                    # Process the question immediately
                    with st.spinner("Getting answer..."):
                        chat_chain = st.session_state[f"chat_chain_{cv_id}"]
                        response, error = cv_processor.chat_with_cv(cv_id, question, chat_chain)
                        
                        if response and not error:
                            answer = response.get('answer', 'No answer provided')
                            sources = response.get('source_documents', [])
                            
                            # Add to history
                            st.session_state[f"chat_history_{cv_id}"].append((question, answer, sources))
                            st.rerun()
                        else:
                            st.error(f"Error: {error or 'Unknown error occurred'}")
        
        # Question input using form
        with st.form(key=f"chat_form_{cv_id}", clear_on_submit=True):
            question = st.text_input(
                "Your question:",
                placeholder="Ask anything about this CV...",
            )
            
            col_ask, col_clear = st.columns([1, 1])
            
            with col_ask:
                ask_submitted = st.form_submit_button("Ask Question")
            
            with col_clear:
                clear_submitted = st.form_submit_button("Clear Chat")
        
        # Handle form submissions
        if ask_submitted:
            if question.strip():  # Check if question is not empty
                with st.spinner("Getting answer..."):
                    chat_chain = st.session_state[f"chat_chain_{cv_id}"]
                    response, error = cv_processor.chat_with_cv(cv_id, question, chat_chain)
                    
                    if response and not error:
                        answer = response.get('answer', 'No answer provided')
                        sources = response.get('source_documents', [])
                        
                        # Add to history
                        st.session_state[f"chat_history_{cv_id}"].append((question, answer, sources))
                        
                        st.rerun()
                    else:
                        st.error(f"Error: {error or 'Unknown error occurred'}")
            else:
                st.warning("Please enter a question before submitting.")
        
        if clear_submitted:
            st.session_state[f"chat_history_{cv_id}"] = []
            st.rerun()
    
    with col2:
        if st.button("← Back"):
            if f'chat_cv' in st.session_state:
                del st.session_state.chat_cv
            st.rerun()
        
        # Search within CV content
        if st.session_state.get('show_cv_search', False):
            st.subheader("🔍 Search CV")
            search_query = st.text_input("Search in CV:", key=f"cv_search_{cv_id}")
            
            if search_query:
                search_results = cv_processor.search_cv_content(cv_id, search_query)
                
                if search_results:
                    st.write(f"Found {len(search_results)} relevant sections:")
                    for i, doc in enumerate(search_results):
                        with st.expander(f"Result {i+1}"):
                            st.write(doc.page_content)
                else:
                    st.write("No relevant content found.")
            
            if st.button("Close Search"):
                st.session_state.show_cv_search = False
                st.rerun()

def show_cv_details(cv_processor, cv_id):
    """Show detailed view of a CV"""
    cv_data = cv_processor.get_cv_by_id(cv_id)
    if not cv_data:
        st.error("CV not found!")
        return
    
    st.header(f"CV Details: {cv_data['info']['name']}")
    
    # Basic Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        info = cv_data['info']
        st.write(f"**Name:** {info['name']}")
        st.write(f"**Email:** {info['email']}")
        st.write(f"**Phone:** {info['phone']}")
        st.write(f"**Category:** {info['category']}")
        st.write(f"**Experience Level:** {info['experience_level']}")
        st.write(f"**Years of Experience:** {info['years_experience']}")
        st.write(f"**Education:** {info['education']}")
    
    with col2:
        st.subheader("Skills")
        if info['skills']:
            for skill in info['skills']:
                st.write(f"• {skill}")
        else:
            st.write("No skills listed")
    
    # PDF Viewing Options
    st.subheader("CV Document")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download link
        download_link = get_pdf_download_link(cv_processor, cv_id, "📥 Download CV PDF")
        if download_link:
            st.markdown(download_link, unsafe_allow_html=True)
    
    with col2:
        if st.button("📄 View PDF in Browser"):
            st.session_state.view_pdf = cv_id
    
    # Full CV Text
    st.subheader("Full CV Content")
    with st.expander("Click to view full CV text"):
        st.text_area("CV Content", cv_data['text'], height=400, disabled=True)
    
    if st.button("← Back to Search"):
        if 'selected_cv' in st.session_state:
            del st.session_state.selected_cv

def main():
    st.set_page_config(
        page_title="CV Management System",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize CV processor
    cv_processor = get_cv_processor()
    
    # Sidebar for navigation
    st.sidebar.title("🔍 CV Management System")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload CVs", "Search CVs", "Analytics", "Browse All CVs"]
    )
    
    # Show selected CV details if any
    if 'selected_cv' in st.session_state:
        show_cv_details(cv_processor, st.session_state.selected_cv)
        return
    
    # Show CV chat if requested
    if 'chat_cv' in st.session_state:
        show_cv_chat(cv_processor, st.session_state.chat_cv)
        return
    
    # Show PDF viewer if requested
    if 'view_pdf' in st.session_state:
        cv_id = st.session_state.view_pdf
        cv_data = cv_processor.get_cv_by_id(cv_id)
        
        if cv_data:
            st.header(f"📄 CV Document: {cv_data['info']['name']}")
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button("← Back"):
                    del st.session_state.view_pdf
                    st.rerun()
                
                st.write(f"**Name:** {cv_data['info']['name']}")
                st.write(f"**Category:** {cv_data['info']['category']}")
                st.write(f"**Experience:** {cv_data['info']['experience_level']}")
                
                # Download button
                download_link = get_pdf_download_link(cv_processor, cv_id, "📥 Download PDF")
                if download_link:
                    st.markdown(download_link, unsafe_allow_html=True)
            
            with col2:
                # Display PDF
                display_pdf_viewer(cv_processor, cv_id)
        else:
            st.error("CV not found!")
            if st.button("← Back"):
                del st.session_state.view_pdf
        return
    
    if page == "Upload CVs":
        st.header("📤 Upload CV/Resume Files")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more CV/Resume PDF files"
        )
        
        if uploaded_files:
            if st.button("Process CVs"):
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            cv_id = cv_processor.add_cv(uploaded_file, uploaded_file.name)
                            success_count += 1
                            
                            # Get and display the processed CV info
                            cv_data = cv_processor.get_cv_by_id(cv_id)
                            if cv_data:
                                st.success(f"✅ {uploaded_file.name} processed successfully! ID: {cv_id[:8]}")
                                
                                # Show extracted information
                                with st.expander(f"📊 Extracted Information from {uploaded_file.name}"):
                                    info = cv_data['info']
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Name:** {info['name']}")
                                        st.write(f"**Email:** {info['email']}")
                                        st.write(f"**Phone:** {info['phone']}")
                                        st.write(f"**Category:** {info['category']}")
                                    
                                    with col2:
                                        st.write(f"**Experience Level:** {info['experience_level']}")
                                        st.write(f"**Years of Experience:** {info['years_experience']}")
                                        st.write(f"**Education:** {info['education']}")
                                        if info['skills']:
                                            st.write(f"**Skills:** {', '.join(info['skills'])}")
                                
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        # Show the error details for debugging
                        with st.expander("Error Details"):
                            st.code(str(e))
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success(f"Processed {success_count} out of {len(uploaded_files)} files successfully!")
    
    elif page == "Search CVs":
        st.header("🔎 Search CVs")
        
        # Search options
        search_type = st.selectbox(
            "Search Type",
            ["Semantic Search", "Category Filter", "Skills Filter"]
        )
        
        if search_type == "Semantic Search":
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query = st.text_input(
                    "Enter your search query:",
                    placeholder="e.g., 'Python developer with machine learning experience'"
                )
            
            with col2:
                categories = cv_processor.get_all_categories()
                category_filter = st.selectbox(
                    "Filter by Category (optional)",
                    ["All"] + categories
                )
            
            if query:
                category = None if category_filter == "All" else category_filter
                results = cv_processor.semantic_search(query, category, limit=10)
                
                if results:
                    st.write(f"Found {len(results)} matching CVs:")
                    for cv_data in results:
                        display_cv_card(cv_data)
                else:
                    st.write("No matching CVs found.")
        
        elif search_type == "Category Filter":
            categories = cv_processor.get_all_categories()
            
            if categories:
                selected_category = st.selectbox("Select Category", categories)
                
                if st.button("Search"):
                    results = cv_processor.search_cvs_by_category(selected_category)
                    
                    if results:
                        st.write(f"Found {len(results)} CVs in {selected_category}:")
                        for cv_data in results:
                            display_cv_card(cv_data)
                    else:
                        st.write(f"No CVs found in {selected_category} category.")
            else:
                st.write("No categories available. Please upload some CVs first.")
        
        elif search_type == "Skills Filter":
            skills_input = st.text_input(
                "Enter required skills (comma-separated):",
                placeholder="Python, Machine Learning, SQL"
            )
            
            if skills_input:
                skills = [skill.strip() for skill in skills_input.split(",")]
                results = cv_processor.search_cvs_by_skills(skills)
                
                if results:
                    st.write(f"Found {len(results)} CVs with matching skills:")
                    for cv_data in results:
                        display_cv_card(cv_data)
                else:
                    st.write("No CVs found with the specified skills.")
    
    elif page == "Analytics":
        st.header("📊 CV Database Analytics")
        
        stats = cv_processor.get_statistics()
        
        if stats["total_cvs"] == 0:
            st.write("No CVs in database yet. Please upload some CVs first.")
            return
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total CVs", stats["total_cvs"])
        
        with col2:
            st.metric("Categories", len(stats["categories"]))
        
        with col3:
            st.metric("Experience Levels", len(stats["experience_levels"]))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if stats["categories"]:
                st.subheader("CVs by Category")
                fig = px.pie(
                    values=list(stats["categories"].values()),
                    names=list(stats["categories"].keys()),
                    title="Distribution of CV Categories"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if stats["experience_levels"]:
                st.subheader("CVs by Experience Level")
                fig = px.bar(
                    x=list(stats["experience_levels"].keys()),
                    y=list(stats["experience_levels"].values()),
                    title="Distribution of Experience Levels"
                )
                fig.update_layout(xaxis_title="Experience Level", yaxis_title="Number of CVs")
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Browse All CVs":
        st.header("📋 Browse All CVs")
        
        stats = cv_processor.get_statistics()
        
        if stats["total_cvs"] == 0:
            st.write("No CVs in database yet. Please upload some CVs first.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = ["All"] + cv_processor.get_all_categories()
            selected_category = st.selectbox("Filter by Category", categories)
        
        with col2:
            experience_levels = ["All"] + list(stats["experience_levels"].keys())
            selected_exp = st.selectbox("Filter by Experience", experience_levels)
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Upload Date", "Name", "Category"])
        
        # Get all CVs
        all_cvs = list(cv_processor.cv_database.values())
        
        # Apply filters
        filtered_cvs = all_cvs
        
        if selected_category != "All":
            filtered_cvs = [cv for cv in filtered_cvs if cv["info"]["category"] == selected_category]
        
        if selected_exp != "All":
            filtered_cvs = [cv for cv in filtered_cvs if cv["info"]["experience_level"] == selected_exp]
        
        # Sort
        if sort_by == "Upload Date":
            filtered_cvs.sort(key=lambda x: x["upload_date"], reverse=True)
        elif sort_by == "Name":
            filtered_cvs.sort(key=lambda x: x["info"]["name"])
        elif sort_by == "Category":
            filtered_cvs.sort(key=lambda x: x["info"]["category"])
        
        st.write(f"Showing {len(filtered_cvs)} CVs:")
        
        # Display CVs
        for cv_data in filtered_cvs:
            display_cv_card(cv_data)

if __name__ == "__main__":
    main()
