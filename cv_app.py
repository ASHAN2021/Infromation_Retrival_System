import streamlit as st
import pandas as pd
from src.cv_processor import CVProcessor
import plotly.express as px
import plotly.graph_objects as go

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
            
            if st.button(f"View Details", key=f"view_{cv_data['id']}"):
                st.session_state.selected_cv = cv_data['id']
        
        st.divider()

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
                            st.success(f"✅ {uploaded_file.name} processed successfully! ID: {cv_id[:8]}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
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
