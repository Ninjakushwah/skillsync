import streamlit as st
import pandas as pd
import PyPDF2
import ast

st.set_page_config(page_title="SkillSync - Job Skills Gap Analyzer", page_icon="⭐", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("job_descriptions_small.csv")
    df['job_title_clean'] = df['Job Title'].str.lower().str.strip()
    
    return df

df = load_data()

def parse_skills(skill_val):
    if pd.isna(skill_val):
        return []
    try:
        return ast.literal_eval(skill_val)
    except:
        return [s.strip() for s in str(skill_val).split(',')]
    
    SKILLS_DB = [
    # Programming
    'python', 'sql', 'r', 'java', 'scala', 'javascript', 'c++',
    
    # ML/AI
    'machine learning', 'deep learning', 'nlp', 
    'natural language processing', 'computer vision',
    'neural networks', 'reinforcement learning',
    
    # Libraries
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv',
    
    # Data
    'data analysis', 'data visualization', 'data wrangling',
    'feature engineering', 'eda', 'statistics',
    
    # Tools
    'git', 'docker', 'aws', 'azure', 'gcp', 'spark',
    'hadoop', 'tableau', 'power bi', 'excel',
    
    # Databases
    'mysql', 'postgresql', 'mongodb', 'redis',
    
    # Other
    'api', 'flask', 'fastapi', 'streamlit', 'linux'
]
    
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def analyze_gap(resume_skills, job_title):
    job_data = df[df['job_title_clean'] == job_title.lower()]
    
    if job_data.empty:
        return None
    
    all_job_text = " ".join([str(s) for s in job_data['skills_list']]).lower()
    
    required_skills = set([skill for skill in SKILLS_DB 
                           if skill in all_job_text])
    
    user_skills = set(resume_skills)
    
    matched = user_skills & required_skills
    missing = required_skills - user_skills
    
    match_score = round(len(matched)/len(required_skills)*100, 2)
    
    return {
        "score": match_score,
        "matched": list(matched),
        "missing": list(missing)
    }

st.title(" SkillSync — Job Skills Gap Analyzer")
st.markdown("** upload  your resume → **")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Upload resume ")
    uploaded_file = st.file_uploader("upload in  PDF format", type=['pdf'])

with col2:
    st.subheader("  Select Job Role ")
    job_options = df['job_title_clean'].value_counts().head(20).index.tolist()
    selected_job = st.selectbox("Available jobs:", job_options)
