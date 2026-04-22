import streamlit as st
import pandas as pd
import PyPDF2
import ast

st.set_page_config(page_title="SkillSync", page_icon="🎯", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("job_descriptions_small.csv")
    df['job_title_clean'] = df['Job Title'].str.lower().str.strip()
    def parse_skills(skill_val):
        if pd.isna(skill_val): return []
        try: return ast.literal_eval(skill_val)
        except: return [s.strip() for s in str(skill_val).split(',')]
    df['skills_list'] = df['skills'].apply(parse_skills)
    return df

df = load_data()

SKILLS_DB = ['python', 'sql', 'r', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'docker', 'aws', 'spark', 'hadoop', 'data analysis', 'statistics']

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join([page.extract_text() for page in reader.pages])

def extract_skills(text):
    return [s for s in SKILLS_DB if s in text.lower()]

def analyze_gap(resume_skills, job_title):
    job_data = df[df['job_title_clean'] == job_title.lower()]
    if job_data.empty: return None
    job_text = " ".join([str(s) for s in job_data['skills_list']]).lower()
    required = {s for s in SKILLS_DB if s in job_text}
    user = set(resume_skills)
    matched = user & required
    missing = required - user
    score = round(len(matched)/len(required)*100, 2) if required else 0
    return {"score": score, "matched": list(matched), "missing": list(missing)}

st.title(" SkillSync — Job Skills Gap Analyzer")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload PDF Resume", type=['pdf'])
with col2:
    jobs = df['job_title_clean'].value_counts().head(20).index.tolist()
    selected_job = st.selectbox("Select Job:", jobs)

if st.button(" Analyze"):
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_skills = extract_skills(resume_text)
    result = analyze_gap(resume_skills, selected_job)
    if result:
        col1, col2, col3 = st.columns(3)
        col1.metric("Match Score", f"{result['score']}%")
        col2.metric("Matched", len(result['matched']))
        col3.metric("Missing", len(result['missing']))
        st.write(" You Have:", result['matched'])
        st.write(" You Need:", result['missing'])
