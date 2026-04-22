# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r"C:\Users\ninja\Downloads\job_descriptions.csv")




# only  5000 rows 
df = df.head(5000)

# Save karo
df.to_csv("job_descriptions_small.csv", index=False)
print(" Small file saved")

print(" Shape:", df.shape)
print("\n Columns:", df.columns.tolist())


# %%
###### data quality check 
print("Missing values : ")
print(df.isnull().sum())

print("\n Data types : ")
print(df.dtypes)

print("\n Skills column sample :")
print(df['skills'].head(10))


# %%
# Top 10 Job Titles
plt.figure(figsize=(12,5))
df['Job Title'].value_counts().head(10).plot(kind='barh', color='steelblue')
plt.title('Top 10 job titles')
plt.tight_layout()
plt.show()

# Job Level Distribution
plt.figure(figsize=(8,4))
df['Job Title'].str.extract('(Senior|Junior|Lead|Manager|Intern)')[0]\
.value_counts().plot(kind='bar', color='coral')
plt.title('Job level distribution')
plt.tight_layout()
plt.show()

# %%
# Check format in skills column 
print(type(df['skills'][0]))
print(df['skills'][0])

# %%
import ast

def parse_skills(skill_val):
    if pd.isna(skill_val):
        return []
    try:
        
        return ast.literal_eval(skill_val)
    except:
        
        return [s.strip() for s in str(skill_val).split(',')]

df['skills_list'] = df['skills'].apply(parse_skills)


print(df['skills_list'].head(5))

# %%
# all skills in  one list 
all_skills = [skill for sublist in df['skills_list'] for skill in sublist]

# Top 20 skills
top_skills = Counter(all_skills).most_common(20)
skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=skills_df, x='Count', y='Skill', palette='viridis')
plt.title('Top 20 Most In-Demand Skills')
plt.tight_layout()
plt.show()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# convert skills list to string
df['skills_str'] = df['skills_list'].apply(lambda x: ' '.join(x))

# Job Title clean karo
df['job_title_clean'] = df['Job Title'].str.lower().str.strip()

print(df[['job_title_clean', 'skills_str']].head(3))

# %%
# TF-IDF — covert every skills to vectore
tfidf = TfidfVectorizer(max_features=100)
skills_matrix = tfidf.fit_transform(df['skills_str'])

print(" Skills matrix shape:", skills_matrix.shape)
print(" Sample features:", tfidf.get_feature_names_out()[:10])

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Target — predict job Title from skills 
le = LabelEncoder()
df['job_label'] = le.fit_transform(df['job_title_clean'])

# Train/Test Split
X = skills_matrix
y = df['job_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_, labels=np.unique(y_pred)))

# %%
#  check how many unique job titles
print("Unique Jobs:", df['job_title_clean'].nunique())

# %%
import joblib

joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(le, 'label_encoder.pkl')

print(" Model saved successfully!")

# %%
# Hardcoded proper skills list
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

def extract_skills(text):
    text_lower = text.lower()
    found_skills = [skill for skill in SKILLS_DB 
                    if skill in text_lower]
    return found_skills


# Re-run karo
resume_text = """
python sql machine learning deep learning
tensorflow scikit-learn pandas numpy statistics nlp r
"""

resume_skills = extract_skills(resume_text)
print(" Skills:", resume_skills)



# %%
def analyze_gap(resume_skills, job_title):
    #  extract the Job data 
    job_data = df[df['job_title_clean'] == job_title.lower()]
    
    if job_data.empty:
        print(" Job not found! Available jobs:")
        print(df['job_title_clean'].value_counts().head(10))
        return
    
    # Also match the skills in the dataset to skills_db
    all_job_text = " ".join([
        str(skills) for skills in job_data['skills_list']
    ]).lower()
    
    #  Required skills — Match the text of the dataset with SKILLS_DB
    required_skills = set([
        skill for skill in SKILLS_DB 
        if skill in all_job_text
    ])
    
    user_skills = set(resume_skills)
    
    #  calculate gap
    matched = user_skills & required_skills
    missing = required_skills - user_skills
    
    match_score = round(len(matched) / len(required_skills) * 100, 2) if required_skills else 0
    
    print(f"\n Job: {job_title}")
    print(f" Match Score: {match_score}%")
    print(f" Skills You Have ({len(matched)}): {matched}")
    print(f" Skills You Need ({len(missing)}): {missing}")
    
    return {
        "score": match_score,
        "matched": list(matched),
        "missing": list(missing)
    }


result = analyze_gap(resume_skills, "data scientist")

# %%



print(df['job_title_clean'].value_counts().head(15))

# %%
import os
print(os.getcwd())

# %%


# %%



