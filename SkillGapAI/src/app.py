import streamlit as st
import pdfplumber
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


st.set_page_config(page_title="SkillGapAI", layout="centered")

st.title("SkillGapAI - AI Resume Analyzer")
st.write("Upload your resume and compare it with a job description to identify skill gaps.")


# Load Skills Database


def load_skills():
    with open("skills.txt") as f:
        skills = [line.strip().lower() for line in f]
    return skills


skills_db = load_skills()


# Resume Text Extraction


def extract_resume_text(uploaded_file):

    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

    except Exception as e:
        st.error("Error reading the PDF file.")
        return ""

    return text


# Text Preprocessing


stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Skill Extraction


def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills_db:

        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))


# Streamlit UI


uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_description = st.text_area("Paste Job Description")


# Resume Analysis


if st.button("Analyze Resume"):

    if uploaded_file and job_description:

        with st.spinner("Analyzing Resume..."):

            resume_text = extract_resume_text(uploaded_file)

            if resume_text == "":
                st.stop()

            # Preprocess text
            clean_resume = preprocess(resume_text)
            clean_jd = preprocess(job_description)

            # Extract skills
            resume_skills = extract_skills(clean_resume)
            job_skills = extract_skills(clean_jd)

            resume_skill_text = " ".join(resume_skills)
            job_skill_text = " ".join(job_skills)

            # TF-IDF similarity
            vectorizer = TfidfVectorizer()

            tfidf_matrix = vectorizer.fit_transform([resume_skill_text, job_skill_text])

            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

            score = similarity[0][0]

            # Skill comparison
            matched_skills = list(set(resume_skills) & set(job_skills))

            missing_skills = [skill for skill in job_skills if skill not in resume_skills]

            match_ratio = len(matched_skills) / len(job_skills) if job_skills else 0


            # Match category
            if match_ratio < 0.25:
                category = "🔴 High Improvement Required"

            elif match_ratio < 0.50:
                category = "🟠 Moderate Alignment"

            elif match_ratio < 0.75:
                category = "🟡 Strong Alignment"

            else:
                category = "🟢 Excellent Match"


        # -------------------------------
        # Output Section
        # -------------------------------

        st.subheader("Resume Analysis")

        st.metric("Similarity Score", round(score, 2))

        st.write("### Match Level")
        st.write(category)


        st.write("### Matched Skills")

        if matched_skills:
            st.success(", ".join(matched_skills))
        else:
            st.write("No matching skills detected.")


        st.write("### Missing Skills")

        if missing_skills:
            st.error(", ".join(missing_skills))
        else:
            st.success("No missing skills detected!")


    else:

        st.warning("Please upload a resume and paste a job description.")


# Footer
st.markdown("---")
st.caption("SkillGapAI | Made by Abhilasha")