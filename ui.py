import streamlit as st
import joblib
import ast
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# NLTK SETUP
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
tfidf = joblib.load("tfidf.pkl")
job_vectors = joblib.load("job_vectors.pkl")
df = joblib.load("jobs_df.pkl")

# -----------------------------
# CONFIG
# -----------------------------
CORE_SKILLS = {
    "python", "machine learning", "deep learning", "data science",
    "sql", "tensorflow", "pytorch", "nlp", "computer vision",
    "java", "javascript", "react", "node", "docker", "aws"
}

TOP_N = 3
MIN_TFIDF_SCORE = 0.15
MIN_SKILL_MATCH_PERCENT = 10

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Job Recommendation System", layout="centered")

st.title("💼 Job Recommendation System")
st.caption("Add your skills, rate them, and get the best matching jobs")

st.divider()

# -----------------------------
# SESSION STATE
# -----------------------------
if "skills" not in st.session_state:
    st.session_state.skills = {}

# -----------------------------
# ADD SKILLS
# -----------------------------
st.subheader("➕ Add Your Skills")

new_skill = st.text_input("Enter a skill (e.g. Python, React, ML)")

col1, col2 = st.columns(2)

with col1:
    if st.button("Add Skill"):
        if new_skill.strip():
            st.session_state.skills[new_skill.lower().strip()] = 5

with col2:
    if st.button("Clear All Skills"):
        st.session_state.skills = {}

st.divider()

# -----------------------------
# SKILL SLIDERS
# -----------------------------
st.subheader("🎯 Rate Your Skills (0–10)")

if not st.session_state.skills:
    st.info("Add skills above to rate them")
else:
    for skill in list(st.session_state.skills.keys()):
        st.session_state.skills[skill] = st.slider(
            skill.capitalize(), 0, 10, st.session_state.skills[skill], key=skill
        )

st.divider()

# -----------------------------
# RECOMMENDATION LOGIC
# -----------------------------
if st.button("🔍 Recommend Best Jobs", use_container_width=True):

    if not st.session_state.skills:
        st.warning("Please add at least one skill")
    else:
        with st.spinner("Analyzing your skills..."):

            user_skills = {k: v for k, v in st.session_state.skills.items() if v > 0}
            user_skill_names = set(user_skills.keys())

            weighted_text = []
            for skill, weight in user_skills.items():
                weighted_text.extend([skill] * weight)

            user_input = clean_text(" ".join(weighted_text))
            user_vector = tfidf.transform([user_input])

            scores = cosine_similarity(user_vector, job_vectors).flatten()
            results = []

            for idx, score in enumerate(scores):
                if score < MIN_TFIDF_SCORE:
                    continue

                job = df.iloc[idx]

                try:
                    job_skills = ast.literal_eval(job["job_skill_set"])
                except:
                    continue

                job_skills_lower = {s.lower() for s in job_skills}

                if not (job_skills_lower & CORE_SKILLS & user_skill_names):
                    continue

                matched = []
                for js in job_skills_lower:
                    for us in user_skill_names:
                        if us in js or js in us:
                            matched.append(js)

                matched = list(set(matched))
                total = len(job_skills_lower)
                match_percent = (len(matched) / total) * 100 if total else 0

                if match_percent < MIN_SKILL_MATCH_PERCENT:
                    continue

                missing = list(job_skills_lower - set(matched))
                priority_missing = [s for s in missing if s in CORE_SKILLS]
                other_missing = [s for s in missing if s not in CORE_SKILLS]

                results.append({
                    "job_title": job["job_title"],
                    "match_score": round(score, 2),
                    "skill_match_percentage": round(match_percent, 2),
                    "matched_skills": matched[:5],
                    "missing_skills": (priority_missing + other_missing)[:5]
                })

            results = sorted(results, key=lambda x: x["match_score"], reverse=True)

        if not results:
            st.warning("No suitable jobs found. Try adding more skills.")
        else:
            st.success("✅ Best Job Matches")

            for i, job in enumerate(results[:TOP_N], start=1):
                st.markdown(f"## {i}. {job['job_title']}")
                st.progress(int(job["match_score"] * 100))
                st.write(f"**Overall Match:** {int(job['match_score'] * 100)}%")
                st.write(f"**Skill Coverage:** {job['skill_match_percentage']}%")

                colA, colB = st.columns(2)

                with colA:
                    st.markdown("✅ **Matched Skills**")
                    for s in job["matched_skills"]:
                        st.write(f"• {s.capitalize()}")

                with colB:
                    st.markdown("❌ **Skill Gap (Learn Next)**")
                    for s in job["missing_skills"]:
                        st.write(f"• {s.capitalize()}")

                st.divider()
