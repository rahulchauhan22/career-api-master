import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load saved components
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('job_vectors.pkl', 'rb') as f:
    job_vectors = pickle.load(f)

df = pd.read_csv('job_data.csv')

# Flask app setup
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    # Extract user input
    education = data.get('education', '')
    skills = data.get('skills', '')
    interests = data.get('interests', '')
    certifications = data.get('certifications', '')

    # Combine into a profile
    user_profile = f"{education} {skills} {interests} {certifications}"
    user_vector = vectorizer.transform([user_profile])
    similarity = cosine_similarity(user_vector, job_vectors)
    top_indices = similarity.argsort()[0][-5:][::-1]
    results = df.iloc[top_indices][['Job Title', 'Role', 'Industry', 'sal']].to_dict(orient='records')

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))


