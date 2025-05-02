from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize Flask app
app = Flask(__name__)

# CORS configuration
# TEMP: Allow all origins for development
# CORS(app)

# RECOMMENDED: Allow only your deployed frontend
CORS(app, origins=["https://my-frontend.vercel.app"])

# Load saved components
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('job_vectors.pkl', 'rb') as f:
    job_vectors = pickle.load(f)

df = pd.read_csv('job_data.csv')

# Root route
@app.route('/')
def index():
    return 'Career Recommendation API is live. Use POST /recommend'

# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    education = data.get('education', '')
    skills = data.get('skills', '')
    interests = data.get('interests', '')
    certifications = data.get('certifications', '')

    # Combine user input
    user_profile = f"{education} {skills} {interests} {certifications}"
    user_vector = vectorizer.transform([user_profile])
    similarity = cosine_similarity(user_vector, job_vectors)

    top_indices = similarity.argsort()[0][-5:][::-1]
    results = df.iloc[top_indices][['Job Title', 'Role', 'Industry', 'sal']].to_dict(orient='records')

    return jsonify(results)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
