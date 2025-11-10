from nlp_core import compute_similarity, extract_keyword_gaps, suggest_bullets

resume = """
Data Scientist with experience in Python, SQL, and machine learning.
Worked on predictive modeling and A/B testing for product optimization.
"""

job_desc = """
We are seeking a Data Scientist skilled in machine learning, data visualization,
and cloud platforms such as AWS. The role involves building predictive models
and communicating insights effectively.
"""

skills_yaml = "../data/skills.yaml"

print("Similarity Score:", compute_similarity(resume, job_desc))
missing = extract_keyword_gaps(resume, job_desc, skills_yaml)
print("Missing Keywords:", missing)
print("Suggested Bullets:", suggest_bullets(missing))
