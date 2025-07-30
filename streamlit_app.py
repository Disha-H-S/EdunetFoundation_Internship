import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('salary_model.pkl', 'rb') as f:
        return pickle.load(f)

def load_encoders():
    encoders = dict()
    with open('gender_encoder_new.pkl', 'rb') as f:
        encoders['gender'] = pickle.load(f)
    with open('education_encoder_new.pkl', 'rb') as f:
        encoders['education'] = pickle.load(f)
    with open('job_encoder_new.pkl', 'rb') as f:
        encoders['job_title'] = pickle.load(f)
    return encoders
model = load_model()
encoders = load_encoders()
# --- Streamlit UI ---
st.title("Linear Regression Predictor")
st.write("This app predicts an employee's salary based on a given set of features.")

# --- User Input ---
st.header("Enter Feature Values")

# Categorical Inputs
Education_level = st.selectbox("Education level", ["Bachelor's","Master's","PhD"])
gender = st.selectbox("Gender", ['Male', 'Female'])
job_title = st.selectbox("Job Title", ["Software Engineer",
"Data Analyst",
"Senior Manager",
"Sales Associate",
"Director",
"Marketing Analyst",
"Product Manager",
"Sales Manager",
"Marketing Coordinator",
"Senior Scientist",
"Software Developer",
"HR Manager",
"Financial Analyst",
"Project Manager",
"Customer Service Rep",
"Operations Manager",
"Marketing Manager",
"Senior Engineer",
"Data Entry Clerk",
"Sales Director",
"Business Analyst",
"VP of Operations",
"IT Support",
"Recruiter",
"Financial Manager",
"Social Media Specialist",
"Software Manager",
"Junior Developer",
"Senior Consultant",
"Product Designer",
"CEO",
"Accountant",
"Data Scientist",
"Marketing Specialist",
"Technical Writer",
"HR Generalist",
"Project Engineer",
"Customer Success Rep",
"Sales Executive",
"UX Designer",
"Operations Director",
"Network Engineer",
"Administrative Assistant",
"Strategy Consultant",
"Copywriter",
"Account Manager",
"Director of Marketing",
"Help Desk Analyst",
"Customer Service Manager",
"Business Intelligence Analyst",
"Event Coordinator",
"VP of Finance",
"Graphic Designer",
"UX Researcher",
"Social Media Manager",
"Director of Operations",
"Senior Data Scientist",
"Junior Accountant",
"Digital Marketing Manager",
"IT Manager",
"Customer Service Representative",
"Business Development Manager",
"Senior Financial Analyst",
"Web Developer",
"Research Director",
"Technical Support Specialist",
"Creative Director",
"Senior Software Engineer",
"Human Resources Director",
"Content Marketing Manager",
"Technical Recruiter",
"Sales Representative",
"Chief Technology Officer",
"Junior Designer",
"Financial Advisor",
"Junior Account Manager",
"Senior Project Manager",
"Principal Scientist",
"Supply Chain Manager",
"Senior Marketing Manager",
"Training Specialist",
"Research Scientist",
"Junior Software Developer",
"Public Relations Manager",
"Operations Analyst",
"Product Marketing Manager",
"Senior HR Manager",
"Junior Web Developer",
"Senior Project Coordinator",
"Chief Data Officer",
"Digital Content Producer",
"IT Support Specialist",
"Senior Marketing Analyst",
"Customer Success Manager",
"Senior Graphic Designer",
"Software Project Manager",
"Supply Chain Analyst",
"Senior Business Analyst",
"Junior Marketing Analyst",
"Office Manager",
"Principal Engineer",
"Junior HR Generalist",
"Senior Product Manager",
"Junior Operations Analyst",
"Senior HR Generalist",
"Sales Operations Manager",
"Senior Software Developer",
"Junior Web Designer",
"Senior Training Specialist",
"Senior Research Scientist",
"Junior Sales Representative",
"Junior Marketing Manager",
"Junior Data Analyst",
"Senior Product Marketing Manager",
"Junior Business Analyst",
"Senior Sales Manager",
"Junior Marketing Specialist",
"Junior Project Manager",
"Senior Accountant",
"Director of Sales",
"Junior Recruiter",
"Senior Business Development Manager",
"Senior Product Designer",
"Junior Customer Support Specialist",
"Senior IT Support Specialist",
"Junior Financial Analyst",
"Senior Operations Manager",
"Director of Human Resources",
"Junior Software Engineer",
"Senior Sales Representative",
"Director of Product Management",
"Junior Copywriter",
"Senior Marketing Coordinator",
"Senior Human Resources Manager",
"Junior Business Development Associate",
"Senior Account Manager",
"Senior Researcher",
"Junior HR Coordinator",
"Director of Finance",
"Junior Marketing Coordinator"])

# Numeric Inputs
age = st.number_input("Age", min_value=0, max_value=100, value=30)
experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

print("type")
print(type(gender))
# Submit button
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age" : age,
        'Gender': gender.lower(),
        'Education Level': Education_level.lower(),
        'Job Title': job_title.lower(),
        'Years of Experience': experience_years
    }])

    input_df['Gender'] = encoders["gender"].transform(input_df['Gender'])
    input_df['Education Level'] = encoders["education"].transform(input_df['Education Level'])
    input_df['Job Title'] = encoders["job_title"].transform(input_df['Job Title'])

    prediction = model.predict(input_df)
    st.success(f":chart_with_upwards_trend: Predicted Output: **{prediction[0]:.2f}**")

# Optional: Display raw input
with st.expander("See input data"):
    st.write(pd.DataFrame([{
        "Age" : age,
        'Gender': gender,
        'Education Level': Education_level,
        'Job Title': job_title,
        'Years of Experience': experience_years
    }]))
