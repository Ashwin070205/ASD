import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
from PIL import Image
import csv
from datetime import datetime
import hashlib
import pickle
import os
import re
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors

BASE_DIR = os.path.dirname(__file__)
SESSION_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

def save_session(username, data):
    file = os.path.join(SESSION_DIR, f"{username}.csv")

    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.path.getsize(file) == 0:
            writer.writerow(["Time","Age","Gender","Final_Result","Image_Confidence"])
        writer.writerow(data)

def load_sessions(username):
    file = os.path.join(SESSION_DIR, f"{username}.csv")
    if not os.path.exists(file):
        return []
    with open(file, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))[1:]

USERS_FILE = os.path.join(BASE_DIR, "users.json")
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

users = load_users()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)

def strong_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain an uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain a lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain a number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain a special character"
    return True, ""

def signup():
    st.markdown('<div class="auth-title">Create Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle"></div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    st.markdown('<div class="auth-btn">', unsafe_allow_html=True)
    signup_btn = st.button("Create Account")
    st.markdown('</div>', unsafe_allow_html=True)

    if signup_btn:
        if not username or not email or not password:
            st.error("All fields are required")
        elif username in users:
            st.error("Username already exists")
        elif not valid_email(email):
            st.error("Invalid email")
        else:
            strong, msg = strong_password(password)
            if not strong:
                st.error(msg)
            elif password != confirm:
                st.error("Passwords do not match")
            else:
                users[username] = {
                    "email": email,
                    "password": hash_password(password)
                }
                save_users(users)
                st.success("Account created successfully")
                st.session_state.auth_mode = "login"
                st.rerun()

    st.markdown('<div class="auth-switch">Already have an account? <span>Login</span></div>', unsafe_allow_html=True)
    if st.button("Back to Login"):
        st.session_state.auth_mode = "login"
        st.rerun()


def login():
    st.markdown('<div class="auth-title">🔐 Welcome </div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle"></div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    st.markdown('<div class="auth-btn">', unsafe_allow_html=True)
    login_btn = st.button("Login")
    st.markdown('</div>', unsafe_allow_html=True)

    if login_btn:
        if username in users and users[username]["password"] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.user = username
            st.session_state.email = users[username]["email"]
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown('<div class="auth-switch">Don\'t have an account? <span>Sign up</span></div>', unsafe_allow_html=True)
    if st.button("Create Account"):
        st.session_state.auth_mode = "signup"
        st.rerun()

def generate_report(
    username, age, gender,
    answers, demographics,
    tabular_label, image_label, image_confidence,
    final_label, suggestions,
    image_path
):

    file_name = os.path.join(BASE_DIR, f"ASD_Report_{username}.pdf")

    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )

    story.append(Paragraph("Autism Spectrum Disorder Screening Report", title_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>User:</b> {username}", styles['Normal']))
    story.append(Paragraph(f"<b>Age (Months):</b> {age}", styles['Normal']))
    story.append(Paragraph(f"<b>Gender:</b> {gender}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Questionnaire
    story.append(Paragraph("<b>Questionnaire Responses</b>", styles['Heading2']))
    for q, ans in answers.items():
        story.append(Paragraph(f"{q}: <b>{ans}</b>", styles['Normal']))

    story.append(Spacer(1, 12))

    # Demographics
    story.append(Paragraph("<b>Demographics</b>", styles['Heading2']))
    for k, v in demographics.items():
        story.append(Paragraph(f"{k}: <b>{v}</b>", styles['Normal']))

    story.append(Spacer(1, 12))

    # Image
    story.append(Paragraph("<b>Uploaded Image</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    story.append(RLImage(image_path, width=2.5*inch, height=2.5*inch))
    story.append(Spacer(1, 12))

    # Predictions
    story.append(Paragraph("<b>Model Predictions</b>", styles['Heading2']))
    story.append(Paragraph(f"Tabular Model: {tabular_label}", styles['Normal']))
    story.append(Paragraph(
        f"Image Model: {image_label} ({image_confidence*100:.2f}%)",
        styles['Normal']
    ))
    story.append(Paragraph(f"<b>Final Result: {final_label}</b>", styles['Normal']))

    story.append(Spacer(1, 12))

    # Suggestions
    story.append(Paragraph("<b>Suggestions</b>", styles['Heading2']))
    story.append(Paragraph(suggestions.replace("\n","<br/>"), styles['Normal']))

    doc = SimpleDocTemplate(file_name, pagesize=A4)
    doc.build(story)

    return file_name


def logout():
    with st.sidebar:
        st.markdown("<div class='sidebar-title'style='font-size:30px; font-weight:bold;'>🧠 ASD System</div>", unsafe_allow_html=True)

        # User info
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-label'style='font-size:16px; font-weight:bold;'>Username</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-value'>{st.session_state.user}</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-label'style='font-size:16px; font-weight:bold;'>Email</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-value'>{st.session_state.email}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 🔥 Load real sessions
        sessions = load_sessions(st.session_state.user)

        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-label'style='font-size:16px; font-weight:bold;'>Recent Sessions</div>", unsafe_allow_html=True)

        if sessions:
            for s in sessions[::-1][:5]:
                st.markdown(f"<div class='sidebar-item'>🗂 {s[0]} → {s[3]}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='sidebar-item'>No tests yet</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        # How to use system
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-label'style='font-size:16px; font-weight:bold;'>How to Use System</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='sidebar-item'>1️⃣ Fill questionnaire carefully</div>
        <div class='sidebar-item'>2️⃣ Enter child demographic details</div>
        <div class='sidebar-item'>3️⃣ Upload clear face image</div>
        <div class='sidebar-item'>4️⃣ Click Predict ASD</div>
        <div class='sidebar-item'>5️⃣ View final result</div>
        <div class='sidebar-item'>6️⃣ Provide feedback (optional)</div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Logout
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.email = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def auth_page():

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    st.markdown('<div class="auth-container"><div class="auth-card">', unsafe_allow_html=True)

    if st.session_state.auth_mode == "login":
        login()
    else:
        signup()

    st.markdown('</div></div>', unsafe_allow_html=True)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    auth_page()
    st.stop()
else:
    logout()


# ================= LOAD MODELS =================
import os
BASE_DIR = os.path.dirname(__file__)
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
tabular_model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model.sav"))

image_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "asd_cnn_model1.keras")
)

class_names = ["Autistic", "Non_Autistic"]


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="ASD Detection System",
    page_icon="🧠",
    layout="wide"
)


# ================= STYLES =================
st.markdown("""
<style>

.auth-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 85vh;
}

.auth-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 40px;
    width: 420px;
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
    animation: fadeIn 0.6s ease;
}

@keyframes fadeIn {
    from {opacity:0; transform: translateY(20px);}
    to {opacity:1; transform: translateY(0);}
}

.auth-title {
    text-align: center;
    font-size: 28px;
    color: white;
    font-weight: bold;
    margin-bottom: 20px;
}

.auth-subtitle {
    text-align: center;
    color: #aaa;
    margin-bottom: 30px;
}

div[data-baseweb="input"] input {
    background-color: #0f172a !important;
    border-radius: 12px !important;
    border: 1px solid #1e293b !important;
    padding: 14px !important;
    color: white !important;
}

.auth-btn button {
    width: 100%;
    background: linear-gradient(90deg,#6366f1,#22d3ee);
    border-radius: 14px;
    padding: 12px;
    font-weight: bold;
    font-size: 16px;
    border: none;
    color: white;
}

.auth-btn button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 15px #38bdf8;
}

.auth-switch {
    text-align: center;
    margin-top: 20px;
    color: #94a3b8;
}

.auth-switch span {
    color: #38bdf8;
    cursor: pointer;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    padding: 15px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 20px;
}

.sidebar-card {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
}

.sidebar-label {
    font-size: 12px;
    color: #94a3b8;
}

.sidebar-value {
    font-size: 14px;
    font-weight: bold;
    color: white;
}

.sidebar-item {
    padding: 8px;
    border-radius: 8px;
    cursor: pointer;
    color: #cbd5f5;
}

.sidebar-item:hover {
    background: #1e293b;
}

</style>
""", unsafe_allow_html=True)



# ================= TITLE =================
st.markdown("<h1 style='text-align:center;'>🧠 Autism Spectrum Disorder Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================= FORM =================
with st.form("asd_form"):
    st.subheader("📝 Questionnaire")

    col1, col2 = st.columns(2)

    with col1:
        a1 = st.selectbox("Does your child look at you when you call his/her name?", ["YES", "NO"])
        a2 = st.selectbox("Is it easy for you to get eye contact with your child?", ["YES", "NO"])
        a3 = st.selectbox("Does your child point to indicate that she/he wants something?", ["YES", "NO"])
        a4 = st.selectbox("Does your child point to share interest with you", ["YES", "NO"])
        a5 = st.selectbox("Does your child pretend? (e.g. care for dolls, talk on a toy phone)", ["YES", "NO"])

    with col2:
        a6 = st.selectbox("Does your child follow where you’re looking?", ["YES", "NO"])
        a7 = st.selectbox("If someone is upset, does your child show comfort?", ["YES", "NO"])
        a8 = st.selectbox("Would you describe your child’s first words?", ["YES", "NO"])
        a9 = st.selectbox("Does your child use simple gestures? (e.g. wave goodbye)", ["YES", "NO"])
        a10 = st.selectbox("Does your child stare at nothing with no purpose?", ["YES", "NO"])

    st.subheader("👤 Demographics")

    col3, col4, col5 = st.columns(3)

    with col3:
        age = st.slider("Age (Months)", 12, 36, 18)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col4:
        jaundice = st.selectbox("Jaundice at birth?", ["YES", "NO"])
        family_asd = st.selectbox("Family member with ASD?", ["YES", "NO"])

    with col5:
        ethnicity = st.selectbox("Ethnicity", [
            "Middle Eastern", "White European", "Hispanic", "Asian",
            "South Asian", "Native Indian", "Black", "Latino",
            "Mixed", "Pacifica", "Others"
        ])
        who = st.selectbox("Who completed the test?", [
            "Family member", "Health Care Professional", "Self", "Others"
        ])

    st.subheader("🖼 Upload Child Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    submit = st.form_submit_button("🔍 Predict ASD")
    
# ================= PREDICTION =================
if submit and uploaded_image:

    # ---- Encode inputs ----
    yn = lambda x: 1 if x == "YES" else 0

    ethnicity_map = {
        "Middle Eastern": 0, "White European": 1, "Hispanic": 2, "Others": 3,
        "Asian": 4, "South Asian": 5, "Native Indian": 6, "Black": 7,
        "Latino": 8, "Mixed": 9, "Pacifica": 10
    }

    who_map = {
        "Family member": 0,
        "Health Care Professional": 1,
        "Self": 2,
        "Others": 3
    }

    tabular_input = [
        yn(a1), yn(a2), yn(a3), yn(a4), yn(a5),
        yn(a6), yn(a7), yn(a8), yn(a9), yn(a10),
        age,
        1 if gender == "Male" else 0,
        ethnicity_map[ethnicity],
        yn(jaundice),
        yn(family_asd),
        who_map[who]
    ]

    # ---- TABULAR MODEL ----
    tabular_array = np.asarray(tabular_input).reshape(1, -1)
    tabular_scaled = scaler.transform(tabular_array)
    tabular_pred = tabular_model.predict(tabular_scaled)[0]

    tabular_score = 1.0 if tabular_pred == 1 else 0.0
    tabular_label = "ASD" if tabular_pred == 1 else "Non-ASD"

    # ---- IMAGE MODEL ----
    image = Image.open(uploaded_image).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    preds = image_model.predict(img_array)
    image_score = tf.nn.softmax(preds[0]).numpy()
    image_label = class_names[np.argmax(image_score)]
    image_confidence = np.max(image_score)

    # ---- FINAL COMBINATION ----
    final_score = (tabular_score + image_confidence) / 2

    st.markdown("---")
    st.subheader("📊 Results")

    st.write(f"**Tabular Model:** {tabular_label}")
    st.write(f"**Image Model:** {image_label} ({image_confidence*100:.2f}%)")

    if final_score >= 0.5:
        st.error("🧩 FINAL RESULT: Patient is ASD")
        final_label = "ASD"
    else:
        st.success("✅ FINAL RESULT: Patient is NON-ASD")
        final_label = "NON-ASD"
        
    st.markdown("---")
    st.subheader("💡 Suggestions")

    if final_label == "ASD":
        suggestions = """
    Recommended Actions
    • Consult a pediatrician or child psychologist
    • Consider professional ASD screening
    • Start early intervention programs
    • Encourage eye contact and social interaction
    • Engage in speech and occupational therapy
    • Monitor behavioral patterns regularly
    • Join parent support groups
    """
        st.warning(suggestions)

    else:
        suggestions = """
    General Guidance
    • Continue monitoring developmental milestones
    • Encourage social interaction and play activities
    • Maintain regular pediatric check-ups
    • Engage child in communication exercises
    • Provide balanced learning environment
    • Repeat screening if concerns arise
    """
        st.info(suggestions)
        
    # ---------- Generate Report ----------
    # save uploaded image
    image_path = os.path.join(BASE_DIR, f"temp_{st.session_state.user}.jpg")
    image.save(image_path)

    answers = {
        "Looks when name called": a1,
        "Eye contact": a2,
        "Points for needs": a3,
        "Points to share": a4,
        "Pretend play": a5,
        "Follows gaze": a6,
        "Shows comfort": a7,
        "First words": a8,
        "Gestures": a9,
        "Stares blankly": a10
    }

    demographics = {
        "Jaundice": jaundice,
        "Family ASD": family_asd,
        "Ethnicity": ethnicity,
        "Completed by": who
    }

    report_file = generate_report(
        st.session_state.user,
        age,
        gender,
        answers,
        demographics,
        tabular_label,
        image_label,
        image_confidence,
        final_label,
        suggestions,
        image_path
    )

    with open(report_file, "rb") as f:
        st.download_button(
            "📄 Download Full Report",
            f,
            file_name=report_file,
            mime="application/pdf"
        )
        
    save_session(
    st.session_state.user,
    [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        age,
        gender,
        final_label,
        f"{image_confidence*100:.2f}%"
    ]
)

elif submit:
    st.warning("⚠ Please upload an image to proceed.")

#Feedback
st.markdown("---")
st.subheader("🗣️ Feedback")

with st.form("feedback_form"):
    rating = st.slider(
        "How helpful was this prediction?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Not helpful, 5 = Very helpful"
    )

    comment = st.text_area(
        "Any comments or suggestions? (optional)",
        placeholder="Your feedback helps us improve this system..."
    )

    submit_feedback = st.form_submit_button("📨 Submit Feedback")

if submit_feedback:
    feedback_data = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rating,
        comment
    ]

    with open(os.path.join(BASE_DIR, "feedback.csv"), "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(feedback_data)

    st.success("🙏 Thank you for your feedback!")