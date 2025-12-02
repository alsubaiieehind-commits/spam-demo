import streamlit as st
import joblib
import numpy as np

# ---------- Load model ----------
model = joblib.load("spam_model.joblib")
vectorizer = model.named_steps["tfidfvectorizer"]
clf = model.named_steps["multinomialnb"]

st.set_page_config(page_title="Spam Classifier Demo", page_icon="💌", layout="wide")

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    /* Page background */
    body {
        background: #ffffff !important;
        color: #111827 !important;
    }
    .main {
        background: #ffffff !important;
    }
    /* Title + subtitle */
    .title-text {
        font-size: 2.6rem;
        font-weight: 800;
        color: #f9a8d4; /* baby pink */
    }
    .subtitle-text {
        font-size: 1rem;
        color: #6b7280; /* gray-500 */
    }
    /* Cards / boxes */
    .card {
        background: #f3f4f6;  /* light gray */
        border-radius: 1.5rem;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 18px 40px rgba(15,23,42,0.08);
        border: 1px solid #e5e7eb;
        animation: floatIn 0.6s ease-out;
    }
    @keyframes floatIn {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    /* Probability bars */
    .prob-bar-container {
        height: 12px;
        border-radius: 999px;
        background: #e5e7eb; /* gray-200 */
        overflow: hidden;
        margin-top: 4px;
        margin-bottom: 10px;
    }
    .prob-bar-fill-ham {
        height: 100%;
        background: linear-gradient(90deg, #22c55e, #16a34a);
    }
    .prob-bar-fill-spam {
        height: 100%;
        background: linear-gradient(90deg, #fb7185, #e11d48);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session state for message ----------
if "message" not in st.session_state:
    st.session_state.message = ""

# ---------- Layout ----------
col_left, col_right = st.columns([1.2, 1])

with col_left:
    # Title + subtitle
    st.markdown('<div class="title-text">💌 Real-Time Spam Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle-text">'
        'A live demo showing prediction, probabilities, and keyword explanations.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Input card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✍️ Message Input")

    msg = st.text_area(
        "Type a message to analyze:",
        key="message",
        placeholder="Example: Congratulations! You’ve won a cash prize. Click here to claim…",
        height=160,
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    classify_clicked = col_btn1.button("🔍 Classify", use_container_width=True)
    reset_clicked = col_btn2.button("♻️ Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.message = ""
        st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Output card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Output")

    if classify_clicked:
        if not msg.strip():
            st.warning("Please enter a non-empty message.")
        else:
            probs = model.predict_proba([msg])[0]
            pred = model.predict([msg])[0]

            spam_prob = float(probs[1])
            ham_prob = float(probs[0])
            label = "Spam" if pred == 1 else "Not Spam"

            # Prediction badge
            if label == "Spam":
                st.markdown("#### 🚨 Prediction: **SPAM**")
            else:
                st.markdown("#### ✅ Prediction: **NOT SPAM**")

            # Probabilities (text + percentage + bars)
            st.write("**Probabilities**")

            st.write(f"Not Spam: {ham_prob:.2%}")
            st.markdown(
                f'<div class="prob-bar-container">'
                f'<div class="prob-bar-fill-ham" style="width:{ham_prob*100:.1f}%"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.write(f"Spam: {spam_prob:.2%}")
            st.markdown(
                f'<div class="prob-bar-container">'
                f'<div class="prob-bar-fill-spam" style="width:{spam_prob*100:.1f}%"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ---------- Explanation ----------
            st.markdown("### 🧠 Explanation")

            X_vec = vectorizer.transform([msg])
            feature_names = vectorizer.get_feature_names_out()
            word_indices = X_vec.nonzero()[1]
            spam_log_probs = clf.feature_log_prob_[1]

            word_scores = []
            for idx in word_indices:
                word = feature_names[idx]
                score = spam_log_probs[idx]
                word_scores.append((word, score))

            word_scores.sort(key=lambda x: x[1], reverse=True)

            if word_scores:
                top_k = word_scores[:5]
                st.write("Top suspicious keywords in this message:")
                for w, s in top_k:
                    st.write(f"• **{w}**  (spam weight ≈ {s:.2f})")

                keywords = ", ".join([w for w, _ in top_k])
                if label == "Spam":
                    st.info(
                        f"The model marked this as **Spam** mainly because it contains "
                        f"highly spammy words such as: **{keywords}**."
                    )
                else:
                    st.info(
                        f"The model marked this as **Not Spam**. It did not detect strong spam patterns; "
                        f"the detected words (**{keywords}**) are common in normal messages."
                    )
            else:
                st.info(
                    "No known spam keywords from the model's vocabulary appeared in this message. "
                    "The decision is based on the overall word distribution."
                )
    else:
        st.write("Waiting for input… Type a message on the left and press **Classify**.")

    st.markdown('</div>', unsafe_allow_html=True)
