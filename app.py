import streamlit as st
import requests

API_URL = "https://spam-detector-production-b00d.up.railway.app/predict"

st.set_page_config(page_title="Spam Detector", page_icon="🛡️")

st.title("🛡️ Spam Detector")
st.write("Enter any message below to check if it's spam or not.")

email_text = st.text_area("Message", placeholder="Type your message here...", height=150)

if st.button("Check"):
    if email_text.strip() == "":
        st.warning("Please enter a message first!")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json={"text": email_text})
            result = response.json()

        if result["prediction"] == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam!")