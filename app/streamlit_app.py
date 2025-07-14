import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chain import Chain
from portfolio import Portfolio
from utils import clean_text
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os

# === Load environment variables ===
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

FROM_EMAIL = os.getenv("YOUR_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_email(to_email, email_body):
    msg = MIMEText(email_body)
    msg["Subject"] = "Your Personalized Cold Email"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(FROM_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        return str(e)


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")

    # ✅ Fixed job URL (change manually in code if needed)
    default_url = "https://boards.greenhouse.io/discord/jobs/5890957"
    st.markdown(f"🔗 **Using Job URL**: {default_url}")

    url_input = default_url
    submit_button = st.button("Generate Cold Emails")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            raw_text = loader.load().pop().page_content
            cleaned_data = clean_text(raw_text)

            portfolio.load_portfolio()
            jobs = llm.extract_jobs(cleaned_data)

            for idx, job in enumerate(jobs):
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)

                st.subheader(f"🎯 Job Title: {job.get('role')}")
                if not skills:
                    st.warning("⚠️ No skills found for this job.")
                else:
                    st.markdown("🧠 **Skills:** " + ", ".join(skills))

                st.markdown("#### ✉️ **Generated Cold Email:**")
                st.code(email, language='markdown')

                with st.form(f"send_form_{idx}"):
                    to_email = st.text_input("📨 Enter Gmail to send this email:", key=f"email_input_{idx}")
                    send_button = st.form_submit_button("Send Email")

                    if send_button:
                        if "@" not in to_email:
                            st.error("❌ Invalid email address.")
                        else:
                            result = send_email(to_email, email)
                            if result is True:
                                st.success(f"✅ Email sent successfully to {to_email}!")
                            else:
                                st.error(f"❌ Failed to send email: {result}")

                st.markdown("---")

        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
