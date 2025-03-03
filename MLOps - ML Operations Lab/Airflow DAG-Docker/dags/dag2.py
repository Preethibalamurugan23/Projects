from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.smtp.operators.email import EmailOperator
from datetime import datetime, timedelta
import smtplib
import json
import os

# SMTP Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_password"
EMAIL_RECIPIENT = "your_email@gmail.com"

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# File path where the first DAG stores status
STATUS_FILE_PATH = "/opt/airflow/dags/run/status"
PREV_STATE_FILE = "/opt/airflow/dags/run/prev_state.json"

def fetch_new_entries():
    hook = PostgresHook(postgres_conn_id='preethi_news_id')
    conn = hook.get_conn()
    cursor = conn.cursor()
    if not os.path.exists(PREV_STATE_FILE):
        prev_state = {"latest_timestamp": "1970-01-01 00:00:00"}
        with open(PREV_STATE_FILE, "w") as f:
            json.dump(prev_state, f)
    else:
        with open(PREV_STATE_FILE, 'r') as f:
            prev_state = json.load(f)

    cursor.execute("""
        SELECT i.url, a.headline, a.scraped_timestamp 
        FROM images i 
        JOIN articles a ON i.url = a.url
        WHERE a.scraped_timestamp > %s
        ORDER BY a.scraped_timestamp DESC;
    """, (prev_state["latest_timestamp"],))

    new_entries = cursor.fetchall()
    if new_entries:     # Update latest timestamp only if new data exists
        latest_timestamp = new_entries[0][2]  # Get the most recent timestamp
        with open(PREV_STATE_FILE, "w") as f:
            json.dump({"latest_timestamp": latest_timestamp}, f)

    cursor.close()
    conn.close()
    return new_entries


# Function to send email
def send_email(**context):
    new_entries = context['ti'].xcom_pull(task_ids='fetch_new_entries')
    if not new_entries:
        return "No new articles found."
    message_body = "New Articles Added:\n\n"
    for image_url, headline in new_entries:
        message_body += f"Headline: {headline}\nImage: {image_url}\n\n"

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL_SENDER, EMAIL_PASSWORD)
    subject = "New Google News Articles Added"
    email_text = f"Subject: {subject}\n\n{message_body}"
    server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, email_text)
    server.quit()
    return "Email Successfully Sent!"

#delete status file
def delete_status_file():
    if os.path.exists(STATUS_FILE_PATH):
        os.remove(STATUS_FILE_PATH)

# Define DAG
with DAG(
    dag_id='send_email_notification',
    default_args=default_args,
    description='Send an email when new articles are added',
    schedule_interval=None,  # Triggered by FileSensor
    start_date=datetime(2025, 2, 16),
    catchup=False,
) as dag:

    # FileSensor to trigger DAG when status file is created
    wait_for_status = FileSensor(
        task_id='wait_for_status',
        filepath=STATUS_FILE_PATH,
        poke_interval=30,  
        timeout=3600, 
        mode='poke'
    )

    fetch_new_entries_task = PythonOperator(
        task_id='fetch_new_entries',
        python_callable=fetch_new_entries,
    )

    send_email_task = PythonOperator(
        task_id='send_email',
        python_callable=send_email,
        provide_context=True,
    )

    cleanup_task = PythonOperator(  # delete the status file after sending email
        task_id='delete_status_file',
        python_callable=delete_status_file,
    )

    wait_for_status >> fetch_new_entries_task >> send_email_task >> cleanup_task
