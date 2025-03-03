from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import logging
from google_news_scraper import get_google_news_homepage, parse_google_news, save_articles
import os
import json

# # Function to load configuration
# def load_config():
#     import json
#     with open('config.json', 'r') as f:
#         config = json.load(f)
#     return config

def load_config():
    import json
    import os
    print("Current Working Directory:", os.getcwd())
    try:
        with open('/opt/airflow/dags/config.json', 'r') as f:
            config = json.load(f)
        print("Config Loaded:", config)
        return config
    except FileNotFoundError:
        print("config.json not found.")
        raise


# Task 1: Get and Parse Articles
def get_and_parse_articles(**context):
    config = context['ti'].xcom_pull(task_ids='load_config')
    html = get_google_news_homepage(config.get("google_news_url", "https://news.google.com/"))
    if not html:
        raise Exception("Failed to fetch Google News homepage.")
    articles = parse_google_news(html)
    context['ti'].xcom_push(key='articles', value=articles)

# Task 2: Save Articles to File
def save_article_data(**context):
    config = context['ti'].xcom_pull(task_ids='load_config')
    articles = context['ti'].xcom_pull(task_ids='get_and_parse_articles', key='articles')
    if not articles:
        logging.warning("No articles to save.")
        return
    output_file_path = ("news_articles.json") # Corrected file name
    print(f"Saving articles to: {os.path.abspath(output_file_path)}")
    save_articles(articles, output_file_path)
    loggingart = json.load(open('news_articles.json', 'r')) if os.path.exists('articles_log.json') else None
    logging.info(loggingart)
    # save_articles(articles, config.get("output_file", "news_articles.json"))

create_image_table = PostgresOperator(
    task_id="create_image_table",
    postgres_conn_id="preethi_news_id",  # Replace with your connection ID
    sql="""
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,  -- Auto-incrementing ID
            url TEXT UNIQUE NOT NULL, -- URL of the image, ensures uniqueness
            image_data TEXT NOT NULL -- Base64 encoded image data
        );
    """,
)

create_article_table = PostgresOperator(
    task_id="create_article_table",
    postgres_conn_id="preethi_news_id",  # Replace with your connection ID
    sql="""
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,  -- Auto-incrementing ID
            url TEXT UNIQUE NOT NULL, -- URL of the article, ensures uniqueness
            headline TEXT NOT NULL,
            article_date DATE,  -- Date of the article
            scraped_timestamp TIMESTAMP WITH TIME ZONE NOT NULL, -- Timestamp of scraping
            other_metadata JSONB -- Store other metadata as JSON
        );
    """,
)    

def count_successful_inserts():
    hook = PostgresHook(postgres_conn_id='preethi_news_id')
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles WHERE scraped_timestamp = (SELECT MAX(scraped_timestamp) FROM articles);")
    count = cursor.fetchone()[0]

    # Ensure the directory exists
    status_dir = "/opt/airflow/dags/run"
    os.makedirs(status_dir, exist_ok=True)

    status_file = os.path.join(status_dir, "status")
    with open(status_file, "w") as f:
        f.write(str(count))
    cursor.close()
    conn.close()

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Defining the DAG
with DAG(
    dag_id='google_news_scraper',
    default_args=default_args,
    description='DAG for scraping Google News every hour',
    schedule_interval='0 * * * *',  # Runs at the beginning of every hour -- Crontab
    start_date=datetime(2025, 2, 16),
    catchup=False,
) as dag:

    # Load Config
    load_config_task = PythonOperator(
        task_id='load_config',
        python_callable=load_config,
    )

    # Get and Parse Articles
    get_and_parse_task = PythonOperator(
        task_id='get_and_parse_articles',
        python_callable=get_and_parse_articles,
        provide_context=True,
    )

    # Save Articles in json for reference
    save_articles_task = PythonOperator(
        task_id='save_article_data',
        python_callable=save_article_data,
        provide_context=True,
    )

    create_image_table
    create_article_table

    count_inserts_task = PythonOperator(
    task_id='count_successful_inserts',
    python_callable=count_successful_inserts,
    dag=dag
    )

    trigger_email_dag = TriggerDagRunOperator(
    task_id='trigger_send_email_dag',
    trigger_dag_id='send_email_notification',
    )

    # Set task dependencies
    load_config_task >> get_and_parse_task >> save_articles_task >> create_article_table >> create_image_table >> count_inserts_task >> trigger_email_dag
