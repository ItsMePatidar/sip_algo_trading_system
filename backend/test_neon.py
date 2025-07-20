import os
import pg8000.dbapi
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
print("Connecting to database with URL:", DATABASE_URL)
url = urlparse(DATABASE_URL)
conn = pg8000.dbapi.connect(
    host=url.hostname,
    port=url.port or 5432,  # Default PostgreSQL port
    database=url.path[1:],
    user=url.username,
    password=url.password,
    ssl_context=True
)

cursor = conn.cursor()
cursor.execute("SELECT version()")
result = cursor.fetchone()
print(result[0])