# Handles connection to PostgreSQL and table creation
from sqlalchemy import create_engine, Column, String, Float, Date, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()

class HistoricalPrice(Base):
    __tablename__ = 'historical_prices'
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class DatabaseManager:
    def __init__(self, db_url=None):
        # Use Neon DB connection string from environment
        print("Initializing DatabaseManager with url:", db_url)
        if db_url is None:
            db_url = DATABASE_URL
        # if db_url and db_url.startswith('postgresql://'):
        #     db_url = db_url.replace('postgresql://', 'postgresql+pg8000://')
        self.db_url = db_url
        print(f"Connecting to database at {self.db_url}")
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        return self.Session()

# Neon DB setup instructions:
# 1. Set the POSTGRES_URL environment variable to your Neon connection string.
#    Example Neon connection string:
#    postgresql://<user>:<password>@<hostname>/<dbname>
# 2. SQLAlchemy requires a driver, e.g., pg8000 or psycopg2. Your code auto-converts for pg8000.
# 3. Do NOT hardcode credentials in code. Use environment variables.
