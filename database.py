import psycopg2
from psycopg2.extras import RealDictCursor
import os
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL")

# Fix Render's postgres:// to postgresql://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def get_connection():
    """Get database connection"""
    if DATABASE_URL:
        # Production - PostgreSQL
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    else:
        # Local - You'll need to set up local PostgreSQL or use SQLite alternative
        raise Exception("DATABASE_URL not set. Set it in environment variables.")

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_db():
    """Create tables if they don't exist"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Create stock_decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_decisions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(50) NOT NULL,
                completion_code VARCHAR(50) NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                previous_open DECIMAL(10, 2) NOT NULL,
                current_price DECIMAL(10, 2) NOT NULL,
                ai_suggestion VARCHAR(10) NOT NULL,
                ai_prediction VARCHAR(10) NOT NULL,
                user_decision VARCHAR(10) NOT NULL,
                user_confidence INTEGER CHECK (user_confidence >= 1 AND user_confidence <= 10),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker 
            ON stock_decisions(ticker)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON stock_decisions(timestamp DESC)
        """)
        
        cursor.close()
        print("Database tables created/verified")


def migrate_db():
    """Add missing columns"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Add user_confidence if missing
        cursor.execute("""
            ALTER TABLE stock_decisions 
            ADD COLUMN IF NOT EXISTS user_confidence INTEGER 
            CHECK (user_confidence >= 1 AND user_confidence <= 10)
        """)
        
        # Add completion_code if missing
        cursor.execute("""
            ALTER TABLE stock_decisions 
            ADD COLUMN IF NOT EXISTS completion_code VARCHAR(50)
        """)
        
        cursor.close()
        print("Database migration complete")

