# Import necessary libraries
import sqlite3
import os
from pathlib import Path
import sys

# Add the parent directory to the Python path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Get the absolute path to the database file
def get_db_path():
    """Get the absolute path to the SQLite database file"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

# Initialize the database with required tables
def init_db():
    """Initialize the SQLite database with required tables if they don't exist"""
    db_path = get_db_path()
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}. Creating new database...")
    
    # Connect to the database (this will create the file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dictionary-like access to rows
    
    try:
        # Create users table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        
        # Create foods table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS foods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                calories REAL,
                protein REAL,
                carbs REAL,
                fat REAL,
                fiber REAL,
                sugars REAL,
                sodium REAL,
                cholesterol REAL
            )
        ''')
        
        # Create food_logs table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                food_name TEXT NOT NULL,
                calories REAL NOT NULL,
                protein REAL,
                carbs REAL,
                fat REAL,
                portion REAL DEFAULT 100,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Commit the changes
        conn.commit()
        print("Database initialized successfully")
        
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        raise
    finally:
        # Always close the connection
        conn.close()

# Get a database connection
def get_db():
    """Get a connection to the SQLite database"""
    db_path = get_db_path()
    
    # Ensure the database exists and is initialized
    if not os.path.exists(db_path):
        init_db()
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dictionary-like access to rows
    return conn

# Initialize the database if this script is run directly
if __name__ == '__main__':
    init_db()