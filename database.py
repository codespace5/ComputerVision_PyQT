import sqlite3

class Database:
    def __init__(self, db_name="farm_monitoring.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.initialize_tables()

    def initialize_tables(self):
        """
        Initializes the required tables if they don't exist.
        """
        users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            farm_name TEXT,
            address TEXT,
            user_id TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            api_key TEXT UNIQUE
        )
        """
        self.cursor.execute(users_table)
        self.conn.commit()

    def register_user(self, username, farm_name, address, user_id, password):
        """
        Registers a new user.
        """
        query = """
        INSERT INTO users (username, farm_name, address, user_id, password)
        VALUES (?, ?, ?, ?, ?)
        """
        self.cursor.execute(query, (username, farm_name, address, user_id, password))
        self.conn.commit()

    def get_password(self, user_id):
        """
        Retrieves the password for a user based on their user ID.
        """
        query = "SELECT password FROM users WHERE user_id=?"
        self.cursor.execute(query, (user_id,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def register_api_key(self, user_id, api_key):
        """
        Registers an API Key for a user.
        """
        query = "UPDATE users SET api_key=? WHERE user_id=?"
        self.cursor.execute(query, (api_key, user_id))
        self.conn.commit()

    def is_api_key_registered(self, user_id, api_key):
        """
        Checks if an API Key is already registered for a given user ID.
        """
        query = "SELECT * FROM users WHERE user_id=? AND api_key=?"
        self.cursor.execute(query, (user_id, api_key))
        result = self.cursor.fetchone()
        return result is not None

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()