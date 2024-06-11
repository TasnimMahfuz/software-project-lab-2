import sqlite3

def create_database():
    conn = sqlite3.connect('employee_tracking.db')
    cursor = conn.cursor()

    # Create Employee Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Employee (
        id TEXT PRIMARY KEY,
        name TEXT,
        department TEXT,
        position TEXT
    )
    ''')

    # Create Activity Log Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ActivityLog (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        activity_type TEXT,
        product_count INTEGER,
        working_time REAL,
        idle_time REAL,
        FOREIGN KEY (employee_id) REFERENCES Employee (id)
    )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
