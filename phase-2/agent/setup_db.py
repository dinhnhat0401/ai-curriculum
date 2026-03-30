"""
Create and populate the sample SQLite database for the agent module.

Tables:
    - customers: id, name, email, city, joined_date
    - orders: id, customer_id, product, amount, order_date
    - products: id, name, category, price, stock

Run: python setup_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "sample.db")


def create_database():
    """Create and populate the sample database."""

    # Remove existing database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.executescript("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            city TEXT NOT NULL,
            joined_date TEXT NOT NULL
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product TEXT NOT NULL,
            amount REAL NOT NULL,
            order_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
    """)

    # Insert customers
    customers = [
        (1, "Alice Johnson", "alice@example.com", "New York", "2023-01-15"),
        (2, "Bob Smith", "bob@example.com", "San Francisco", "2023-02-20"),
        (3, "Carol Williams", "carol@example.com", "Chicago", "2023-03-10"),
        (4, "David Brown", "david@example.com", "Austin", "2023-04-05"),
        (5, "Eva Martinez", "eva@example.com", "Seattle", "2023-05-12"),
        (6, "Frank Lee", "frank@example.com", "Boston", "2023-06-01"),
        (7, "Grace Kim", "grace@example.com", "Denver", "2023-07-20"),
        (8, "Henry Chen", "henry@example.com", "Portland", "2023-08-15"),
        (9, "Iris Patel", "iris@example.com", "Miami", "2023-09-30"),
        (10, "Jack Wilson", "jack@example.com", "New York", "2023-10-25"),
    ]
    cursor.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    # Insert products
    products = [
        (1, "DataFlow Starter", "Software", 499.00, 100),
        (2, "DataFlow Pro", "Software", 1499.00, 50),
        (3, "DataFlow Enterprise", "Software", 4999.00, 20),
        (4, "Training Package", "Service", 2000.00, 30),
        (5, "Premium Support", "Service", 500.00, 100),
        (6, "API Access Pack", "Add-on", 299.00, 200),
        (7, "Custom Connector", "Service", 3000.00, 10),
        (8, "Data Migration", "Service", 5000.00, 15),
    ]
    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    # Insert orders
    orders = [
        (1, 1, "DataFlow Pro", 1499.00, "2024-01-10"),
        (2, 1, "Premium Support", 500.00, "2024-01-10"),
        (3, 2, "DataFlow Starter", 499.00, "2024-01-15"),
        (4, 3, "DataFlow Enterprise", 4999.00, "2024-02-01"),
        (5, 3, "Training Package", 2000.00, "2024-02-01"),
        (6, 4, "DataFlow Pro", 1499.00, "2024-02-15"),
        (7, 5, "DataFlow Starter", 499.00, "2024-03-01"),
        (8, 5, "API Access Pack", 299.00, "2024-03-01"),
        (9, 6, "DataFlow Enterprise", 4999.00, "2024-03-10"),
        (10, 6, "Custom Connector", 3000.00, "2024-03-10"),
        (11, 6, "Premium Support", 500.00, "2024-03-10"),
        (12, 7, "DataFlow Pro", 1499.00, "2024-04-01"),
        (13, 8, "DataFlow Starter", 499.00, "2024-04-15"),
        (14, 9, "DataFlow Pro", 1499.00, "2024-05-01"),
        (15, 9, "Training Package", 2000.00, "2024-05-01"),
        (16, 10, "DataFlow Enterprise", 4999.00, "2024-05-15"),
        (17, 10, "Data Migration", 5000.00, "2024-05-15"),
        (18, 2, "DataFlow Pro", 1499.00, "2024-06-01"),
        (19, 4, "Premium Support", 500.00, "2024-06-15"),
        (20, 1, "Training Package", 2000.00, "2024-07-01"),
    ]
    cursor.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)

    conn.commit()

    # Print summary
    print(f"Database created: {DB_PATH}\n")
    for table in ["customers", "orders", "products"]:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    # Print schema
    print("\nSchema:")
    for table in ["customers", "orders", "products"]:
        cols = cursor.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [f"{c[1]} ({c[2]})" for c in cols]
        print(f"  {table}: {', '.join(col_names)}")

    conn.close()


if __name__ == "__main__":
    create_database()
