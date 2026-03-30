"""
Dataset Preparation for Fine-tuning

Creates a synthetic text-to-SQL dataset in both:
- HuggingFace format (for local fine-tuning with PEFT)
- OpenAI format (for API fine-tuning)

The dataset teaches a model to convert natural language questions
into SQL queries against a sample e-commerce database.

Usage: python prepare_dataset.py
"""

import json
import random
import os

# Schema description (included in system prompt)
SCHEMA = """Database schema:
- users (id, name, email, city, created_at)
- orders (id, user_id, product_name, amount, status, created_at)
- products (id, name, category, price, stock_count)"""

SYSTEM_PROMPT = f"You are a SQL expert. Convert the user's question into a SQL query.\n\n{SCHEMA}"

# Training examples: (question, sql) pairs
EXAMPLES = [
    ("How many users do we have?", "SELECT COUNT(*) FROM users;"),
    ("List all products", "SELECT * FROM products;"),
    ("What's the most expensive product?", "SELECT * FROM products ORDER BY price DESC LIMIT 1;"),
    ("How many orders were placed today?", "SELECT COUNT(*) FROM orders WHERE DATE(created_at) = CURRENT_DATE;"),
    ("Show me all orders over $100", "SELECT * FROM orders WHERE amount > 100;"),
    ("What is our total revenue?", "SELECT SUM(amount) FROM orders WHERE status = 'completed';"),
    ("Which city has the most users?", "SELECT city, COUNT(*) as user_count FROM users GROUP BY city ORDER BY user_count DESC LIMIT 1;"),
    ("Show me users who signed up this month", "SELECT * FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE);"),
    ("What's the average order value?", "SELECT AVG(amount) FROM orders;"),
    ("List products that are out of stock", "SELECT * FROM products WHERE stock_count = 0;"),
    ("How many orders did each user place?", "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name ORDER BY order_count DESC;"),
    ("What's our best selling product?", "SELECT product_name, COUNT(*) as order_count FROM orders GROUP BY product_name ORDER BY order_count DESC LIMIT 1;"),
    ("Show me pending orders", "SELECT * FROM orders WHERE status = 'pending';"),
    ("Who are our newest users?", "SELECT * FROM users ORDER BY created_at DESC LIMIT 10;"),
    ("What categories do we sell?", "SELECT DISTINCT category FROM products;"),
    ("How much has user 5 spent?", "SELECT SUM(amount) FROM orders WHERE user_id = 5 AND status = 'completed';"),
    ("Show revenue by month", "SELECT DATE_TRUNC('month', created_at) as month, SUM(amount) as revenue FROM orders WHERE status = 'completed' GROUP BY month ORDER BY month;"),
    ("Which products are under $50?", "SELECT * FROM products WHERE price < 50;"),
    ("Count orders by status", "SELECT status, COUNT(*) as count FROM orders GROUP BY status;"),
    ("Show me the top 5 customers by spending", "SELECT u.name, SUM(o.amount) as total_spent FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' GROUP BY u.name ORDER BY total_spent DESC LIMIT 5;"),
    ("How many products in each category?", "SELECT category, COUNT(*) as product_count FROM products GROUP BY category;"),
    ("What was our revenue last week?", "SELECT SUM(amount) FROM orders WHERE status = 'completed' AND created_at >= CURRENT_DATE - INTERVAL '7 days';"),
    ("List users from New York", "SELECT * FROM users WHERE city = 'New York';"),
    ("Show cancelled orders from January", "SELECT * FROM orders WHERE status = 'cancelled' AND EXTRACT(MONTH FROM created_at) = 1;"),
    ("What's the cheapest product in electronics?", "SELECT * FROM products WHERE category = 'Electronics' ORDER BY price ASC LIMIT 1;"),
    ("How many new users each month?", "SELECT DATE_TRUNC('month', created_at) as month, COUNT(*) as new_users FROM users GROUP BY month ORDER BY month;"),
    ("Show orders with product details", "SELECT o.id, o.amount, o.status, p.name, p.category FROM orders o JOIN products p ON o.product_name = p.name;"),
    ("What percentage of orders are completed?", "SELECT ROUND(100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*), 2) as completion_rate FROM orders;"),
    ("Find users who haven't ordered anything", "SELECT u.* FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.id IS NULL;"),
    ("Show the top 3 product categories by revenue", "SELECT p.category, SUM(o.amount) as revenue FROM orders o JOIN products p ON o.product_name = p.name WHERE o.status = 'completed' GROUP BY p.category ORDER BY revenue DESC LIMIT 3;"),
]


def create_datasets():
    """Create train/val splits in both HuggingFace and OpenAI formats."""

    # Shuffle and split: 80% train, 20% validation
    random.seed(42)
    indices = list(range(len(EXAMPLES)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    output_dir = os.path.dirname(__file__)

    # --- HuggingFace format ---
    hf_train = []
    for idx in train_indices:
        q, sql = EXAMPLES[idx]
        hf_train.append({
            "instruction": SYSTEM_PROMPT,
            "input": q,
            "output": sql,
        })

    hf_val = []
    for idx in val_indices:
        q, sql = EXAMPLES[idx]
        hf_val.append({
            "instruction": SYSTEM_PROMPT,
            "input": q,
            "output": sql,
        })

    with open(os.path.join(output_dir, "train_hf.jsonl"), "w") as f:
        for item in hf_train:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(output_dir, "val_hf.jsonl"), "w") as f:
        for item in hf_val:
            f.write(json.dumps(item) + "\n")

    # --- OpenAI format ---
    oai_train = []
    for idx in train_indices:
        q, sql = EXAMPLES[idx]
        oai_train.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": sql},
            ]
        })

    oai_val = []
    for idx in val_indices:
        q, sql = EXAMPLES[idx]
        oai_val.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": sql},
            ]
        })

    with open(os.path.join(output_dir, "train_openai.jsonl"), "w") as f:
        for item in oai_train:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(output_dir, "val_openai.jsonl"), "w") as f:
        for item in oai_val:
            f.write(json.dumps(item) + "\n")

    # Print summary
    print("Dataset created!")
    print(f"  Training examples:   {len(hf_train)}")
    print(f"  Validation examples: {len(hf_val)}")
    print(f"\n  HuggingFace format:  train_hf.jsonl, val_hf.jsonl")
    print(f"  OpenAI format:       train_openai.jsonl, val_openai.jsonl")
    print(f"\n  Sample training example:")
    print(f"    Input:  {EXAMPLES[train_indices[0]][0]}")
    print(f"    Output: {EXAMPLES[train_indices[0]][1]}")


if __name__ == "__main__":
    create_datasets()
