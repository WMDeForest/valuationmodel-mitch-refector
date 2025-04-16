# Database Utilities

This package provides utilities for connecting to and querying databases through SSH tunnels.

## Features

- SSH tunnel management for secure database connections
- Connection pooling with automatic retries and recovery
- SQLAlchemy engine integration for pandas dataframe operations
- Helper functions for common database operations

## Modules

- `config.py`: Configuration settings for SSH and database connections
- `ssh_tunnel.py`: Functions for establishing and managing SSH tunnels
- `connection.py`: Database connection pool and SQLAlchemy engine management
- `query.py`: Functions for executing queries and handling results

## Usage Examples

### Basic Database Query

```python
from utils.database import fetch_all, close_all_connections
import atexit

# Always register the cleanup function to ensure proper shutdown
atexit.register(close_all_connections)

# Execute a query and get all results
artists = fetch_all("SELECT * FROM artists LIMIT 10")
for artist in artists:
    print(artist)
```

### Working with Pandas DataFrames

```python
from utils.database import query_to_dataframe, dataframe_to_table
import pandas as pd

# Get data as a DataFrame
df = query_to_dataframe("SELECT * FROM monthly_listeners WHERE artist_id = %s", params=[123456])

# Process the data using pandas
df['monthly_listeners'] = df['monthly_listeners'].fillna(0)
df['log_listeners'] = np.log1p(df['monthly_listeners'])

# Write processed data back to the database
dataframe_to_table(df, 'processed_listeners', if_exists='replace')
```

### Batch Operations

```python
from utils.database import execute_batch

# Prepare data for batch insert
data = [
    (1, 'Artist 1', 50000),
    (2, 'Artist 2', 75000),
    (3, 'Artist 3', 120000)
]

# Insert multiple records in one batch operation
execute_batch(
    "INSERT INTO artists (id, name, monthly_listeners) VALUES %s",
    data,
    page_size=100
)
```

### Custom Connection Configuration

```python
from utils.database import initialize_connection_pool, get_db_connection, release_db_connection

# Custom database parameters
custom_db_params = {
    'dbname': 'custom_db',
    'user': 'custom_user',
    'password': 'custom_password',
    'host': 'localhost',
    'port': '5432'
}

# Initialize with custom parameters
initialize_connection_pool(
    db_params=custom_db_params,
    min_conn=5,
    max_conn=20
)

# Use the connection
conn = get_db_connection()
try:
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM custom_table")
        results = cursor.fetchall()
        print(results)
finally:
    release_db_connection(conn)
```

## Error Handling

The utilities include built-in retry logic and error handling, but you can also implement your own error handling:

```python
from utils.database import fetch_all
import time

max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        results = fetch_all("SELECT * FROM potentially_problematic_table")
        break
    except Exception as e:
        retry_count += 1
        print(f"Error on attempt {retry_count}: {e}")
        if retry_count < max_retries:
            time.sleep(2)  # Wait before retrying
        else:
            print("Failed after multiple attempts")
```

## Best Practices

1. Always register `close_all_connections` with `atexit` to ensure proper cleanup
2. Use the appropriate function for your needs:
   - `fetch_all` for multiple results
   - `fetch_one` for a single result
   - `execute_query` for DDL statements or when you need the cursor
   - `execute_batch` for bulk operations
   - `query_to_dataframe` when working with pandas
3. Always release connections back to the pool after use
4. Use parameterized queries to prevent SQL injection 