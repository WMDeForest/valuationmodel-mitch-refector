"""
Database Query Utilities

This module provides functions for executing queries and handling results
from the database connections.
"""

import logging
import pandas as pd
from psycopg2.extras import execute_values, DictCursor

# Import from other utility modules
from utils.database.connection import get_db_connection, release_db_connection, get_sqlalchemy_engine
from utils.database.ssh_tunnel import safe_log

# Set up logging
logger = logging.getLogger('database.query')

def execute_query(query, params=None, commit=False):
    """
    Execute a SQL query with optional parameters.
    
    Args:
        query: SQL query string to execute
        params: Optional parameters for the query
        commit: Whether to commit the transaction (for INSERT/UPDATE/DELETE)
    
    Returns:
        The cursor object after query execution.
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if commit:
                connection.commit()
                
            return cursor
    except Exception as e:
        if commit:
            connection.rollback()
        safe_log('ERROR', f"Error executing query: {e}")
        raise
    finally:
        release_db_connection(connection)

def fetch_all(query, params=None, as_dict=False):
    """
    Execute a query and fetch all results.
    
    Args:
        query: SQL query string to execute
        params: Optional parameters for the query
        as_dict: If True, return results as dictionaries instead of tuples
    
    Returns:
        List of query results.
    """
    connection = get_db_connection()
    try:
        cursor_factory = DictCursor if as_dict else None
        with connection.cursor(cursor_factory=cursor_factory) as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            results = cursor.fetchall()
            
            # If we used DictCursor, convert to regular dictionaries
            if as_dict:
                results = [dict(row) for row in results]
                
            return results
    except Exception as e:
        safe_log('ERROR', f"Error fetching query results: {e}")
        raise
    finally:
        release_db_connection(connection)

def fetch_one(query, params=None, as_dict=False):
    """
    Execute a query and fetch a single result.
    
    Args:
        query: SQL query string to execute
        params: Optional parameters for the query
        as_dict: If True, return result as dictionary instead of tuple
    
    Returns:
        Single query result or None if no results.
    """
    connection = get_db_connection()
    try:
        cursor_factory = DictCursor if as_dict else None
        with connection.cursor(cursor_factory=cursor_factory) as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            result = cursor.fetchone()
            
            # If we used DictCursor, convert to regular dictionary
            if result and as_dict:
                result = dict(result)
                
            return result
    except Exception as e:
        safe_log('ERROR', f"Error fetching single query result: {e}")
        raise
    finally:
        release_db_connection(connection)

def execute_batch(query, data_list, page_size=100, commit=True):
    """
    Execute a batch operation (insert/update) with a list of data.
    
    Args:
        query: SQL query template with %s placeholders
        data_list: List of data tuples to use with the query
        page_size: Number of records to process in each batch
        commit: Whether to commit the transaction
    
    Returns:
        Number of rows affected.
    """
    if not data_list:
        return 0
        
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            execute_values(
                cursor, 
                query, 
                data_list,
                page_size=page_size
            )
            
            if commit:
                connection.commit()
                
            return cursor.rowcount
    except Exception as e:
        if commit:
            connection.rollback()
        safe_log('ERROR', f"Error executing batch operation: {e}")
        raise
    finally:
        release_db_connection(connection)

def query_to_dataframe(query, params=None):
    """
    Execute a query and return the results as a pandas DataFrame.
    
    Args:
        query: SQL query string to execute
        params: Optional parameters for the query
    
    Returns:
        pandas DataFrame with query results.
    """
    try:
        engine = get_sqlalchemy_engine()
        df = pd.read_sql(query, engine, params=params)
        return df
    except Exception as e:
        safe_log('ERROR', f"Error creating DataFrame from query: {e}")
        raise

def dataframe_to_table(df, table_name, if_exists='append', index=False):
    """
    Write a pandas DataFrame to a database table.
    
    Args:
        df: pandas DataFrame to write
        table_name: Name of the destination table
        if_exists: What to do if the table exists ('fail', 'replace', or 'append')
        index: Whether to write the DataFrame index as a column
    
    Returns:
        Number of rows written.
    """
    try:
        engine = get_sqlalchemy_engine()
        df.to_sql(
            table_name, 
            engine, 
            if_exists=if_exists,
            index=index,
            chunksize=1000  # Write in chunks to avoid memory issues
        )
        return len(df)
    except Exception as e:
        safe_log('ERROR', f"Error writing DataFrame to table: {e}")
        raise 