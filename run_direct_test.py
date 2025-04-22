#!/usr/bin/env python3
import requests
import json
import toml
import os
import time

# Load the refresh token from secrets.toml
try:
    secrets_path = os.path.join('.streamlit', 'secrets.toml')
    secrets = toml.load(secrets_path)
    REFRESH_TOKEN = secrets.get('CM_TOKEN')
    if not REFRESH_TOKEN:
        print("Error: CM_TOKEN not found in .streamlit/secrets.toml")
        exit(1)
except Exception as e:
    print(f"Error loading secrets file: {str(e)}")
    exit(1)

def get_access_token(refresh_token):
    """Get a valid JWT access token using the refresh token"""
    token_url = "https://api.chartmetric.com/api/token"
    data = {"refreshtoken": refresh_token}
    
    try:
        response = requests.post(token_url, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('token')
        else:
            print(f"Error getting access token: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception when getting access token: {str(e)}")
        return None

def test_api_call(artist_id, access_token):
    """Make a direct API call to ChartMetric using the access token"""
    if not access_token:
        print("Error: No valid access token available")
        return
        
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    url = f"https://api.chartmetric.com/api/artist/{artist_id}/where-people-listen"
    params = {"date": "2024-08-12"}
    
    print(f"\nTesting artist ID: {artist_id}")
    print(f"URL: {url}?date=2024-08-12")
    
    # Make the direct API call
    response = requests.get(url, headers=headers, params=params)
    
    # Print status code
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        
        # Check for cities data only
        if 'obj' in data:
            obj = data['obj']
            
            if 'cities' in obj and obj['cities']:
                cities = obj['cities']
                print(f"Cities: Found data for {len(cities)} cities")
                # Print all city data
                print("All cities data:")
                for city, data in cities.items():
                    print(f"  {city}: {data}")
            else:
                print("Cities: None found in response")
                
            # Save response to file for inspection
            filename = f"artist_{artist_id}_response.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Full response saved to {filename}")
                
        else:
            print("Error: Response doesn't contain 'obj' field")
    elif response.status_code == 429:
        # If rate limited, extract the retry time and wait
        retry_time = 1  # Default to 1 second
        try:
            error_text = response.text
            if "retry in" in error_text:
                # Extract the retry time in milliseconds
                import re
                match = re.search(r'retry in (\d+) ms', error_text)
                if match:
                    retry_ms = int(match.group(1))
                    retry_time = (retry_ms / 1000) + 0.1  # Convert to seconds + buffer
        except:
            pass
            
        print(f"Rate limited. Waiting {retry_time:.2f} seconds before retrying...")
        time.sleep(retry_time)
        
        # Retry the call
        print("Retrying request...")
        return test_api_call(artist_id, access_token)
    else:
        print(f"Error: API call failed: {response.text}")

if __name__ == "__main__":
    print("Starting direct API test for your artist ID...")
    print(f"Using refresh token from .streamlit/secrets.toml")
    
    # First get a valid access token
    print("Getting access token...")
    access_token = get_access_token(REFRESH_TOKEN)
    
    if not access_token:
        print("Failed to get access token. Exiting.")
        exit(1)
    
    print(f"Successfully obtained access token")
    
    # Test your artist ID
    test_api_call(4276517, access_token)
    
    print("\nTest complete!") 