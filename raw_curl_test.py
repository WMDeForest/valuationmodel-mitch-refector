#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import requests
import json

st.title("Raw ChartMetric API Test")
st.write("Direct equivalent of:")
st.code("curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/artist/2762/where-people-listen?latest=true")

# Get the API token from the user
api_token = st.text_input("Enter your ChartMetric API token:", type="password")

# Test two artist IDs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Example Artist (ID: 2762)")
    if api_token and st.button("Test Example Artist"):
        try:
            # Set up the headers with authorization
            headers = {
                "Authorization": f"Bearer {api_token}"
            }
            
            # Make the direct API call equivalent to the curl example
            url = "https://api.chartmetric.com/api/artist/2762/where-people-listen"
            params = {"latest": "true"}
            
            with st.spinner("Making API call..."):
                response = requests.get(url, headers=headers, params=params)
                
            # Display response status
            st.write(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                
                # Display the raw response
                with st.expander("View Raw API Response"):
                    st.json(data)
                
                # Check for cities and countries
                if 'obj' in data:
                    obj = data['obj']
                    
                    if 'cities' in obj and obj['cities']:
                        cities = obj['cities']
                        st.write(f"🏙️ API returned data for {len(cities)} cities")
                    else:
                        st.warning("No cities data in the response")
                        
                    if 'countries' in obj and obj['countries']:
                        countries = obj['countries']
                        st.write(f"🌎 API returned data for {len(countries)} countries")
                    else:
                        st.warning("No countries data in the response")
                else:
                    st.error("Response doesn't contain 'obj' field")
            else:
                st.error(f"API call failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error making API call: {str(e)}")

with col2:
    st.subheader("Your Artist (ID: 4276517)")
    if api_token and st.button("Test Your Artist"):
        try:
            # Set up the headers with authorization
            headers = {
                "Authorization": f"Bearer {api_token}"
            }
            
            # Make the direct API call for the user's artist
            url = "https://api.chartmetric.com/api/artist/4276517/where-people-listen"
            params = {"latest": "true"}
            
            with st.spinner("Making API call..."):
                response = requests.get(url, headers=headers, params=params)
                
            # Display response status
            st.write(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                
                # Display the raw response
                with st.expander("View Raw API Response"):
                    st.json(data)
                
                # Check for cities and countries
                if 'obj' in data:
                    obj = data['obj']
                    
                    if 'cities' in obj and obj['cities']:
                        cities = obj['cities']
                        st.write(f"🏙️ API returned data for {len(cities)} cities")
                    else:
                        st.warning("No cities data in the response")
                        
                    if 'countries' in obj and obj['countries']:
                        countries = obj['countries']
                        st.write(f"🌎 API returned data for {len(countries)} countries")
                    else:
                        st.warning("No countries data in the response")
                else:
                    st.error("Response doesn't contain 'obj' field")
            else:
                st.error(f"API call failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error making API call: {str(e)}")

st.write("---")
st.write("Note: You'll need a valid ChartMetric API token. This script doesn't use any existing services or authentication methods from your codebase.") 