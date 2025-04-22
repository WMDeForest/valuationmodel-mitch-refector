#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import json
from services.chartmetric_services import chartmetric_service as chartmetric

st.title("Direct ChartMetric API Test")
st.write("Testing the direct API call equivalent to:")
st.code("curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/artist/2762/where-people-listen?latest=true")

# Make the exact same call as in the curl example
try:
    # Use the chartmetric service to handle authentication
    artist_id = 2762  # The example artist ID from the docs
    params = {'latest': True}
    
    with st.spinner("Making API call..."):
        response = chartmetric.get_artist_track_where_people_listen(
            artist_id=artist_id,
            params=params
        )
    
    # Display the raw response
    st.subheader("Raw API Response:")
    st.json(response)
    
    # Check for cities and countries
    if isinstance(response, dict) and 'obj' in response:
        obj = response['obj']
        
        if 'cities' in obj:
            cities = obj['cities']
            st.write(f"🏙️ API returned data for {len(cities)} cities")
        else:
            st.warning("No cities data in the response")
            
        if 'countries' in obj:
            countries = obj['countries']
            st.write(f"🌎 API returned data for {len(countries)} countries")
        else:
            st.warning("No countries data in the response")
    else:
        st.error("Response doesn't contain 'obj' field")
        
except Exception as e:
    st.error(f"Error making API call: {str(e)}") 