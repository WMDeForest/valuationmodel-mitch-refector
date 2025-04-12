import streamlit as st
from services.chartmetric_services.chartmetric import ChartMetricService
from services.chartmetric_services.http_client import RequestsHTTPClient

# Initialize and export the ChartMetric service
chartmetric_service = ChartMetricService(
    refresh_token=st.secrets['CM_TOKEN'],
    client=RequestsHTTPClient()
)
