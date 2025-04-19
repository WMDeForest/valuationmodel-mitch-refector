#!/usr/bin/env python
# coding: utf-8

import streamlit as st

# Import tab components
from tabs.file_uploader_tab import render_file_uploader_tab
from tabs.chartmetric_tab import render_chartmetric_tab

# ===== APP INTERFACE SETUP =====
st.title('mitch_refactor_valuation_app')

# Create main navigation tabs
tab1, tab2= st.tabs(["File Uploader", "CM API Search"])

with tab1:
    # Render the File Uploader tab
    render_file_uploader_tab()

with tab2:
    # Render the ChartMetric API tab
    render_chartmetric_tab()

