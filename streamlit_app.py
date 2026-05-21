# Streamlit UI

import streamlit as st
import os
import time
import random
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
import cairosvg
from dotenv import load_dotenv
import cv2
import numpy as np
import json
import re

# Load environment variables
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("Server Configuration Error: GEMINI_API_KEY not found in environment. Please contact the administrator.")
    st.stop()

# UI layout
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("1. Upload Assets")
    shirt_files = st.file_uploader("Upload Clothing Images (Required, JPEG/PNG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with col2:
    st.subheader("2. Ad Details")
    clothing_type = st.selectbox("Clothing Type", ["Auto-Detect", "Shirt", "Denim Shirt - Short Sleeve", "Denim Shirt - Long Sleeve", "T-Shirt", "Kurta", "Jacket", "Hoodie", "Polo"])
    logoless = st.checkbox("Logoless Output", help="If checked, any logos in the input image will be removed and no logo will be added to the output.")
    model_no = st.text_input("Model Number", value="HN-0230")
    cost = st.text_input("Price / Cost Text", value="1399/-")
    instagram = st.text_input("Instagram Handle", value="@hatchersclothingcompany")
    gemini_model = st.selectbox("AI Model", [
        "Gemini 3 Pro — Best Quality ($0.12/img)",
        "Gemini 3.1 Flash — Quality + Speed ($0.06/img)",
        "Gemini 2.5 Flash — Cheapest ($0.03/img)"], index=0)
    image_size = st.selectbox("Image Resolution", ["1K (~1024px)", "2K (~2048px)", "4K (~4096px)"], index=0)
    aspect_ratio = st.selectbox("Aspect Ratio", [
        "3:4 (Portrait - Instagram Post)",
        "9:16 (Portrait - Story/Reels)",
        "2:3 (Portrait - Pinterest)",
        "4:5 (Portrait - Instagram Feed)",
        "1:1 (Square)",
        "4:3 (Landscape)",
        "16:9 (Landscape - Wide)",
        "3:2 (Landscape - Classic)",
        "5:4 (Landscape)"], index=0)

# Model mappings
MODEL_MAP = {
    "Gemini 3 Pro — Best Quality ($0.12/img)": {
        "name": "gemini-3-pro-image-preview",
        "cost": {"1K": 0.039, "2K": 0.134, "4K": 0.24},
        "supports_2k_4k": True,
    },
    "Gemini 3.1 Flash — Quality + Speed ($0.06/img)": {
        "name": "gemini-3.1-flash-image-preview",
        "cost": {"1K": 0.02, "2K": 0.06, "4K": 0.06},
        "supports_2k_4k": True,
    },
    "Gemini 2.5 Flash — Cheapest ($0.03/img)": {
        "name": "gemini-2.5-flash-image",
        "cost": {"1K": 0.01, "2K": 0.03, "4K": 0.03},
        "supports_2k_4k": False,
    },
}

# Background templates (trimmed for brevity)
FASHION_BACKGROUNDS = [
    "in a minimalist white concrete studio with soft diffused skylight",
    "in an industrial Brooklyn loft with massive sunlit windows and exposed brick",
    # ... other backgrounds ...
]
FASHION_POSES = [
    "He is walking confidently, mid-stride, one hand in pocket, looking slightly off-camera with a relaxed expression. Natural movement, like a candid street-style photo.",
    # ... other poses ...
]

if "generated_campaigns" not in st.session_state:
    st.session_state.generated_campaigns = []

if st.button("Generate AI Advertisement", type="primary"):
    if len(shirt_files) == 0:
        st.error("Please upload at least one shirt image.")
    else:
        # Placeholder for generation logic (omitted for brevity)
        st.success("Generation logic would run here.")

# Display generated campaigns (placeholder)
if st.session_state.generated_campaigns:
    for campaign in st.session_state.generated_campaigns:
        st.markdown(f"### Campaign {campaign['index']+1}: {campaign['shirt_name']}")
        st.info(f"Location: {campaign['location']}")
        # ... display images ...
