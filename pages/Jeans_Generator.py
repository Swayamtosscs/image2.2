import streamlit as st
import os
import time
import base64
import random
from datetime import datetime
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
import cairosvg
from dotenv import load_dotenv
import numpy as np
import cv2
import json
import re
from auth import require_auth, render_logout_button

load_dotenv()

st.set_page_config(page_title="Jeans Ad Generator", layout="wide")

# ── JWT Auth Guard ───────────────────────────────────────────────────────────
require_auth()
render_logout_button()
# ─────────────────────────────────────────────────────────────────────────────


def _colorize_alpha(img, color):
    """Recolor visible pixels. If the image is fully opaque, treat white as transparent."""
    img = img.convert("RGBA")
    data = list(img.getdata())
    
    # Check if the image is fully opaque (likely a logo on a white background)
    is_opaque = all(p[3] == 255 for p in data[:min(len(data), 1000)])
    
    new_data = []
    if is_opaque:
        grayscale = img.convert("L")
        gs_data = list(grayscale.getdata())
        for i, pixel in enumerate(data):
            # Use darkness as alpha: black (0) -> 255, white (255) -> 0
            alpha = 255 - gs_data[i]
            new_data.append((color[0], color[1], color[2], alpha))
    else:
        for pixel in data:
            if pixel[3] > 0:
                new_data.append((color[0], color[1], color[2], pixel[3]))
            else:
                new_data.append(pixel)
                
    result = img.copy()
    result.putdata(new_data)
    return result


def create_jeans_overlay(base_img_obj, logo_pos="left"):
    """
    Creates a minimal overlay containing only the rugged_jeans logo.
    No model number, no price, no handle, no border.
    """
    base_rgba = base_img_obj.convert("RGBA")
    width, height = base_rgba.size
    app_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.abspath(os.path.join(app_dir, "..", "assets"))

    logo_path = os.path.join(assets_dir, "rugged_jeans.png")
    if not os.path.exists(logo_path):
        return base_img_obj.convert("RGB")

    try:
        logo_full = Image.open(logo_path).convert("RGBA")

        # Position logic similar to Home.py
        header_height = int(height * 0.11)  # Reduced for better aesthetics
        hsize = int(header_height * 0.85)
        wpercent = hsize / float(logo_full.size[1])
        wsize = int(float(logo_full.size[0]) * wpercent)
        
        if logo_pos == "left":
            logo_x = int(width * 0.04)
        else:
            logo_x = int(width * 0.96 - wsize)
        logo_y = int(height * 0.04)

        # Adaptive colorization
        sample_x2 = min(logo_x + wsize + int(width * 0.05), width)
        sample_y2 = min(logo_y + hsize + int(height * 0.03), height)
        bg_region = base_rgba.crop((logo_x, logo_y, sample_x2, sample_y2))
        avg_brightness = sum(bg_region.convert("L").getdata()) / max(1, len(bg_region.convert("L").getdata()))

        if avg_brightness > 128:
            logo_tinted = _colorize_alpha(logo_full, (20, 20, 20))
            shadow_color = "white"
        else:
            logo_tinted = _colorize_alpha(logo_full, (255, 255, 255))
            shadow_color = "black"

        buf = BytesIO()
        logo_tinted.save(buf, format="PNG")
        b64_logo = base64.b64encode(buf.getvalue()).decode('ascii')

        shadow_filter_def = f'''<defs>
    <filter id="logo-shadow" x="-10%" y="-10%" width="130%" height="130%">
      <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="{shadow_color}" flood-opacity="0.6"/>
    </filter>
  </defs>'''

        logo_svg_element = (
            f'<image x="{logo_x}" y="{logo_y}" width="{wsize}" height="{hsize}" '
            f'href="data:image/png;base64,{b64_logo}" filter="url(#logo-shadow)"/>'
        )

        svg_string = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  {shadow_filter_def}
  {logo_svg_element}
</svg>'''

        svg_png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'),
                                          output_width=width, output_height=height)
        overlay_img = Image.open(BytesIO(svg_png_bytes)).convert("RGBA")
        final = Image.alpha_composite(base_rgba, overlay_img)
        return final.convert("RGB")

    except Exception as e:
        st.error(f"Logo overlay failed: {e}")
        return base_img_obj.convert("RGB")

JEANS_CREATIVE_TEMPLATES = [
    {
        "name": "90s Cinematic Noir",
        "background": "in a dimly lit, moody industrial Brooklyn loft with massive windows and a view of the rainy city",
        "pose": "The model is captured mid-step, looking slightly away from the camera, embodying a cinematic 'frozen moment' in time.",
        "vibe": "Moody, high-contrast lighting with visible film grain and a 90s indie-film aesthetic. Deep shadows and cool tones."
    },
    {
        "name": "Organic Minimalism",
        "background": "in a clean, sun-drenched architectural studio with raw concrete floors and soft white fabric drapes",
        "pose": "Minimalist standing pose, leaning slightly back against a soft white surface, hands relaxed at sides to show the clean silhouette of the jeans.",
        "vibe": "Clean, bright, organic minimalism. Single soft light source, emphasizing the texture and hand-feel of the denim. Light and airy feel."
    },
    {
        "name": "Gritty Urban Motion",
        "background": "in an abandoned architectural space with raw brick, weathered steel beams, and sunlight filtering through gaps",
        "pose": "A dynamic motion stillness shot—the model is caught walking confidently across the frame, legs in a wide stride to show movement and fit.",
        "vibe": "Gritty, textural, and authentic. Warm high-noon sun creating sharp geometric shadows on the concrete floor."
    },
    {
        "name": "Mediterranean Golden Hour",
        "background": "on a sun-bleached stone terrace in a coastal town, with a weathered stone wall behind and a hint of the ocean",
        "pose": "A relaxed, effortless pose—leaning one forearm on a stone ledge, weight on one hip, embodying a sophisticated 'quiet luxury' lifestyle.",
        "vibe": "Warm golden hour glow, sepia-tinted highlights, and a soft, nostalgic feeling. Rich, saturated colors."
    },
    {
        "name": "Modern Studio Pop",
        "background": "against a bold, high-contrast solid cream background in a minimalist photo studio",
        "pose": "A direct, powerful low-angle standing pose. Arms relaxed, looking straight into the lens with confidence.",
        "vibe": "Commercial luxury style. Crisp, sharp focus, perfectly balanced lighting with no distracting elements, making the denim colors pop."
    },
    {
        "name": "Texas Ranch Authentic",
        "background": "leaning against a weathered wooden fence on a sun-soaked ranch under the intense Texas high-noon sun",
        "pose": "A rugged, stoic pose—resting one hand on a belt buckle, weight on one leg, looking out over the horizon.",
        "vibe": "Authentic cowboy editorial vibe. Intense golden flares, sun-soaked color grade, and photorealistic raw textures."
    }
]

def generate_jeans_ad(api_key, jeans_bytes, image_size="1K", aspect_ratio="3:4"):
    """
    Generates a jeans advertisement using a randomized creative template.
    """
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return None, f"Error initializing client: {e}"

    app_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.abspath(os.path.join(app_dir, "..", "assets"))
    
    # Randomly pick a style reference
    ref_choice = random.choice(["jeans_reference1", "jeans_reference2", "jeans_reference3"])
    reference_path = os.path.join(assets_dir, f"{ref_choice}.jpeg")
    
    # Randomly pick a creative template
    template = random.choice(JEANS_CREATIVE_TEMPLATES)
    
    reference_bytes = None
    if os.path.exists(reference_path):
        with open(reference_path, "rb") as f:
            reference_bytes = f.read()

    clothing_detail = (
        "The model MUST wear this EXACT pair of jeans. Study the input image carefully. "
        "You MUST reproduce the jeans EXACTLY as they appear in the photo. "
        "STRICTLY NO LOGOS OR EMBROIDERY. The garment MUST be 100% logo-free. "
        "If the input image contains a logo, REMOVE and ERASE it entirely. "
        "The fabric must be perfectly clean and unbranded."
    )

    prompt = (
        f"I am providing TWO images.\n\n"
        f"IMAGE 1 (FIRST image): STYLE REFERENCE — study its photographic style, lighting, "
        f"camera angle, framing, composition, and editorial quality.\n\n"
        f"IMAGE 2 (SECOND image): CLOTHING ITEM (JEANS). {clothing_detail}\n\n"
        f"High-end luxury fashion photography, GQ editorial style.\n\n"
        f"STRICT RULES FOR JEANS ADVERTISEMENT:\n"
        f"- The JEANS are the main product. The jeans MUST be fully visible and fill the majority of the frame.\n"
        f"- The shot MUST be taken from a strict extreme low angle to make the legs look powerful and to showcase the jeans prominently.\n"
        f"- The focus is from the waist down to the shoes. Do NOT frame from mid-thigh up; include the ENTIRE pair of jeans.\n\n"
        f"CREATIVE THEME: {template['name']}\n"
        f"SETTING: {template['background']}.\n"
        f"POSE: {template['pose']}\n"
        f"VIBE & LIGHTING: {template['vibe']}\n\n"
        f"COMPOSITION: Position the model centered. The TOP-LEFT QUADRANT (upper 25% height, left 45% width) MUST be 100% CLEAN background only. "
        f"Absolutely no part of the model's body, face, head, hair, or clothing may enter this zone. Keep this area as minimal negative space.\n\n"
        f"TECHNICAL: Sharp focus, {aspect_ratio} aspect ratio.\n"
        f"CRITICAL: Output must contain ZERO text, logos, or watermarks. Clean photo only."
    )

    contents = []
    if reference_bytes:
        contents.append(types.Part.from_bytes(data=reference_bytes, mime_type='image/jpeg'))
    contents.append(types.Part.from_bytes(data=jeans_bytes, mime_type='image/jpeg'))
    contents.append(prompt)

    system_prompt = (
        "You are a professional fashion photographer specialized in denim. "
        "Reproduce the jeans from the input photo EXACTLY. "
        "The focus must be on the jeans, using an extreme low angle to make the product look iconic. "
        "Absolutely no logos or text on the clothing."
    )

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(image_size=image_size, aspect_ratio=aspect_ratio),
                system_instruction=system_prompt,
            )
        )
        
        if not response.candidates:
            return None, "Gemini returned no candidates."

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return Image.open(BytesIO(part.inline_data.data)), "Success"
        
        return None, "No image data returned."
    except Exception as e:
        return None, f"Generation failed: {e}"

# --- UI ---

st.title("👖 Rugged Jeans Creative Studio")
st.markdown("Automated high-impact jeans advertisements. Just upload your jeans and hit generate—the algorithm will handle the creative direction.")

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("API Key missing.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Jeans")
    jeans_file = st.file_uploader("Upload Jeans Image", type=["png", "jpg", "jpeg"])

with col2:
    st.subheader("2. Generation Settings")
    image_size = st.radio("Output Resolution", ["1K", "2K", "4K"], horizontal=True)
    aspect_ratio = st.selectbox("Aspect Ratio", ["3:4", "9:16", "1:1"])

if st.button("Generate Randomized Jeans Ad", type="primary"):
    if not jeans_file:
        st.error("Please upload a jeans photo.")
    else:
        with st.spinner("Algorithm picking creative direction..."):
            jeans_bytes = jeans_file.read()
            raw_img, msg = generate_jeans_ad(api_key, jeans_bytes, image_size=image_size, aspect_ratio=aspect_ratio)
            
            if raw_img:
                logo_side = "left"
                final_img = create_jeans_overlay(raw_img, logo_pos=logo_side)
                
                st.success("Generation Complete!")
                res1, res2 = st.columns(2)
                with res1:
                    st.markdown("**AI Generated**")
                    st.image(raw_img, width="stretch")
                with res2:
                    st.markdown("**Final Ad (Rugged Jeans Logo)**")
                    st.image(final_img, width="stretch")
                    
                    buf = BytesIO()
                    final_img.save(buf, format="JPEG")
                    st.download_button("Download Ad", buf.getvalue(), "jeans_ad.jpg", "image/jpeg")
            else:
                st.error(msg)
            
            if raw_img:
                logo_side = "left"
                final_img = create_jeans_overlay(raw_img, logo_pos=logo_side)
                
                st.success("Generation Complete!")
                res1, res2 = st.columns(2)
                with res1:
                    st.markdown("**AI Generated**")
                    st.image(raw_img, width="stretch")
                with res2:
                    st.markdown("**Final Ad (Rugged Jeans Logo)**")
                    st.image(final_img, width="stretch")
                    
                    buf = BytesIO()
                    final_img.save(buf, format="JPEG")
                    st.download_button("Download Ad", buf.getvalue(), "jeans_ad.jpg", "image/jpeg")
            else:
                st.error(msg)
