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

from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np

import json
import re

def get_source_logo_bbox(api_key, image_bytes):
    """
    Step A: Ask Gemini 1.5 Pro to find the emblem/logo on the original shirt.
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = (
            "Find the primary logo, brand mark, or embroidery on this shirt. "
            "Return ONLY its bounding box in the standard normalized format [ymin, xmin, ymax, xmax] scaled 0 to 1000. "
            "If there is NO logo or embroidery visible, return strictly the word 'NONE'."
        )
        
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[img, prompt],
        )
        
        text = response.text.strip()
        print(f"Step A Response: {text}")
        if "NONE" in text.upper():
            return None
            
        match = re.search(r'\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]', text)
        if match:
            return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
        else:
            print("Failed to parse coordinates from Step A.")
            return None
    except Exception as e:
        print(f"Error in Step A Vision API: {e}")
        return None

def get_target_logo_center(api_key, image_pil, source_bbox):
    """
    Step B: Ask Gemini 1.5 Pro where the exact pixel should be on the generated image.
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = (
            f"This is an AI-generated person wearing a plain shirt. A logo was originally positioned at coordinates {source_bbox} on the original shirt. "
            "Analyze the person's pose and torso, and return the bounding box of where the logo SHOULD sit on this new shirt. "
            "Return ONLY the standard normalized bounding box [ymin, xmin, ymax, xmax] scaled 0 to 1000. Look for landmarks like pockets or standard chest hit areas."
        )
        
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[image_pil, prompt],
        )
        
        text = response.text.strip()
        print(f"Step B Response: {text}")
        
        match = re.search(r'\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]', text)
        if match:
            ymin, xmin, ymax, xmax = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
            x_center_1000 = (xmin + xmax) / 2.0
            y_center_1000 = (ymin + ymax) / 2.0
            
            pixel_x = int((x_center_1000 / 1000.0) * image_pil.width)
            pixel_y = int((y_center_1000 / 1000.0) * image_pil.height)
            return (pixel_x, pixel_y)
        else:
            print("Failed to parse coordinates from Step B.")
            return None
    except Exception as e:
        print(f"Error in Step B Vision API: {e}")
        return None

def preprocess_logo(logo_path):
    """
    Pre-Processing step intended to clean dirty assets: 
    strips white backgrounds, solidifies dotted shapes, and thickens the emblem. 
    Returns a clean solid-alpha RGBA PIL Image.
    """
    img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
        
    # If no alpha channel, add one
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
    # 1. Automated Transparency (Strip White > 230)
    b, g, r, a = cv2.split(img)
    white_mask = (r > 230) & (g > 230) & (b > 230)
    a[white_mask] = 0
    
    # 2. Logo Solidification (Fill the Gaps via Contours)
    contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_alpha = np.zeros_like(a)
    cv2.drawContours(solid_alpha, contours, -1, 255, thickness=cv2.FILLED)
    
    # 3. Alpha Dilation (Thicken Lines)
    kernel_dilate = np.ones((3, 3), np.uint8)
    solid_alpha = cv2.dilate(solid_alpha, kernel_dilate, iterations=1)
    
    # Apply processed alpha back
    img[:, :, 3] = solid_alpha
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

def apply_stitched_logo(base_img_pil, logo_img_pil, position, scale_percent):
    """
    Luminance-aware blending of a logo onto the base image using OpenCV.
    """
    base_img_cv = cv2.cvtColor(np.array(base_img_pil), cv2.COLOR_RGB2BGR)
    logo_img_cv = cv2.cvtColor(np.array(logo_img_pil.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
    
    # 1. Solidify Emblem (Flood Fill Contours)
    logo_alpha_orig = logo_img_cv[:, :, 3]
    contours, _ = cv2.findContours(logo_alpha_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_alpha = np.zeros_like(logo_alpha_orig)
    cv2.drawContours(solid_alpha, contours, -1, 255, thickness=cv2.FILLED)
    
    # 2. Thicken Lines
    kernel_dilate = np.ones((3, 3), np.uint8)
    solid_alpha = cv2.dilate(solid_alpha, kernel_dilate, iterations=1)
    logo_img_cv[:, :, 3] = solid_alpha
    
    h_base, w_base, _ = base_img_cv.shape
    new_w = int(w_base * (scale_percent / 100.0))
    
    aspect = logo_img_cv.shape[0] / float(max(1, logo_img_cv.shape[1]))
    new_h = int(new_w * aspect)
    
    if new_w <= 0 or new_h <= 0:
        return base_img_pil
        
    logo_resized = cv2.resize(logo_img_cv, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    x, y = int(position[0]), int(position[1])
    x_tl = int(x - new_w / 2.0)
    y_tl = int(y - new_h / 2.0)
    
    # Check boundaries
    if x_tl + new_w > w_base or y_tl + new_h > h_base or x_tl < 0 or y_tl < 0:
        return base_img_pil
        
    roi = base_img_cv[y_tl:y_tl+new_h, x_tl:x_tl+new_w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 3. High-Contrast Enforcement
    avg_brightness = np.mean(roi_gray)
    if avg_brightness >= 127:
        logo_color = [30, 30, 30]  # Dark Grey/Black for Light Shirts
    else:
        logo_color = [225, 225, 225] # Light Grey/White for Dark Shirts
        
    logo_resized[:, :, 0] = logo_color[0]
    logo_resized[:, :, 1] = logo_color[1]
    logo_resized[:, :, 2] = logo_color[2]
    
    # Lighting map: normalize 0-1
    luminance = roi_gray / 255.0
    
    # Sharpen the alpha channel to combat downscaling blur
    alpha_channel = logo_resized[:, :, 3].astype(float)
    kernel_sharpen = np.array([[0, -1, 0], 
                               [-1, 5, -1], 
                               [0, -1, 0]])
    alpha_channel = cv2.filter2D(alpha_channel, -1, kernel_sharpen)
    
    # Alpha Thresholding
    alpha_channel = np.where(alpha_channel > 127.5, 255.0, 0.0)
    alpha = alpha_channel / 255.0
    alpha3 = np.expand_dims(alpha, axis=2)
    
    # Soft Light / Overlay luminance blend
    logo_bgr = logo_resized[:, :, :3].astype(float)
    luminance3 = np.expand_dims(luminance, axis=2)
    
    logo_shaded = np.where(
        luminance3 < 0.5, 
        2 * logo_bgr * luminance3, 
        255.0 - 2 * (255.0 - logo_bgr) * (1.0 - luminance3)
    )
    logo_shaded = np.clip(logo_shaded, 0, 255)
    
    roi_float = roi.astype(float)
    roi_blended = roi_float * (1.0 - alpha3) + logo_shaded * alpha3
    
    base_img_cv[y_tl:y_tl+new_h, x_tl:x_tl+new_w] = np.clip(roi_blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(cv2.cvtColor(base_img_cv, cv2.COLOR_BGR2RGB))



st.set_page_config(page_title="AI Fashion Ad Generator", layout="wide")

def _colorize_alpha(img, color):
    """Recolor all visible (non-transparent) pixels to the given color, preserving alpha."""
    data = list(img.getdata())
    new_data = []
    for pixel in data:
        if pixel[3] > 0:
            new_data.append((color[0], color[1], color[2], pixel[3]))
        else:
            new_data.append(pixel)
    result = img.copy()
    result.putdata(new_data)
    return result

def create_svg_overlay(base_img_obj, model_no, cost, instagram, logo_pos="left"):
    """
    Creates a high-quality SVG overlay frame and composites it onto the base image.
    SVG elements: white border, top-right price badge, bottom-left IG handle, top-left logo.
    The emblem is no longer overlaid — it's embedded in the AI-generated scene by Gemini.
    """
    base_rgba = base_img_obj.convert("RGBA")
    width, height = base_rgba.size
    app_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Dimensions (percentage-based, matching original layout) ---
    bw = int(width * 0.02)  # border width

    # --- Price badge geometry ---
    price_text = f"{model_no}  MRP \u20b9{cost}"
    font_size_price = int(width * 0.030)
    # Estimate text width for badge sizing
    text_w_est = font_size_price * 0.62 * len(price_text)
    text_h_est = font_size_price * 1.3
    badge_h = int(text_h_est + height * 0.04)
    badge_w = int(text_w_est + width * 0.08)
    angle_off = int(badge_w * 0.15)

    if logo_pos == "left":
        # Standard right-side price badge
        price_points = (f"{width - badge_w},0 {width},0 "
                        f"{width},{badge_h} "
                        f"{width - badge_w + angle_off},{badge_h}")
        price_center_x = width - badge_w + angle_off + (badge_w - angle_off) / 2
    else:
        # Swap: Price badge on the left
        price_points = (f"0,0 {badge_w},0 "
                        f"{badge_w - angle_off},{badge_h} "
                        f"0,{badge_h}")
        price_center_x = (badge_w - angle_off) / 2

    price_center_y = badge_h / 2

    # --- Bottom-left Instagram badge geometry ---
    ig_handle = instagram.lstrip('@')
    ig_text = f"@{ig_handle}"
    font_size_ig = int(width * 0.032)  # bigger IG text (was 0.022)
    ig_text_w_est = font_size_ig * 0.58 * len(ig_text)
    ig_text_h_est = font_size_ig * 1.3
    side_badge_w = int(ig_text_h_est + width * 0.05)  # wider badge
    side_badge_h = int(ig_text_w_est + height * 0.10)  # taller badge
    angle_off_2 = int(side_badge_h * 0.1)

    bl_points = (f"0,{height - side_badge_h - angle_off_2} "
                 f"{side_badge_w},{height - side_badge_h} "
                 f"{side_badge_w},{height} "
                 f"0,{height}")

    # Center of the IG badge for rotated text
    ig_cx = side_badge_w / 2
    ig_cy = height - side_badge_h / 2

    # --- Instagram icon embedding (render insta.svg → base64 PNG) ---
    ig_icon_element = ""
    ig_icon_size = int(side_badge_w * 0.55)
    ig_icon_x = int(ig_cx - ig_icon_size / 2)
    ig_icon_y = int(height - ig_icon_size - side_badge_w * 0.20)
    try:
        insta_svg_path = os.path.join(app_dir, "assets", "insta.svg")
        if os.path.exists(insta_svg_path):
            insta_png_bytes = cairosvg.svg2png(
                url=insta_svg_path,
                output_width=ig_icon_size * 2,
                output_height=ig_icon_size * 2
            )
            insta_img = Image.open(BytesIO(insta_png_bytes)).convert("RGBA")
            insta_tinted = _colorize_alpha(insta_img, (51, 51, 51))
            buf = BytesIO()
            insta_tinted.save(buf, format="PNG")
            b64_insta = base64.b64encode(buf.getvalue()).decode('ascii')
            ig_icon_element = (
                f'<image x="{ig_icon_x}" y="{ig_icon_y}" width="{ig_icon_size}" height="{ig_icon_size}" '
                f'href="data:image/png;base64,{b64_insta}"/>'
            )
    except Exception as e:
        print(f"Warning: Could not add Instagram icon: {e}")

    # --- Logo embedding (adaptive colorization + base64) ---
    logo_svg_element = ""
    shadow_filter_def = ""
    try:
        logo_full = Image.open(os.path.join(app_dir, "assets", "logo.png")).convert("RGBA")

        # Calculate display size (where it will appear on the final image)
        header_height = int(height * 0.11)  # Reduced from 0.15 for better aesthetics
        header_width = width - badge_w
        hsize = int(header_height * 0.85)
        wpercent = hsize / float(logo_full.size[1])
        wsize = int(float(logo_full.size[0]) * wpercent)
        if wsize > header_width * 0.85:
            wsize = int(header_width * 0.85)
            hsize = int(float(logo_full.size[1]) * (wsize / float(logo_full.size[0])))

        if logo_pos == "left":
            logo_x = int(width * 0.04)
        else:
            logo_x = int(width * 0.96 - wsize)
        logo_y = int(height * 0.04)

        # Adaptive colorization — sample a larger region around the logo for better detection
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

        # Encode FULL resolution logo as base64 — SVG handles the scaling
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
    except Exception as e:
        print(f"Warning: Could not add logo to SVG: {e}")

    # --- Build SVG ---
    # Escape special XML characters in text
    price_text_escaped = price_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    ig_text_escaped = ig_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    svg_string = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">

  {shadow_filter_def}

  <!-- White border frame -->
  <rect x="{bw // 2}" y="{bw // 2}" width="{width - bw}" height="{height - bw}"
        fill="none" stroke="white" stroke-width="{bw}"/>

  <!-- Top price badge -->
  <polygon points="{price_points}" fill="white"/>
  <text x="{price_center_x}" y="{price_center_y}"
        font-family="Noto Sans, Liberation Sans, Arial, Helvetica, sans-serif" font-weight="bold"
        font-size="{font_size_price}" fill="#333333"
        text-anchor="middle" dominant-baseline="central">{price_text_escaped}</text>

  <!-- Bottom-left Instagram badge -->
  <polygon points="{bl_points}" fill="white"/>
  <!-- Instagram icon (from insta.svg) -->
  {ig_icon_element}
  <text transform="rotate(-90, {ig_cx}, {ig_cy})"
        x="{ig_cx}" y="{ig_cy}"
        font-family="Noto Sans, Liberation Sans, Arial, Helvetica, sans-serif" font-weight="bold"
        font-size="{font_size_ig}" fill="#333333"
        text-anchor="middle" dominant-baseline="central">{ig_text_escaped}</text>

  <!-- Brand logo -->
  {logo_svg_element}

</svg>'''

    # Render SVG to PNG and composite
    try:
        svg_png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'),
                                          output_width=width, output_height=height)
        overlay_img = Image.open(BytesIO(svg_png_bytes)).convert("RGBA")
        final = Image.alpha_composite(base_rgba, overlay_img)
        return final.convert("RGB")
    except Exception as e:
        print(f"Warning: SVG overlay rendering failed: {e}. Returning base image.")
        return base_img_obj.convert("RGB")

def generate_ad(api_key, shirt_bytes, background, pose, image_size="1K", aspect_ratio="3:4", clothing_type="Auto-Detect", model_name="gemini-3-pro-image-preview", logoless=False):
    """
    Sends clothing image + style reference + emblem to Gemini.
    Returns the RAW AI-generated image (with emblem embedded in scene) — no overlays.
    """
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return None, f"Error initializing client: {e}"

    app_dir = os.path.dirname(os.path.abspath(__file__))

    # Load reference image from disk if available
    reference_bytes = None
    reference_path = os.path.join(app_dir, "reference.jpeg")
    if os.path.exists(reference_path):
        with open(reference_path, "rb") as f:
            reference_bytes = f.read()

    # We no longer send the emblem to Gemini.
    # The logo overlay is applied computationally using OpenCV in the main loop.
    emblem_bytes = None

    # Determine clothing description
    if clothing_type == "Auto-Detect":
        clothing_desc = "Analyze the clothing item and determine its type (shirt, t-shirt, kurta, jacket, hoodie, polo, etc.)."
    elif clothing_type == "Denim Shirt - Short Sleeve":
        clothing_desc = "This is a denim short-sleeve shirt with a rugged, durable fabric texture."
    elif clothing_type == "Denim Shirt - Long Sleeve":
        clothing_desc = "This is a denim long-sleeve shirt with a rugged, durable fabric texture. The sleeves should be folded slightly below the elbow."
    else:
        clothing_desc = f"This is a {clothing_type}."

    # Denim theme addition
    denim_theme = ""
    if "Denim Shirt" in clothing_type:
        denim_theme = (
            "THEME: True sunny Texas ranch vibe with an authentic cowboy feel. "
            "COLOR PALETTE: STRICT warm golden and yellowish tones. High-noon sun with intense golden flares, "
            "sun-soaked yellowish color grade, and sepia-tinted highlights. "
            "AESTHETIC: 90s-early 2000s cowboy editorial style (GQ/Vogue). "
            "MODEL ACTION: The model must embody a rugged cowboy persona — confident, weathered, and stoic. "
            "Poses should feel authentic to a ranch: leaning against a weathered wooden fence, resting a hand on a belt buckle, or looking out over the horizon. "
            "LIGHTING & BRIGHTNESS: The model and the denim shirt MUST be VERY brightly lit and vibrantly illuminated by the intense Texas sun. "
            "Ensure the shirt 'pops' with high clarity and brightness, standing out distinctly from the warm background. Avoid heavy shadows on the garment.\n\n"
        )

    # Emblem instruction block (reused across prompt variants)
    emblem_instruction = (
        f"BRAND EMBLEM/SYMBOL. This image shows the exact emblem that MUST appear in the "
        f"background of the generated scene.\n"
        f"The emblem MUST appear — this is NON-NEGOTIABLE.\n\n"
        f"EMBLEM RULES:\n"
        f"- SHAPE: The emblem is shaped like the letter 'H' — two wide vertical pillars "
        f"connected by a narrower center. The left and right sides have CONCAVE CURVED edges "
        f"(they curve INWARD toward the center like an hourglass). In the exact center there "
        f"is a small 4-pointed diamond star. Study the provided image and reproduce THIS "
        f"specific shape. Do NOT make a snowflake, cross, generic star, or any other shape.\n"
        f"- SIZE: HUGE — covering 30-50% of the background. It dominates the scene.\n"
        f"- STYLE: Can be rendered as ANY of these — choose what fits the scene best:\n"
        f"  * 3D relief carved into a wall or surface (deep grooves, beveled edges)\n"
        f"  * A large freestanding sculpture or monument (stone, metal, concrete, wood)\n"
        f"  * An architectural feature or decorative element in the environment\n"
        f"- MATERIAL: Match the scene's environment — same stone, wood, metal, concrete, "
        f"ice, or whatever material makes sense for the setting.\n"
        f"- PLACEMENT: CENTER-RIGHT or RIGHT SIDE of background, clearly visible behind or "
        f"beside the model.\n"
        f"- QUALITY: Photorealistic, physically real, part of the world.\n\n"
    )

    # Ensure Gemini does not hallucinate logos because we stitch our own logo via OpenCV layer
    if logoless:
        logo_instruction = (
            f"CRITICAL RULES FOR LOGO REMOVAL:\n"
            f"- THE GARMENT MUST BE 100% LOGO-FREE. You MUST REMOVE and ERASE any logos, emblems, brand marks, or embroidery present in the input image. "
            f"Generate only plain, clean fabric in their place. This is MANDATORY.\n"
            f"- DO NOT generate any new logos or symbols on the clothing. The chest area must be completely unbranded.\n"
        )
    else:
        logo_instruction = (
            f"CRITICAL RULES FOR CHEST AREA:\n"
            f"- YOU MUST LEAVE THE CHEST COMPLETELY PLAIN AND CLEAN. Specifically, NO emblem, logo, or embroidery should be present on the chest. Even if the original clothing has one, you MUST ERASE it.\n"
            f"- DO NOT GENERATE OR ADD ANY LOGOS, EMBLEMS, EMBROIDERY, OR TEXT TO THE CLOTHING.\n"
            f"- The shirt must have NO logos, NO text, and NO embroidery. It must be a 100% plain, unbranded fabric surface in the chest area.\n"
        )

    # Clothing detail instruction — reproduce ONLY what exists in the original image
    clothing_detail = (
        f"The model MUST wear this EXACT item. {denim_theme} Study the clothing image carefully — "
        f"reproduce the clothing EXACTLY as it appears in the photo. ONLY include details "
        f"that are VISIBLE in the original clothing image. Do NOT add, invent, or place "
        f"any logos, symbols, embroidery, or designs that are NOT in the original photo.\n\n"
        f"{logo_instruction}\n"
        f"Preserve ALL other details exactly: pockets, buttons, stitching, collar shape, "
        f"cuff style, fabric texture, color, and pattern exactly as photographed."
    )

    # Define the prompt — image count depends on what's available
    if reference_bytes:
        prompt = (
            f"I am providing TWO images.\n\n"
            f"IMAGE 1 (FIRST image): STYLE REFERENCE — study its photographic style, lighting, "
            f"camera angle, framing, composition, and editorial quality. "
            f"DO NOT copy any text, logos, branding, or typography from it.\n\n"
            f"IMAGE 2 (SECOND image): CLOTHING ITEM. {clothing_desc} "
            f"{clothing_detail}\n\n"
        )
    else:
        prompt = (
            f"High-end luxury fashion photography, GQ magazine editorial style. "
            f"The provided image is a CLOTHING ITEM. {clothing_desc} "
            f"Generate a handsome, confident male fashion model wearing this exact item. "
            f"{clothing_detail}\n\n"
        )

    prompt += (
        f"STYLING RULES:\n"
        f"- This is a CASUAL brand — the overall look must feel relaxed, stylish, and "
        f"approachable. NOT formal, NOT corporate, NOT stiff.\n"
        f"- Pair with jeans, chinos, or casual bottoms.\n"
        f"- Kurta: with traditional bottoms, sleeves rolled above wrist.\n"
        f"- Pants MUST contrast — NEVER same fabric/pattern as top.\n\n"
        f"POSE: {pose}\n"
        f"POSE RULES:\n"
        f"- The FRONT of the shirt/clothing MUST be clearly visible — this is the PRODUCT.\n"
        f"- NO crossed arms, folded arms, or hands covering the chest/torso area.\n"
        f"- The model can look slightly off-camera, at an angle, or directly at camera — "
        f"keep it natural and editorial, NOT stiff or posed like a mannequin.\n"
        f"- Slight body angles (3/4 turn) are fine as long as the shirt front is visible.\n"
        f"- Arms can be at sides, in pockets, touching face/hair, or resting on nearby surfaces.\n\n"
        f"SETTING: {background}. IMPORTANT: The right side of the background MUST have a "
        f"large clean surface (wall, rock, building, open sky) where the brand emblem can "
        f"be placed. Do NOT fill the entire background with busy patterns or decorations.\n\n"
        f"MODEL & FRAMING:\n"
        f"- VERY IMPORTANT: Leave massive headroom (at least 25-30% empty space) at the VERY TOP of the image.\n"
        f"- The model's head MUST NOT touch or get close to the top edge.\n"
        f"- Frame from the waist or mid-thigh up, scaling the model so they occupy only the bottom 60-70% of the image to allow for the top headroom.\n"
        f"- The model should look like a regular, good-looking guy — NOT a runway supermodel. "
        f"Natural, approachable, casual vibe. Think weekend hangout, not fashion show.\n"
        f"- Relaxed expression — slight smile, calm eyes, easy-going.\n"
        f"- Natural lighting on face and clothing, NOT dramatic studio lighting.\n\n"
        f"COMPOSITION ZONES (follow exactly):\n"
        f"- Position the model CENTER to CENTER-RIGHT of the frame, positioned lower in the frame.\n"
        f"- ENTIRE TOP 30% OF THE IMAGE: This entire top area MUST be 100% CLEAN "
        f"background only (e.g., plain wall, soft sky). ABSOLUTELY NO part of the model's body, face, head, hair, or clothing may "
        f"enter this top zone. The model's head must be positioned well BELOW this 30% mark to create negative space for branding.\n"
        f"- TOP-RIGHT CORNER: MUST be clean, uncluttered background space.\n"
        f"- BOTTOM-LEFT EDGE: MUST be clean, minimalist background space.\n"
    )

    if emblem_bytes:
        prompt += (
            f"- CENTER-RIGHT / BACKGROUND: The brand emblem from IMAGE 3 MUST appear here. "
            f"It can be carved into a wall, rendered as a freestanding sculpture, or integrated "
            f"as an architectural feature — whatever fits the scene. It must be LARGE and clearly "
            f"visible. This is MANDATORY for every image.\n\n"
        )
    else:
        prompt += f"- BOTTOM-RIGHT CORNER: Keep relatively clean.\n\n"

    prompt += (
        f"TECHNICAL: Bright, fresh, well-lit photography with vibrant colors and a light, "
        f"airy, refreshing feel. Use natural daylight, golden hour warmth, or bright studio "
        f"lighting. AVOID dark, moody, underexposed, or heavily shadowed images. "
        f"Sharp focus, {aspect_ratio} aspect ratio.\n\n"
    )

    if emblem_bytes:
        prompt += (
            f"CRITICAL: Output must contain ZERO text, logos, watermarks, or typography. "
            f"The only graphic allowed is the brand emblem symbol from IMAGE 3, rendered as "
            f"part of the scene environment — not as text or a pasted logo. Clean photo only."
        )
    else:
        prompt += (
            f"CRITICAL: Output must contain ZERO text, logos, watermarks, or typography. Clean photo only."
        )

    # Build contents list — order must match IMAGE 1/2/3 in the prompt
    # Send emblem TWICE: once in sequence (referenced by prompt) and once after prompt
    # to reinforce its shape at the end where Gemini pays most attention (recency bias)
    contents = []
    if reference_bytes:
        contents.append(types.Part.from_bytes(data=reference_bytes, mime_type='image/jpeg'))
    contents.append(types.Part.from_bytes(data=shirt_bytes, mime_type='image/jpeg'))
    if emblem_bytes:
        contents.append(types.Part.from_bytes(data=emblem_bytes, mime_type='image/png'))
    contents.append(prompt)

    # System instruction for faithful reproduction
    system_prompt = (
        "You are a casual lifestyle brand photographer. The vibe is relaxed, approachable, "
        "and natural — NOT formal or high-fashion. Reproduce the clothing item from the "
        "reference photo EXACTLY as shown — same fabric, color, pattern, pockets, buttons, "
        "and all details. Do not alter or reinterpret any part of the clothing.\n"
        f"MANDATORY: { 'STRICTLY NO LOGOS OR EMBROIDERY ANYWHERE. Ensure the garment is completely plain and unbranded.' if logoless else 'NO LOGOS OR EMBROIDERY ON CHEST. Even if the input image shows a logo or branding, OMIT IT entirely. The chest must be a perfectly clean, plain, and unadorned fabric canvas.'}"
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(image_size=image_size, aspect_ratio=aspect_ratio),
                system_instruction=system_prompt,
            )
        )
    except Exception as e:
        return None, f"Error generating image with Gemini: {e}"

    if not response.candidates:
        return None, "Error: Gemini returned no candidates (possible content policy block)."

    generated_img = None
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            generated_img = Image.open(BytesIO(part.inline_data.data))
            break

    if not generated_img:
        return None, "Error: No image data was returned by Gemini 3."

    # Return raw AI-generated image (emblem is embedded in scene by Gemini)
    # Overlay frame is applied separately in the UI loop
    return generated_img, "Success"

# --- STREAMLIT UI ---

st.title("AI Fashion Ad Studio")
st.markdown("Instantly generate high-end, magazine-style fashion advertisements from a simple photo of a shirt.")

# API Key handling (retrieved from server environment)
api_key = os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    st.error("Server Configuration Error: GEMINI_API_KEY not found in environment. Please contact the administrator.")
    st.stop()

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
        "Gemini 2.5 Flash — Cheapest ($0.03/img)",
    ], index=0)
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
        "5:4 (Landscape)",
    ], index=0)

# Model name mapping and per-image cost (USD)
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

# High-end commercial photography templates
FASHION_BACKGROUNDS = [
    # Studio / Indoor
    "in a minimalist white concrete studio with soft diffused skylight",
    "in an industrial Brooklyn loft with massive sunlit windows and exposed brick",
    "in a clean, modern art gallery with abstract sculptures and bright gallery lighting",
    "sitting in a plush leather armchair in a bright, sunlit cocktail lounge with large windows",
    "in a bright tropical resort lobby with lush greenery, natural light, and open-air design",
    "in a high-end barber shop with vintage leather chairs, bright warm lighting, and polished wood paneling",
    "in a luxurious penthouse living room with floor-to-ceiling windows, bright natural daylight flooding in",
    "in a sleek, modern fitness club with bright overhead lighting, clean white walls, and natural light",
    "in an upscale lounge with smooth cream-colored walls, leather seating, warm golden lighting, and tall windows",
    "in a premium car showroom standing next to a luxury sports car, polished concrete floor, dramatic spotlights",
    "in a sophisticated library with tall bookshelves, a rolling ladder, and warm reading lamp light",
    "in a modern architect's office with exposed concrete, steel beams, and a drafting table",
    # Urban / Street
    "walking down a bustling, high-end metropolitan street in SoHo, sharp depth of field",
    "on a bright Tokyo street during daytime with colorful signage, clean sidewalks, and vibrant energy",
    "in a sunlit urban alleyway with a large concrete wall on the right side, warm light",
    "on the steps of a grand European museum or courthouse with massive stone columns",
    "at a busy outdoor café terrace in Paris with wrought-iron chairs and cobblestone streets",
    "on a Brooklyn bridge walkway at golden hour with the Manhattan skyline in soft focus behind",
    "in a Mediterranean old town street with a large stone building wall on the right side, warm sunlight",
    "on a London street with classic red phone booth and black taxi in the blurred background",
    # Nature / Outdoor
    "on a sun-drenched European villa balcony overlooking the Mediterranean",
    "leaning against a classic vintage car in a desert landscape during golden hour",
    "standing on a coastal boardwalk with a concrete sea wall on the right, ocean and bright sky behind",
    "at an exclusive rooftop party at sunset, city skyline glowing in the background",
    "on a pristine white sand beach at sunrise with turquoise water and soft warm light",
    "at a forest cabin with a large flat wooden wall on the right side, golden sunlight filtering through trees",
    "at a mountain lodge with a large stone wall behind, valley views and bright sky in the distance",
    "in a sunlit olive grove or vineyard in Tuscany with rolling hills in the background",
    "at a marina with luxury yachts, calm water reflections, and a warm sunset glow",
    "on a desert highway at sunset with endless road stretching to the horizon, golden light",
    "standing at the edge of an infinity pool overlooking mountains, steam rising in cool air",
    "in a lavender field in Provence at golden hour with purple rows stretching to the horizon",
    # Architectural / Luxury
    "in a Moroccan riad with a large stone arch wall on the right, tilework floor, and a fountain in the background",
    "on the terrace of a Santorini villa with white-washed walls, blue domes, and Aegean Sea views",
    "in a grand hotel corridor with art deco geometric patterns, gold accents, and bright warm lighting",
    "under the arches of a centuries-old European cathedral cloister with stone columns and soft light",
    "in a Japanese zen garden with a large stone wall on the right side, raked gravel and bamboo, soft light",
    "on the steps of a Havana colonial building with faded pastel walls and vintage character",
    "in a bright converted industrial warehouse with large skylights flooding the space with natural light",
]

FASHION_POSES = [
    # Walking / movement — natural editorial energy
    "He is walking confidently, mid-stride, one hand in pocket, looking slightly off-camera with a relaxed expression. Natural movement, like a candid street-style photo.",
    "He is caught mid-walk on a runway-style stride, arms swinging naturally, looking straight ahead with determination.",
    "He is walking at a slight angle to the camera, looking over with a cool glance, hands relaxed at sides.",
    # Standing relaxed — hands in pockets, natural
    "He is standing with both hands in his pockets, weight on one leg, looking slightly off-camera with a calm, confident expression.",
    "He is standing relaxed with one hand in his pocket, the other hanging at his side, looking at the camera with a subtle smile.",
    "He is standing with shoulders slightly angled, one hand casually in pocket, chin slightly up, sharp jawline, intense gaze at camera.",
    # Leaning — cool and editorial
    "He is leaning against a wall with one shoulder, hands in pockets, looking off to the side with a thoughtful expression. Shirt fully visible.",
    "He is casually leaning on a railing or ledge, one forearm resting on it, looking at the camera with relaxed confidence.",
    "He is leaning back against a surface, both hands in pockets, one foot propped behind him, looking off-camera.",
    # Seated — relaxed editorial
    "He is sitting casually on a stool or low surface, one hand on his thigh, the other resting beside him, looking slightly off-camera with a relaxed expression.",
    "He is sitting on steps with hands resting on his knees (not covering the shirt), looking at the camera with a calm, editorial gaze.",
    "He is seated in a chair, leaning back slightly, one ankle resting on the other knee, hands at his sides, relaxed and confident.",
    # Candid / lifestyle moments
    "He is adjusting his sunglasses on his face with one hand, looking slightly off-camera — a candid lifestyle moment.",
    "He is looking down at something in the distance with a thoughtful expression, hands relaxed at his sides, slight body angle.",
    "He is standing with one hand lightly touching the back of his neck, looking off-camera with a natural, relaxed expression.",
    "He has one hand running through his hair, looking slightly upward with a natural, effortless expression.",
    # Dynamic angles
    "Dramatic low-angle shot — he is standing tall, hands at his sides, looking down at the camera with powerful presence.",
    "He is standing at a slight 3/4 angle, looking over his shoulder towards the camera with a confident expression. Shirt front still clearly visible.",
    "He is standing near a prop (car, pillar, furniture), one hand resting on it, looking at the camera with editorial cool.",
    # With accessories/props
    "He is wearing stylish sunglasses, standing with hands in pockets, looking slightly off-camera — effortless cool, editorial fashion.",
    "He is standing near a vintage or luxury element in the scene, one hand lightly touching it, looking at the camera with confidence.",
]

if "generated_campaigns" not in st.session_state:
    st.session_state.generated_campaigns = []
    
if st.button("Generate AI Advertisement", type="primary"):
    if len(shirt_files) == 0:
        st.error("Please upload at least one shirt image.")
    else:
        # Calculate total shots needed
        total_shirts = len(shirt_files)
        total_shots = total_shirts
        current_shot = 0
        
        # Set up progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Clear previous session data if generating new
        st.session_state.generated_campaigns = []
        st.session_state.total_cost = 0.0
        selected_size = image_size.split(" ")[0]
        model_info = MODEL_MAP[gemini_model]
        selected_model_name = model_info["name"]
        cost_per_image = model_info["cost"].get(selected_size, 0.06)

        # Create output directory for this batch
        app_dir = os.path.dirname(os.path.abspath(__file__))
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(app_dir, "output", batch_timestamp)
        batch_start_time = time.time()

        for index, s_file in enumerate(shirt_files):
            progress_text.text(f"Preparing Campaign {index+1} / {total_shirts}...")
            campaign_start_time = time.time()

            # Each shirt gets a unique location
            selected_bg = random.choice(FASHION_BACKGROUNDS)

            # Pick 1 random pose
            selected_poses = [random.choice(FASHION_POSES)]

            shirt_bytes = s_file.read()

            # Create campaign folder: output/<timestamp>/<shirt_name>/
            shirt_name_clean = os.path.splitext(s_file.name)[0]
            campaign_dir = os.path.join(output_base, shirt_name_clean)
            input_dir = os.path.join(campaign_dir, "input")
            generated_dir = os.path.join(campaign_dir, "generated")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(generated_dir, exist_ok=True)

            # Save original input image
            with open(os.path.join(input_dir, s_file.name), "wb") as f:
                f.write(shirt_bytes)

            # --- Step A: Analyze Input for Source Placement ---
            progress_text.text(f"Preparing Campaign {index+1} / {total_shirts} - Semantic Placement Step A...")
            emblem_path = os.path.join(app_dir, "assets", "emblem.png")
            has_logo = False
            source_bbox = None
            
            if os.path.exists(emblem_path) and not logoless:
                source_bbox = get_source_logo_bbox(api_key, shirt_bytes)
                if source_bbox:
                    print(f"Gemini Step A identified logo source bbox: {source_bbox}")
                    has_logo = True
                else:
                    print("Gemini Step A found NO logo. Aborting mapping.")
            # -------------------------------------------------------------

            campaign_shots = []

            for i, pose in enumerate(selected_poses):
                raw_img = None
                status_msg = ""

                for attempt in range(3):
                    progress_text.text(f"Campaign {index+1}/{total_shirts} - Generating shot {i+1} (Attempt {attempt+1}/3)...")

                    raw_img, status_msg = generate_ad(
                        api_key=api_key,
                        shirt_bytes=shirt_bytes,
                        background=selected_bg,
                        pose=pose,
                        image_size=image_size.split(" ")[0],
                        aspect_ratio=aspect_ratio.split(" ")[0],
                        clothing_type=clothing_type,
                        model_name=selected_model_name,
                        logoless=logoless
                    )

                    if raw_img is None and "429" in str(status_msg):
                        if attempt < 2:
                            wait_time = 25.0
                            import re
                            match = re.search(r'Please retry in ([\d\.]+)s', str(status_msg))
                            if match:
                                wait_time = float(match.group(1)) + 2.0
                            for sec in range(int(wait_time), 0, -1):
                                progress_text.text(f"API Rate Limit Hit. Pausing. Retrying in {sec} seconds...")
                                time.sleep(1)
                            continue
                    break # Success or non-rate-limit error

                if raw_img:
                    # Step B & C: Semantic Output Placement & OpenCV Stitching
                    if has_logo and os.path.exists(emblem_path):
                        progress_text.text(f"Campaign {index+1}/{total_shirts} - Semantic Placement Step B...")
                        try:
                            # Step B
                            chest_pos = get_target_logo_center(api_key, raw_img, source_bbox)
                            if chest_pos:
                                print(f"Gemini Step B identified target mapping pixel coordinates: {chest_pos}. Stitching...")
                                # Step C
                                logo_pil = preprocess_logo(emblem_path)
                                if logo_pil is not None:
                                    raw_img = apply_stitched_logo(raw_img, logo_pil, chest_pos, 3.5)
                                else:
                                    print("Failed to preprocess logo, skipping stitching.")
                            else:
                                print("Gemini Step B failed to detect mapping coordinate, skipping logo.")
                        except Exception as e:
                            st.warning(f"Could not apply semantic stitched logo: {e}")

                    # Step 1: Save intermediate image (which now includes the stitched logo)
                    raw_buf = BytesIO()
                    raw_img.save(raw_buf, format="JPEG")
                    raw_bytes = raw_buf.getvalue()

                    # Step 2: Apply SVG overlay frame
                    # The safe logo position logic was removed. Defaulting to left.
                    logo_side = "left"
                    
                    progress_text.text(f"Campaign {index+1}/{total_shirts} - Applying SVG overlay (Side: {logo_side})...")
                    final_img = create_svg_overlay(raw_img, model_no, cost, instagram, logo_pos=logo_side)
                    final_buf = BytesIO()
                    final_img.save(final_buf, format="JPEG")
                    final_bytes = final_buf.getvalue()

                    st.session_state.total_cost += cost_per_image

                    # Save to disk
                    raw_filename = f"ai_generated_{i+1}.jpg"
                    final_filename = f"final_ad_{i+1}.jpg"
                    with open(os.path.join(generated_dir, raw_filename), "wb") as f:
                        f.write(raw_bytes)
                    with open(os.path.join(generated_dir, final_filename), "wb") as f:
                        f.write(final_bytes)

                    campaign_shots.append({
                        "raw_bytes": raw_bytes,
                        "final_bytes": final_bytes,
                        "caption": f"Shot {i+1}",
                        "filename_raw": f"ai_generated_{index}_{i+1}.jpg",
                        "filename_final": f"fashion_ad_{index}_{i+1}.jpg"
                    })
                else:
                    st.error(f"Failed to generate shot {i+1} for campaign {index+1}: {status_msg}")

                # Update progress bar
                current_shot += 1
                progress_bar.progress(current_shot / total_shots)

            campaign_elapsed = time.time() - campaign_start_time
            campaign_cost_usd = cost_per_image * len(campaign_shots)
            campaign_cost_inr = campaign_cost_usd * 85

            # Write info.txt for this campaign
            with open(os.path.join(campaign_dir, "info.txt"), "w", encoding="utf-8") as f:
                f.write(f"Campaign: {s_file.name}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Clothing Type: {clothing_type}\n")
                f.write(f"Model No: {model_no}\n")
                f.write(f"Price: {cost}\n")
                f.write(f"Instagram: {instagram}\n")
                f.write(f"AI Model: {selected_model_name}\n")
                f.write(f"Resolution: {selected_size}\n")
                f.write(f"Aspect Ratio: {aspect_ratio.split(' ')[0]}\n")
                f.write(f"Background: {selected_bg}\n")
                f.write(f"Pose: {selected_poses[0]}\n")
                f.write(f"Shots Generated: {len(campaign_shots)}\n")
                f.write(f"Time Taken: {campaign_elapsed:.1f}s\n")
                f.write(f"Cost: ${campaign_cost_usd:.3f} USD / ₹{campaign_cost_inr:.2f} INR\n")

            st.session_state.generated_campaigns.append({
                "shirt_name": s_file.name,
                "location": selected_bg,
                "shots": campaign_shots,
                "index": index
            })

        batch_elapsed = time.time() - batch_start_time
        total_cost_usd = st.session_state.total_cost
        total_cost_inr = total_cost_usd * 85  # approximate USD to INR

        # Write batch summary info.txt
        with open(os.path.join(output_base, "batch_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Batch Summary\n")
            f.write(f"{'='*40}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Campaigns: {total_shirts}\n")
            f.write(f"Total Shots: {current_shot}\n")
            f.write(f"AI Model: {selected_model_name}\n")
            f.write(f"Resolution: {selected_size}\n")
            f.write(f"Total Time: {batch_elapsed:.1f}s\n")
            f.write(f"Total Cost: ${total_cost_usd:.3f} USD / ₹{total_cost_inr:.2f} INR\n")

        progress_text.text("All Campaigns Completed successfully!")
        st.success(f"Estimated API Cost: ${total_cost_usd:.3f} USD / ₹{total_cost_inr:.2f} INR | Model: {selected_model_name} | {selected_size} | Time: {batch_elapsed:.1f}s")
        st.info(f"Saved to: `{output_base}`")

# Display logic - this runs outside the button so it survives download button clicks
if st.session_state.generated_campaigns:
    for campaign in st.session_state.generated_campaigns:
        st.markdown(f"### Campaign {campaign['index']+1}: {campaign['shirt_name']}")
        st.info(f"Location: {campaign['location']}")

        for i, shot in enumerate(campaign['shots']):
            st.markdown(f"**{shot['caption']}**")
            col_raw, col_final = st.columns(2)

            with col_raw:
                st.markdown("**Step 1: AI Generated** *(emblem in scene)*")
                st.image(shot["raw_bytes"], width="stretch")
                st.download_button(
                    label="Download Raw Image",
                    data=shot["raw_bytes"],
                    file_name=shot["filename_raw"],
                    mime="image/jpeg",
                    key=f"dl_raw_{campaign['index']}_{i}"
                )

            with col_final:
                st.markdown("**Step 2: Final Ad** *(SVG overlay applied)*")
                st.image(shot["final_bytes"], width="stretch")
                st.download_button(
                    label="Download Final Ad",
                    data=shot["final_bytes"],
                    file_name=shot["filename_final"],
                    mime="image/jpeg",
                    key=f"dl_final_{campaign['index']}_{i}"
                )
        st.divider()