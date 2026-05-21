import streamlit as st
import os
import json
import re
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
from auth import require_auth, render_logout_button

load_dotenv()



st.set_page_config(page_title="AI Fashion Ad Generator", layout="wide")

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





def create_svg_overlay(base_img_obj, model_no, cost, instagram, logo_pos="left"):

    """

    Creates a high-quality SVG overlay frame and composites it onto the base image.

    SVG elements: white border, top-right price badge, bottom-left IG handle, top-left logo - note logo must not overlap the model's face.

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



def superimpose_logo(base_img_obj, api_key, shirt_bytes):
    """
    Uses Gemini to find the emblem location, with safety fallbacks for size and placement.
    Pastes the emblem securely without destroying its original colors.
    """
    try:
        client = genai.Client(api_key=api_key)
        
        buf = BytesIO()
        base_img_obj.convert("RGB").save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        
        prompt = (
            "IMAGE 1 is the original input clothing. IMAGE 2 is the generated fashion ad. "
            "Find the exact location on the model's shirt in IMAGE 2 where the chest emblem should go. "
            "Return ONLY a valid JSON array with the bounding box coordinates [ymin, xmin, ymax, xmax]. "
            "Values must be normalized between 0.0 and 1.0. Do NOT return any markdown formatting. "
            "Example: [0.35, 0.65, 0.40, 0.70]"
        )
        
        import re
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                types.Part.from_bytes(data=shirt_bytes, mime_type='image/jpeg'),
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                prompt
            ],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        text = response.text
        match = re.search(r'\[\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*\]', text)
        
        width, height = base_img_obj.size
        
        # Default fallback coordinates (Standard left chest area)
        ymin, xmin, ymax, xmax = 0.35, 0.60, 0.45, 0.70 
        
        if match:
            parsed_ymin = float(match.group(1))
            parsed_xmin = float(match.group(2))
            parsed_ymax = float(match.group(3))
            parsed_xmax = float(match.group(4))
            
            # Sanity Check: Ensure the box isn't dangerously small, massive, or on the face
            box_width = parsed_xmax - parsed_xmin
            if 0.05 < box_width < 0.3 and parsed_ymin > 0.3:
                ymin, xmin, ymax, xmax = parsed_ymin, parsed_xmin, parsed_ymax, parsed_xmax
            else:
                print("Warning: LLM coordinates out of safe bounds. Using fallback.")
        else:
            print(f"Warning: Could not parse bounding box from Gemini: {text}. Using fallback.")

        # Center of bounding box
        chest_cx = int(((xmin + xmax) / 2) * width)
        chest_cy = int(((ymin + ymax) / 2) * height)
        
        # Explicit load of assets/emblem.png
        app_dir = os.path.dirname(os.path.abspath(__file__))
        emblem_path = os.path.join(app_dir, "assets", "emblem.png")
        if not os.path.exists(emblem_path):
            print(f"Warning: Logo path not found: {emblem_path}")
            return base_img_obj
            
        emblem = Image.open(emblem_path).convert("RGBA")
        
        # --- FIX 1: Auto-Crop Invisible Padding ---
        bbox_crop = emblem.getbbox()
        if bbox_crop:
            emblem = emblem.crop(bbox_crop)
        
        # --- FIX 2: Premium Sizing (3.5% of image width) ---
        target_w = int(width * 0.035) 
        emb_w, emb_h = emblem.size
        target_h = int(emb_h * (target_w / emb_w))
        emblem = emblem.resize((target_w, target_h), Image.LANCZOS)
        
        chest_x = chest_cx - target_w // 2
        chest_y = chest_cy - target_h // 2
        
        # --- FIX 3: Adaptive "Thread" Colorization (Subtle & Integrated) ---
        # Sample background luminance to choose a non-jarring tone
        sample_x1 = max(0, chest_x - target_w)
        sample_y1 = max(0, chest_y - target_h)
        sample_x2 = min(width, chest_x + 2*target_w)
        sample_y2 = min(height, chest_y + 2*target_h)
        
        bg_region = base_img_obj.convert("RGB").crop((sample_x1, sample_y1, sample_x2, sample_y2))
        avg_brightness = sum(bg_region.convert("L").getdata()) / max(1, len(bg_region.convert("L").getdata()))
        
        # Muted tones: charcoal for light shirts, silver for dark shirts
        if avg_brightness > 128:
            emblem = _colorize_alpha(emblem, (60, 60, 60)) 
        else:
            emblem = _colorize_alpha(emblem, (210, 210, 210))
            
        # --- FIX 4: Texture-Aware Blending (75% Opacity) ---
        alpha = emblem.split()[3]
        alpha = alpha.point(lambda p: int(p * 0.75))
        emblem.putalpha(alpha)
        
        base_rgba = base_img_obj.convert("RGBA")
        overlay = Image.new("RGBA", base_rgba.size, (0,0,0,0))
        
        # Safety boundary check
        if chest_x >= 0 and chest_y >= 0 and chest_x + target_w <= width and chest_y + target_h <= height:
             overlay.paste(emblem, (chest_x, chest_y), emblem)
        else:
             print("Warning: Coordinates fell off canvas. Dropping logo to prevent crash.")
             return base_img_obj
        
        final = Image.alpha_composite(base_rgba, overlay)
        return final.convert("RGB")
        
    except Exception as e:
        print(f"Warning: Logo superimposition failed: {e}")
        return base_img_obj

def analyze_clothing_features(api_key, image_bytes, feature_type):
    """
    Uses Gemini 1.5 Pro to analyze the input image for specific features like pockets or logos.
    """
    try:
        client = genai.Client(api_key=api_key)
        img = Image.open(BytesIO(image_bytes))
        
        if feature_type == "Pocket":
            prompt = (
                "Analyze this clothing item. Does it have any pockets? If yes, describe them in detail: "
                "their position (left chest, right chest, etc.), shape (square, pointed, rounded), "
                "closure (button, flap, open), and any visible stitching patterns. "
                "If there are no pockets, say 'No pockets visible'."
            )
        elif feature_type == "Logo":
            prompt = (
                "Analyze this clothing item. Does it have any logos, emblems, or embroidery? "
                "If yes, describe them: their position, color, shape, and what they depict. "
                "If there are no logos, say 'No logos visible'."
            )
        else:
            return ""

        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[img, prompt],
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error in feature analysis: {e}")
        return ""

def generate_ad(api_key, shirt_bytes, background, pose, image_size="1K", aspect_ratio="3:4", clothing_type="Auto-Detect", model_name="gemini-3-pro-image-preview", clothing_features="Default (Logo-less)"):

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



    # Feature analysis for high precision if requested
    feature_desc = ""
    if clothing_features == "Pocket":
        feature_desc = analyze_clothing_features(api_key, shirt_bytes, "Pocket")
    elif clothing_features == "Logo (Add Emblem)":
        # We don't necessarily need to analyze the input logo if we are just adding the emblem, 
        # but we could. For now, let's just focus on the instructions.
        pass

    # Determine clothing description
    if clothing_type == "Auto-Detect":
        clothing_desc = (
            "Analyze the clothing item and determine its type (shirt, t-shirt, kurta, jacket, hoodie, polo, etc.). "
            "If it is a shirt, it should be a half-sleeve shirt unless it has long sleeves. "
            "If it is a long-sleeve shirt, the sleeves must be folded up so they end a few inches below the elbow."
        )
    elif clothing_type == "Shirt":
        clothing_desc = "This is a half-sleeve shirt."
    elif clothing_type == "Full Sleeve Shirt":
        clothing_desc = "This is a long-sleeve shirt. The sleeves must be folded up - note they shouldn't cross teh elbow, the hands of teh shirt should be below elbow strictly."
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



    # Clothing detail instruction — reproduce ONLY what exists in the original image

    clothing_detail = (

        f"The model MUST wear this EXACT item shown in the CLOTHING ITEM image. {denim_theme} Study the clothing image carefully. "
        f"You MUST reproduce the clothing EXACTLY as it appears in the photo.\n\n"


        f"STRICT RULES FOR FAITHFUL REPRODUCTION:\n"
    )

    if clothing_features == "Pocket":
        clothing_detail += f"- POCKETS: {feature_desc}. Ensure the generated clothing has these pockets exactly as described.\n"
    elif clothing_features == "Pocket-less":
        clothing_detail += "- NO POCKETS: The generated clothing MUST NOT have any pockets, even if the input image shows one.\n"
    else:
        clothing_detail += "- NO FAKE POCKETS: If the original clothing does not have a pocket, the generated clothing MUST NOT have a pocket.\n"

    if clothing_features == "Logo (Add Emblem)":
        clothing_detail += "- CHEST AREA: Keep the chest area (specifically the left chest) clean and plain fabric. This is where we will add our own logo overlay, so do NOT generate any AI logos there.\n"
    elif clothing_features == "Logo-less" or clothing_features == "Default (Logo-less)":
        clothing_detail += "- NO LOGOS OR EMBROIDERY: Strictly ensure there is NO logo, emblem, brand mark, or embroidery on the garment, especially on the chest. You MUST REMOVE and ERASE any logos, emblems, or brand marks present in the input image and generate plain fabric in its place. The garment must be a 100% unbranded, plain canvas.\n"
    else:
        clothing_detail += "- NO FAKE LOGOS: If the original clothing does not have a logo or embroidery, the generated clothing MUST NOT have a logo or embroidery.\n"

    clothing_detail += "- NO EXTRA DETAILS: Do not invent missing details. Preserve everything EXACTLY as photographed EXCEPT for the specific feature instructions above.\n"

    # Define the prompt
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
    styling_rules = (
        f"STYLING RULES:\n"
        f"- This is a CASUAL brand — the overall look must feel relaxed, stylish, and approachable. NOT formal, NOT corporate, NOT stiff.\n"
        f"- Pair with jeans, chinos, or casual bottoms.\n"
        f"- Kurta: with traditional bottoms, sleeves rolled above wrist.\n"
        f"- Pants MUST contrast — NEVER same fabric/pattern as top.\n\n"
    )
    pose_rules = (
        f"POSE RULES:\n"
        f"- The FRONT of the shirt/clothing MUST be clearly visible — this is the PRODUCT.\n"
        f"- NO crossed arms, folded arms, or hands covering the chest/torso area.\n"
        f"- The model can look slightly off-camera, at an angle, or directly at camera — keep it natural and editorial, NOT stiff or posed like a mannequin.\n"
        f"- Slight body angles (3/4 turn) are fine as long as the shirt front is visible.\n"
        f"- Arms can be at sides, in pockets, touching face/hair, or resting on nearby surfaces.\n\n"
    )
    framing_rules = (
        f"MODEL & FRAMING:\n"
        f"- VERY IMPORTANT: Leave massive headroom (at least 25-30% empty space) at the VERY TOP of the image.\n"
        f"- The model's head MUST NOT touch or get close to the top edge.\n"
        f"- Frame from the waist or mid-thigh up, scaling the model so they occupy only the bottom 60-70% of the image to allow for the top headroom.\n"
        f"- The model should look like a regular, good-looking guy — NOT a runway supermodel. Natural, approachable, casual vibe. Think weekend hangout, not fashion show.\n"
        f"- Relaxed expression — slight smile, calm eyes, easy-going.\n"
        f"- Natural lighting on face and clothing, NOT dramatic studio lighting.\n\n"
    )

    prompt += (
        f"{styling_rules}"
        f"POSE: {pose}\n"
        f"{pose_rules}"
        f"SETTING: {background}.\n\n"
        f"{framing_rules}"
        f"COMPOSITION ZONES (follow exactly):\n"
    )

    prompt += (
        f"- Position the model CENTER to CENTER-RIGHT of the frame, positioned lower in the frame.\n"

        f"- ENTIRE TOP 30% OF THE IMAGE: This entire top area MUST be 100% CLEAN "
        f"background only (e.g., plain wall, soft sky). ABSOLUTELY NO part of the model's body, face, head, hair, or clothing may "
        f"enter this top zone. The model's head must be positioned well BELOW this 30% mark to create negative space for branding.\n"

        f"- TOP-RIGHT CORNER: MUST be clean, uncluttered background space.\n"

        f"- BOTTOM-LEFT EDGE: MUST be clean, minimalist background space.\n"

    )



    prompt += f"- BOTTOM-RIGHT CORNER: Keep relatively clean.\n\n"



    prompt += (

        f"TECHNICAL: Bright, fresh, well-lit photography with vibrant colors and a light, "

        f"airy, refreshing feel. Use natural daylight, golden hour warmth, or bright studio "

        f"lighting. AVOID dark, moody, underexposed, or heavily shadowed images. "

        f"Sharp focus, {aspect_ratio} aspect ratio.\n\n"

    )



    prompt += (
        f"CRITICAL: Output must contain ZERO text, logos, watermarks, or typography. "
        f"The clothing MUST be completely unbranded and plain. Check your work: ensure no tiny badges, emblems, or branding elements appear on the garment."
    )



    # Build contents list — order must match IMAGE 1/2/3 in the prompt

    # Send emblem TWICE: once in sequence (referenced by prompt) and once after prompt

    # to reinforce its shape at the end where Gemini pays most attention (recency bias)

    contents = []

    if reference_bytes:

        contents.append(types.Part.from_bytes(data=reference_bytes, mime_type='image/jpeg'))

    contents.append(types.Part.from_bytes(data=shirt_bytes, mime_type='image/jpeg'))

    # Send emblem TWICE logic removed

    contents.append(prompt)



    # System instruction for faithful reproduction
    system_prompt = (
        "You are a casual lifestyle brand photographer. The vibe is relaxed, approachable, and natural.\n"
        "CRITICAL PRIORITY: Reproduce the clothing item from the reference photo EXACTLY as shown (fabric, color, pattern, etc.).\n"
        "DO NOT hallucinate or add any features not specified. "
    )

    if clothing_features == "Pocket":
        system_prompt += "Ensure the generated clothing has the exact pockets as described in the detail section. Mirror their structure and placement.\n"
    elif clothing_features == "Pocket-less":
        system_prompt += "Strictly ensure the generated clothing HAS NO POCKETS. If the input image has one, remove it.\n"
    else:
        system_prompt += "Do NOT add any pockets if they are not clearly visible in the input image.\n"

    if clothing_features == "Logo (Add Emblem)":
        system_prompt += "Keep the chest area completely clean and plain fabric for a post-processing logo superimposition. Do NOT generate any AI logos on the clothing.\n"
    elif clothing_features == "Logo-less" or clothing_features == "Default (Logo-less)":
        system_prompt += "MANDATORY: NO LOGOS OR EMBROIDERY. The garment MUST be 100% logo-free. If the input image contains a logo or embroidery, REMOVE and ERASE it entirely. Do NOT generate any stitched designs or brand marks on the garment. The fabric must be perfectly clean and unbranded.\n"
    else:
        system_prompt += "Do NOT add any fake logos or embroidery if they are not in the input image.\n"

    system_prompt += "Generate a photorealistic, high-quality image of the model wearing this item."



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

    clothing_type = st.selectbox("Clothing Type", ["Auto-Detect", "Shirt", "Denim Shirt - Short Sleeve", "Denim Shirt - Long Sleeve", "Full Sleeve Shirt", "T-Shirt", "Kurta", "Jacket", "Hoodie", "Polo"])


    model_no = st.text_input("Model Number", value="HN-0230")

    cost = st.text_input("Price / Cost Text", value="1399/-")

    instagram = st.text_input("Instagram Handle", value="@hatchersclothingcompany")

    gemini_model = st.selectbox("AI Model", [

        "Gemini 3 Pro — Best Quality ($0.12/img)",

        "Gemini 3.1 Flash — Quality + Speed ($0.06/img)",

        "Gemini 2.5 Flash — Cheapest ($0.03/img)",

    ], index=0)

    image_size = st.selectbox("Image Resolution", ["1K (~1024px)", "2K (~2048px)", "4K (~4096px)"], index=0)
    clothing_features = st.selectbox("Clothing Features", ["Default (Logo-less)", "Pocket", "Pocket-less", "Logo (Add Emblem)", "Logo-less"])
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
                        clothing_features=clothing_features
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

                    # Step 1: Save raw AI-generated image (with embedded emblem)

                    raw_buf = BytesIO()

                    raw_img.save(raw_buf, format="JPEG")

                    raw_bytes = raw_buf.getvalue()



                    # Step 1.5: Apply superimposition (conditional)
                    if clothing_features == "Logo (Add Emblem)":
                        progress_text.text(f"Campaign {index+1}/{total_shirts} - Superimposing logo...")
                        raw_img = superimpose_logo(raw_img, api_key, shirt_bytes)

                    # Step 2: Apply SVG overlay frame
                    progress_text.text(f"Campaign {index+1}/{total_shirts} - Analyzing safe logo position...")
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