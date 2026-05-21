import streamlit as st
import os
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
import time
from auth import require_auth, render_logout_button

st.set_page_config(page_title="AI Fashion - Upscaler", layout="wide")

# ── JWT Auth Guard ───────────────────────────────────────────────────────────
require_auth()
render_logout_button()
# ─────────────────────────────────────────────────────────────────────────────


st.title("Image Upscaler")
st.markdown("Upload images and upscale them to 4K resolution using Google Gemini.")

# API Key handling (retrieved from server environment)
api_key = os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    st.error("Server Configuration Error: GEMINI_API_KEY not found in environment. Please contact the administrator.")
    st.stop()

# Initialize session state for results to persist across reruns (e.g., after downloads)
if "upscaled_results" not in st.session_state:
    st.session_state.upscaled_results = []

st.subheader("Upload Images")
uploaded_files = st.file_uploader("Upload Images (JPEG/PNG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Total images uploaded: {len(uploaded_files)}")
    
    if st.button("Upscale All to 4K", type="primary"):
        # Clear previous results if starting a new batch
        st.session_state.upscaled_results = []
        
        for index, uploaded_file in enumerate(uploaded_files):
            with st.status(f"Processing Image {index + 1}/{len(uploaded_files)}: {uploaded_file.name}...", expanded=True) as status:
                try:
                    client = genai.Client(api_key=api_key)
                    start_time = time.time()
                    
                    # Read bytes once
                    image_bytes = uploaded_file.read()
                    
                    # Try to infer aspect ratio
                    img = Image.open(BytesIO(image_bytes))
                    width, height = img.size
                    
                    aspect_ratio = "3:4" # Default
                    if width > height:
                        aspect_ratio = "4:3"
                    elif width == height:
                        aspect_ratio = "1:1"
                    else:
                        ratio = width / height
                        if abs(ratio - (9/16)) < 0.05:
                            aspect_ratio = "9:16"
                        elif abs(ratio - (2/3)) < 0.05:
                            aspect_ratio = "2:3"
                        elif abs(ratio - (4/5)) < 0.05:
                            aspect_ratio = "4:5"
                        else:
                            aspect_ratio = "3:4"
                    
                    prompt = "Upscale this image to 4K resolution. Enhance the textures, sharpness, and details of the clothing and lighting. Do NOT alter the composition, colors, or add any new elements. Maintain the exact same scene."
                    
                    contents = [
                        types.Part.from_bytes(data=image_bytes, mime_type=uploaded_file.type),
                        prompt
                    ]
                    
                    response = client.models.generate_content(
                        model="gemini-3-pro-image-preview",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(image_size="4K", aspect_ratio=aspect_ratio),
                        )
                    )
                    
                    if response.candidates:
                        generated_img = None
                        for part in response.candidates[0].content.parts:
                            if part.inline_data:
                                generated_img = Image.open(BytesIO(part.inline_data.data))
                                break
                        
                        if generated_img:
                            elapsed_time = time.time() - start_time
                            cost_usd = 0.24
                            cost_inr = cost_usd * 85
                            
                            status.update(label=f"Completed: {uploaded_file.name} (Time: {elapsed_time:.1f}s)", state="complete")
                            
                            # Save to session state
                            out_buf = BytesIO()
                            generated_img.save(out_buf, format="JPEG")
                            out_bytes = out_buf.getvalue()
                            
                            st.session_state.upscaled_results.append({
                                "name": uploaded_file.name,
                                "original_bytes": image_bytes,
                                "upscaled_bytes": out_bytes,
                                "time": elapsed_time,
                                "cost_usd": cost_usd,
                                "cost_inr": cost_inr
                            })
                        else:
                            st.error(f"Model did not return valid image data for {uploaded_file.name}.")
                    else:
                        st.error(f"Error: Gemini returned no candidates for {uploaded_file.name}.")
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                # Small pause to help with rate limiting
                time.sleep(1)

# Display results from session state (outside the button to persist after rerun)
if st.session_state.upscaled_results:
    st.divider()
    st.subheader("Results")
    for index, result in enumerate(st.session_state.upscaled_results):
        st.markdown(f"### {index + 1}. {result['name']}")
        col1, col2 = st.columns(2)
        with col1:
            st.image(result['original_bytes'], caption="Original", use_container_width=True)
        with col2:
            st.image(result['upscaled_bytes'], caption=f"Upscaled 4K (Time: {result['time']:.1f}s)", use_container_width=True)
            st.download_button(
                label=f"Download {result['name']} (4K)",
                data=result['upscaled_bytes'],
                file_name=f"upscaled_{result['name']}",
                mime="image/jpeg",
                key=f"dl_{index}"
            )
        st.divider()
