import streamlit as st
import os
import tempfile
from pdf2image import convert_from_path
import io
import base64
from PIL import Image
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
import time
import json

# Page configuration
st.set_page_config(
    page_title="PDF Transcription with Gemini",
    page_icon="📄",
    layout="wide"
)

# Application title and description
st.title("PDF Transcription with Google Gemini AI")
st.markdown("""
This application allows you to upload a PDF document and transcribe its content using Google Gemini AI.
The PDF is converted to images and each page is processed sequentially to extract the text content.
""")

# Sidebar for API key configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google Gemini API Key", type="password")

    # Fixed model - only using gemini-2.0-flash
    model_choice = "gemini-2.0-flash"
    st.info(f"Using model: {model_choice}")

    # Debug mode toggle
    debug_mode = st.checkbox("Enable Debug Mode", value=False)

    if api_key:
        try:
            # Configure the Gemini API with the provided key
            genai.configure(api_key=api_key)
            st.success("API key configured successfully!")

            # Verify API connectivity if in debug mode
            if debug_mode:
                try:
                    # Simple model list test to verify API connectivity
                    models = genai.list_models()
                    model_names = [model.name for model in models]
                    st.write("Available models:", model_names)
                except Exception as e:
                    st.error(f"API Connection Test Failed: {str(e)}")
                    st.info("This may indicate an invalid API key or connectivity issues.")
        except Exception as e:
            st.error(f"Error configuring API key: {str(e)}")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses:
    - Streamlit for the web interface
    - pdf2image for PDF to image conversion
    - Google Generative AI SDK for Gemini API integration
    - PIL (Pillow) for image processing
    """)

# Function to convert PDF page to images
def convert_pdf_to_images(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        # Convert PDF to images
        images = convert_from_path(tmp_path, dpi=300)

        # Clean up temp file
        os.unlink(tmp_path)

        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return []

# Function to transcribe an image using Gemini
def transcribe_image(img, model_name="gemini-2.0-flash", debug=False):
    try:
        # Create a generative model instance
        model = genai.GenerativeModel(model_name)

        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Encode image to base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        if debug:
            st.write(f"Image size: {len(img_bytes)} bytes")
            st.write(f"Base64 image size: {len(img_b64)} bytes")

            if len(img_b64) > 20 * 1024 * 1024:  # 20MB limit
                st.warning("Image exceeds 20MB limit when encoded. Consider reducing image quality.")

        # Create prompt parts with the image following Google's Python format
        prompt_parts = [
            "Please transcribe all the text content from this image accurately. Preserve paragraphs, bullet points, and overall formatting. For ALL mathematical expressions and equations, provide them in proper LaTeX format surrounded by $ for inline math or $$ for display math. This is critical for proper rendering. Do not add any explanatory text or commentary to your transcription.",
            {
                "mime_type": "image/png", 
                "data": img_b64
            }
        ]

        if debug:
            st.write("Prompt structure:", type(prompt_parts))
            st.write("Sending request to Gemini API...")

        # Generate content
        response = model.generate_content(prompt_parts)

        if debug:
            st.write("Response received:", type(response))

        return response.text
    except GoogleAPIError as e:
        error_message = f"Gemini API Error: {str(e)}"
        if "403" in str(e):
            error_message += "\n\nThis is likely due to API key authentication issues. Please check that your API key is valid and has access to the Gemini API."
        st.error(error_message)
        return f"Transcription error: {error_message}"
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return f"Transcription error: {str(e)}"

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Create a container for displaying the PDF
    preview_col, results_col = st.columns([1, 2])

    with preview_col:
        st.subheader("PDF Preview")
        # Display PDF file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)

        # Create a button to process the PDF
        if st.button("Process PDF"):
            if not api_key:
                st.error("Please enter your Google Gemini API key in the sidebar first.")
            else:
                # Convert PDF to images
                with st.spinner("Converting PDF to images..."):
                    images = convert_pdf_to_images(uploaded_file)

                if not images:
                    st.error("Failed to convert PDF to images. Please try again with a different PDF.")
                else:
                    st.success(f"Successfully converted PDF to {len(images)} page images.")

                    # Create containers for results
                    transcription_results = []
                    all_text = ""
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Process each image
                    for i, img in enumerate(images):
                        status_text.text(f"Processing page {i+1}/{len(images)}...")

                        # Display the current image being processed
                        st.image(img, caption=f"Page {i+1}", width=300)

                        # If image is too large, resize it to reduce API payload size
                        if img.width > 1000 or img.height > 1000:
                            st.info(f"Resizing large image (original size: {img.width}x{img.height})")
                            # Calculate new dimensions while maintaining aspect ratio
                            ratio = min(1000 / img.width, 1000 / img.height)
                            new_size = (int(img.width * ratio), int(img.height * ratio))
                            img = img.resize(new_size, Image.LANCZOS)
                            st.info(f"Resized to: {img.width}x{img.height}")

                        # Transcribe image with the fixed model (gemini-2.0-flash)
                        transcription = transcribe_image(img, model_name=model_choice, debug=debug_mode)
                        transcription_results.append(transcription)
                        all_text += f"\n\n--- PAGE {i+1} ---\n\n{transcription}"

                        # Update progress
                        progress_bar.progress((i + 1) / len(images))

                        # Brief pause to avoid rate limiting
                        time.sleep(1)

                    status_text.text("Processing completed!")

                    # Store results in session state for display
                    st.session_state.page_images = images
                    st.session_state.transcription_results = transcription_results
                    st.session_state.all_text = all_text.strip()
                    st.session_state.processed = True

    with results_col:
        st.subheader("Transcription Results")

        if 'processed' in st.session_state and st.session_state.processed:
            # Create tabs for viewing results
            tabs = st.tabs(["Complete Document", "Page by Page"])

            with tabs[0]:
                st.text_area("Complete Transcription", 
                              st.session_state.all_text, 
                              height=500)

                # Download button for complete text
                st.download_button(
                    label="Download Complete Transcription",
                    data=st.session_state.all_text,
                    file_name=f"{uploaded_file.name.split('.')[0]}_transcription.txt",
                    mime="text/plain"
                )

            with tabs[1]:
                # Page selection
                page_count = len(st.session_state.page_images)
                selected_page = st.selectbox("Select page", range(1, page_count+1))

                # Display the selected page and its transcription
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        st.session_state.page_images[selected_page-1], 
                        caption=f"Page {selected_page}", 
                        use_column_width=True
                    )

                with col2:
                    st.text_area(
                        f"Page {selected_page} Transcription", 
                        st.session_state.transcription_results[selected_page-1], 
                        height=400
                    )

                    # Download button for single page transcription
                    st.download_button(
                        label=f"Download Page {selected_page} Transcription",
                        data=st.session_state.transcription_results[selected_page-1],
                        file_name=f"{uploaded_file.name.split('.')[0]}_page{selected_page}_transcription.txt",
                        mime="text/plain"
                    )
else:
    # Display sample image when no PDF is uploaded
    st.info("Please upload a PDF document to begin the transcription process.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### How it works:
        1. Upload a PDF document
        2. Enter your Google Gemini API key in the sidebar
        3. Click 'Process PDF' to start transcription
        4. View the results page by page or as a complete document
        5. Download the transcription as a text file
        """)

    with col2:
        st.markdown("""
        ### Tips for best results:
        - Use PDFs with clear, readable text
        - Larger files may take more time to process
        - If you encounter rate limits, wait a few minutes and try again
        - For multi-page PDFs, each page is processed individually
        - Enable debug mode to troubleshoot API issues
        """)

    # API key information box
    st.markdown("""
    ### Getting a Google Gemini API Key:
    1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Sign in with your Google account
    3. Create a new API key
    4. Copy the API key and paste it in the sidebar
    """)