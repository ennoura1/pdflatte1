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
import concurrent.futures
import markdown
import weasyprint
import re
import latex2mathml.converter

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
You can also translate the transcription to Arabic.
""")

# Sidebar for API key configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google Gemini API Key", type="password")

    # Model selection
    model_choice = st.selectbox(
        "Choose Gemini Model",
        ["gemini-2.0-flash", "gemini-2.0-pro-exp-02-05"],
        index=0
    )
    st.info(f"Using model: {model_choice}")

    # Parallel processing option
    parallel_processing = st.checkbox("Enable Parallel Processing (Faster)", value=True)
    if parallel_processing:
        max_workers = st.slider("Maximum Concurrent API Calls", min_value=1, max_value=5, value=3)
        st.info(f"Using up to {max_workers} concurrent API calls")

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
    - Concurrent processing for faster results
    - Markdown and WeasyPrint for PDF generation
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
        # Create a NEW generative model instance for each request
        # This ensures no context bleeding between requests
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
            """Please transcribe all the text content from this image accurately. Format your response using proper Markdown syntax with these requirements:

1. Use # for main titles, ## for subtitles, and ### for section headers
2. Use **bold** for emphasis and important terms
3. Use *italics* for emphasized phrases or references
4. Structure paragraphs with proper line breaks
5. Use proper Markdown lists (- or 1. for numbered lists)
6. For ALL mathematical expressions and equations, provide them in proper LaTeX format surrounded by $ for inline math or $$ for display math (this is critical for proper rendering)
7. Format tables using Markdown table syntax
8. Use > for quoted text
9. Use proper heading levels to maintain document hierarchy
10. Use code blocks with ``` for code snippets

DO NOT WRAP YOUR ENTIRE RESPONSE IN ```markdown CODE BLOCKS.
DO NOT prefix your response with ```markdown.
DO NOT suffix your response with ```.
DO NOT use any triple backtick markdown notation at all in your response.
NEVER use the string ```markdown in your response.
Simply deliver a well-formatted Markdown document that represents the content of the image.
""",
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

# Function to translate text to Arabic
def translate_to_arabic(text, model_name="gemini-2.0-flash", debug=False):
    try:
        # Create a new model instance for translation
        model = genai.GenerativeModel(model_name)

        # Create prompt for translation
        prompt = f"""
        Translate the following text to Arabic. If the text contains any LaTeX math expressions
        (surrounded by $ or $$), keep those expressions exactly as they are without translating them.
        Only translate the regular text, not the LaTeX math syntax or content.
        Format your response in Markdown for proper rendering.

        DO NOT WRAP YOUR RESPONSE IN ```markdown CODE BLOCKS.
        DO NOT prefix your response with ```markdown.
        DO NOT suffix your response with ```.
        DO NOT use any triple backtick markdown notation at all in your response.
        NEVER use the string ```markdown in your response.

        Here's the text to translate:

        {text}
        """

        if debug:
            st.write("Sending translation request to Gemini API...")

        # Generate translation
        response = model.generate_content(prompt)

        if debug:
            st.write("Translation response received")

        return response.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}"

# Function to process a single page - used for both sequential and parallel processing
def process_page(page_data):
    img, i, total_pages, model_choice, debug_mode = page_data

    # If image is too large, resize it to reduce API payload size
    if img.width > 1000 or img.height > 1000:
        if debug_mode:
            st.info(f"Resizing large image (original size: {img.width}x{img.height})")
        # Calculate new dimensions while maintaining aspect ratio
        ratio = min(1000 / img.width, 1000 / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        if debug_mode:
            st.info(f"Resized to: {img.width}x{img.height}")

    # Add debug message for individual page processing
    if debug_mode:
        st.write(f"Starting new API request for page {i+1}")

    # Transcribe image
    transcription = transcribe_image(img, model_name=model_choice, debug=debug_mode)

    return i, transcription

# Function to translate a single page
def translate_page(page_data):
    text, i, total_pages, model_choice, debug_mode = page_data

    if debug_mode:
        st.write(f"Starting translation for page {i+1}")

    # Translate text
    translation = translate_to_arabic(text, model_name=model_choice, debug=debug_mode)

    return i, translation

# Function to convert LaTeX to MathML
def convert_latex_to_mathml(text):
    # Function to convert a single LaTeX expression to MathML
    def replace_with_mathml(match):
        latex_content = match.group(1)
        try:
            # Convert LaTeX to MathML
            mathml = latex2mathml.converter.convert(latex_content)
            return mathml
        except Exception as e:
            # If conversion fails, return the original LaTeX
            return match.group(0)

    # Convert display math expressions ($$...$$)
    display_pattern = r'\$\$(.*?)\$\$'
    text = re.sub(display_pattern, lambda m: replace_with_mathml(m), text, flags=re.DOTALL)

    # Convert inline math expressions ($...$)
    inline_pattern = r'\$(.*?)\$'
    text = re.sub(inline_pattern, lambda m: replace_with_mathml(m), text, flags=re.DOTALL)

    return text

# Function to remove page headers for PDF generation
def remove_page_headers(markdown_text):
    # Remove ## Page X headers
    return re.sub(r'## Page \d+\n\n', '', markdown_text)

# Function to convert markdown to PDF
def markdown_to_pdf(markdown_text, output_path, title="PDF Transcription"):
    try:
        # Remove page headers from the markdown text
        markdown_text = remove_page_headers(markdown_text)

        # Pre-process markdown to convert LaTeX to MathML
        processed_text = convert_latex_to_mathml(markdown_text)

        # Convert markdown to HTML
        html = markdown.markdown(processed_text, extensions=['extra', 'codehilite'])

        # Create the final HTML document
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&display=swap');
                @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400;1,600&display=swap');

                body {{
                    font-family: 'Merriweather', 'Georgia', serif;
                    line-height: 1.8;
                    margin: 3em;
                    color: #333;
                    font-size: 11pt;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    font-family: 'Source Sans Pro', 'Helvetica', sans-serif;
                    color: #1a1a1a;
                    margin-top: 1.5em;
                    margin-bottom: 0.8em;
                    line-height: 1.2;
                }}
                h1 {{ font-size: 24pt; font-weight: 700; }}
                h2 {{ font-size: 20pt; font-weight: 600; }}
                h3 {{ font-size: 16pt; font-weight: 600; }}

                p {{
                    margin-bottom: 1.2em;
                    text-align: justify;
                }}
                pre {{
                    background-color: #f8f8f8;
                    border: 1px solid #e0e0e0;
                    padding: 12px;
                    border-radius: 4px;
                    font-size: 10pt;
                    overflow-x: auto;
                    line-height: 1.4;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 1.5em 0;
                }}

                /* Math styling */
                math {{
                    font-size: 1.15em;
                    font-weight: normal;
                    line-height: 1.5;
                }}
                math > mfrac {{
                    font-size: 1.15em;
                    line-height: 1.5;
                    vertical-align: -0.5em;
                }}
                math > msup, math > msub {{
                    line-height: 1;
                }}
                math > mi, math > mn {{
                    font-style: normal;
                    padding: 0 0.1em;
                }}
                math > mo {{
                    padding: 0 0.2em;
                }}

                /* Equations spacing */
                math[display="block"] {{
                    display: block;
                    text-align: center;
                    margin: 1.5em 0;
                    text-indent: 0;
                }}

                /* Tables */
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1.5em 0;
                }}
                th, td {{
                    padding: 8px 12px;
                    border: 1px solid #e0e0e0;
                }}
                th {{
                    background-color: #f5f5f5;
                    font-weight: 600;
                }}

                /* Quotes */
                blockquote {{
                    border-left: 4px solid #e0e0e0;
                    padding-left: 1em;
                    margin-left: 0;
                    font-style: italic;
                }}

                /* RTL support for Arabic */
                .rtl {{
                    direction: rtl;
                    text-align: right;
                    font-family: 'Amiri', 'Traditional Arabic', serif;
                    line-height: 1.8;
                }}
                .rtl h1, .rtl h2, .rtl h3, .rtl h4, .rtl h5, .rtl h6 {{
                    font-family: 'Amiri', 'Traditional Arabic', serif;
                }}

                /* Print-specific styles */
                @page {{
                    margin: 2.5cm 2cm;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="{'rtl' if 'Arabic' in title else ''}">
                {html}
            </div>
        </body>
        </html>
        """

        # Convert HTML to PDF
        pdf = weasyprint.HTML(string=html).write_pdf()

        # Write PDF to file
        with open(output_path, 'wb') as f:
            f.write(pdf)

        return True
    except Exception as e:
        st.error(f"Error converting markdown to PDF: {str(e)}")
        return False

# JavaScript function for copying text to clipboard
def get_copy_button_js():
    return """
    <script>
    function copyToClipboard(text) {
        const temp = document.createElement('textarea');
        temp.value = text;
        document.body.appendChild(temp);
        temp.select();
        document.execCommand('copy');
        document.body.removeChild(temp);

        // Show a brief "Copied!" message
        const message = document.createElement('div');
        message.textContent = 'Copied!';
        message.style.position = 'fixed';
        message.style.left = '50%';
        message.style.top = '10%';
        message.style.transform = 'translate(-50%, -50%)';
        message.style.padding = '8px 16px';
        message.style.background = '#4CAF50';
        message.style.color = 'white';
        message.style.borderRadius = '4px';
        message.style.zIndex = '9999';
        document.body.appendChild(message);

        setTimeout(() => {
            document.body.removeChild(message);
        }, 2000);
    }
    </script>
    """

# Function to create a copy button for text
def create_copy_button(text, button_label="Copy All Text"):
    # Generate a unique ID for this button
    button_id = f"copy_button_{hash(text)}"

    # Properly encode the text for JavaScript by using JSON serialization
    import json
    js_text = json.dumps(text)

    # Create the HTML for the button and JavaScript
    copy_button_html = f"""
    {get_copy_button_js()}
    <button id="{button_id}" 
            onclick="copyToClipboard({js_text});" 
            style="background-color: #4CAF50; color: white; padding: 8px 16px; 
                   border: none; border-radius: 4px; cursor: pointer; 
                   font-size: 14px; margin: 5px 0; display: inline-flex; 
                   align-items: center; gap: 8px;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        {button_label}
    </button>
    """

    # Display the button using st.markdown with unsafe_allow_html
    st.markdown(copy_button_html, unsafe_allow_html=True)

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
                    transcription_results = [None] * len(images)  # Pre-allocate list
                    all_text = ""
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Determine processing method (parallel or sequential)
                    if parallel_processing:
                        status_text.text(f"Processing {len(images)} pages in parallel...")

                        # Prepare page data for parallel processing
                        page_data = [(img, i, len(images), model_choice, debug_mode) 
                                     for i, img in enumerate(images)]

                        # Process pages in parallel
                        completed = 0
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_page = {executor.submit(process_page, data): data for data in page_data}

                            for future in concurrent.futures.as_completed(future_to_page):
                                i, transcription = future.result()
                                transcription_results[i] = transcription

                                # Display the processed image
                                st.image(images[i], caption=f"Page {i+1}", width=300)

                                # Update progress
                                completed += 1
                                progress_bar.progress(completed / len(images))
                                status_text.text(f"Processed page {i+1}/{len(images)}...")

                        # Combine results in correct order
                        all_text = ""
                        for i, transcription in enumerate(transcription_results):
                            # Use markdown header formatting for page separators
                            all_text += f"\n\n{transcription}"

                    else:
                        # Process each image individually (sequential)
                        for i, img in enumerate(images):
                            status_text.text(f"Processing page {i+1}/{len(images)}...")

                            # Display the current image being processed
                            st.image(img, caption=f"Page {i+1}", width=300)

                            # Process page
                            _, transcription = process_page((img, i, len(images), model_choice, debug_mode))
                            transcription_results[i] = transcription
                            all_text += f"\n\n{transcription}"

                            # Update progress
                            progress_bar.progress((i + 1) / len(images))

                    status_text.text("Processing completed!")

                    # Store results in session state for display
                    st.session_state.page_images = images
                    st.session_state.transcription_results = transcription_results
                    st.session_state.all_text = all_text.strip()
                    st.session_state.processed = True

    with results_col:
        st.subheader("Results")

        if 'processed' in st.session_state and st.session_state.processed:
            # Create tabs for viewing results
            tabs = st.tabs(["Complete Document", "Page by Page", "Arabic Translation", "PDF Export"])

            with tabs[0]:
                st.text_area("Complete Transcription", 
                               st.session_state.all_text, 
                               height=500)

                # Add Copy All button for complete transcription
                create_copy_button(st.session_state.all_text, "Copy Complete Transcription")

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
                        use_container_width=True
                    )

                with col2:
                    page_text = st.session_state.transcription_results[selected_page-1]
                    st.text_area(
                        f"Page {selected_page} Transcription", 
                        page_text, 
                        height=400
                    )

                    # Add Copy button for single page transcription
                    create_copy_button(page_text, f"Copy Page {selected_page} Transcription")

                    # Download button for single page transcription
                    st.download_button(
                        label=f"Download Page {selected_page} Transcription",
                        data=st.session_state.transcription_results[selected_page-1],
                        file_name=f"{uploaded_file.name.split('.')[0]}_page{selected_page}_transcription.txt",
                        mime="text/plain"
                    )

            with tabs[2]:
                st.subheader("Arabic Translation")

                # Radio button to select translation mode
                translation_mode = st.radio(
                    "Translation Mode",
                    ["Translate Complete Document", "Translate Page by Page (More Accurate)"],
                    index=1
                )

                # Button to trigger translation
                if 'translation_processed' not in st.session_state:
                    st.session_state.translation_processed = False
                    st.session_state.arabic_text = ""
                    st.session_state.page_translations = []

                if st.button("Translate to Arabic"):
                    with st.spinner("Translating to Arabic..."):
                        if translation_mode == "Translate Complete Document":
                            # Translate the entire document at once
                            arabic_text = translate_to_arabic(
                                st.session_state.all_text,
                                model_name=model_choice,
                                debug=debug_mode
                            )
                            st.session_state.arabic_text = arabic_text
                            st.session_state.translation_processed = True
                        else:
                            # Translate each page separately for better accuracy
                            page_translations = [None] * len(st.session_state.transcription_results)

                            # Prepare translation data
                            translation_data = [
                                (text, i, len(st.session_state.transcription_results), model_choice, debug_mode)
                                for i, text in enumerate(st.session_state.transcription_results)
                            ]

                            translation_progress = st.progress(0)
                            translation_status = st.empty()
                            translation_status.text("Starting translation of individual pages...")

                            # Process pages in parallel if enabled
                            if parallel_processing:
                                completed = 0
                                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                    future_to_page = {executor.submit(translate_page, data): data for data in translation_data}

                                    for future in concurrent.futures.as_completed(future_to_page):
                                        i, translation = future.result()
                                        page_translations[i] = translation

                                        # Update progress
                                        completed += 1
                                        translation_progress.progress(completed / len(page_translations))
                                        translation_status.text(f"Translated page {i+1}/{len(page_translations)}...")
                            else:
                                # Sequential translation
                                for i, data in enumerate(translation_data):
                                    translation_status.text(f"Translating page {i+1}/{len(translation_data)}...")
                                    _, translation = translate_page(data)
                                    page_translations[i] = translation
                                    translation_progress.progress((i + 1) / len(translation_data))

                            # Combine translations
                            combined_translation = ""
                            for i, translation in enumerate(page_translations):
                                combined_translation += f"\n\n{translation}"

                            st.session_state.arabic_text = combined_translation.strip()
                            st.session_state.page_translations = page_translations
                            st.session_state.translation_processed = True
                            translation_status.text("Translation completed!")

                if st.session_state.translation_processed:
                    # Display the translated text
                    arabic_text = st.session_state.arabic_text
                    st.text_area(
                        "Arabic Translation", 
                        arabic_text,
                        height=500,
                        key="arabic_translation"
                    )

                    # Add Copy button for Arabic translation
                    create_copy_button(arabic_text, "Copy Arabic Translation")

                    # Download button for Arabic translation
                    st.download_button(
                        label="Download Arabic Translation",
                        data=st.session_state.arabic_text,
                        file_name=f"{uploaded_file.name.split('.')[0]}_arabic_translation.txt",
                        mime="text/plain"
                    )

                    # If we translated page by page, show option to view individual pages
                    if translation_mode == "Translate Page by Page (More Accurate)" and hasattr(st.session_state, 'page_translations'):
                        if len(st.session_state.page_translations) > 0:
                            # Page selection for translated content
                            ar_selected_page = st.selectbox(
                                "Select page to view Arabic translation", 
                                range(1, len(st.session_state.page_translations)+1),
                                key="ar_page_selector"
                            )

                            page_ar_text = st.session_state.page_translations[ar_selected_page-1]
                            st.text_area(
                                f"Page {ar_selected_page} Arabic Translation", 
                                page_ar_text, 
                                height=300,
                                key=f"ar_page_{ar_selected_page}"
                            )

                            # Add Copy button for single page Arabic translation
                            create_copy_button(page_ar_text, f"Copy Page {ar_selected_page} Arabic Translation")
                else:
                    st.info("Click 'Translate to Arabic' to generate the Arabic translation.")

            with tabs[3]:
                st.subheader("PDF Export")

                # Select content to export
                export_content = st.radio(
                    "Select content to export as PDF",
                    ["Original Transcription", "Arabic Translation"],
                    key="export_content"
                )

                # Select PDF quality/style
                pdf_style = st.radio(
                    "PDF Export Style",
                    ["Standard", "Academic Journal", "Book Style"],
                    key="pdf_style"
                )

                # Check if we have the selected content available
                content_available = False
                if export_content == "Original Transcription" and hasattr(st.session_state, 'all_text'):
                    content_available = True
                    content_to_export = st.session_state.all_text
                    export_title = f"{uploaded_file.name.split('.')[0]} - Transcription"
                elif export_content == "Arabic Translation" and hasattr(st.session_state, 'translation_processed') and st.session_state.translation_processed:
                    content_available = True
                    content_to_export = st.session_state.arabic_text
                    export_title = f"{uploaded_file.name.split('.')[0]} - Arabic Translation"

                if content_available:
                    if st.button("Generate PDF"):
                        with st.spinner("Generating PDF..."):
                            # Create a temporary file to store the PDF
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                pdf_path = tmp_file.name

                            # Convert markdown to PDF
                            success = markdown_to_pdf(content_to_export, pdf_path, title=export_title)

                            if success:
                                # Read the generated PDF
                                with open(pdf_path, 'rb') as f:
                                    pdf_data = f.read()

                                # Store PDF in session state
                                st.session_state.pdf_data = pdf_data
                                st.session_state.pdf_filename = f"{uploaded_file.name.split('.')[0]}_{export_content.lower().replace(' ', '_')}.pdf"
                                st.session_state.pdf_generated = True

                                # Clean up the temporary file
                                os.unlink(pdf_path)

                                st.success("PDF generated successfully!")
                            else:
                                st.error("Failed to generate PDF. Please try again.")

                    # Download button for PDF (only shown if PDF was generated)
                    if hasattr(st.session_state, 'pdf_generated') and st.session_state.pdf_generated:
                        st.download_button(
                            label=f"Download {export_content} as PDF",
                            data=st.session_state.pdf_data,
                            file_name=st.session_state.pdf_filename,
                            mime="application/pdf"
                        )
                else:
                    if export_content == "Original Transcription":
                        st.info("Please process a PDF document first to generate the transcription.")
                    else:
                        st.info("Please translate the content to Arabic first.")

                st.markdown("""
                #### About PDF Export
                - The PDF export feature converts the markdown-formatted text to a PDF document
                - Mathematical expressions inLaTeX format are rendered properly in the PDF
                - Arabic text is fully supported with right-to-left rendering
                - You can choose to export either the original transcription or the Arabic translation
                """)
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
        6. Translate to Arabic if needed
        7. Export to PDF with proper formatting
        """)

    with col2:
        st.markdown("""
        ### Tips for best results:
        - Use PDFs with clear, readable text
        - Larger files may take more time to process
        - If you encounter rate limits, wait a few minutes and try again
        - For multi-page PDFs, each page is processed individually
        - Enable parallel processing for faster results
        - Enable debug mode to troubleshoot API issues
        - For better translation quality, use the page-by-page option
        """)

    # API key information box
    st.markdown("""
    ### Getting a Google Gemini API Key
    1. Go to https://aistudio.google.com/
    2. Create a Google account if you don't have one
    3. Access the API section and create a key
    4. Copy the API key and paste it in the sidebar
    """)