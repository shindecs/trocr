import os
import sys
import torch
import pandas as pd
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Setup device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Running on device: {device}", flush=True)

# In-memory cache for loaded processors and models
models_cache = {}

def get_model_and_processor(model_type):
    """
    Load and return the processor and model for the selected text type.
    Caches the objects in memory to avoid reloading on subsequent requests.
    """
    model_name = (
        "microsoft/trocr-base-printed"
        if model_type == "Printed"
        else "microsoft/trocr-base-handwritten"
    )
    
    if model_name not in models_cache:
        print(f"[*] Loading TrOCR model and processor for: {model_name}...", flush=True)
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        models_cache[model_name] = (processor, model)
        print(f"[*] Model {model_name} loaded successfully.", flush=True)
        
    return models_cache[model_name]

def generate_sample_images():
    """
    Helper function to generate sample text images on startup
    so the user has examples to play with right away.
    """
    os.makedirs("samples", exist_ok=True)
    
    # Try to load a clean system font, fall back to default PIL font if not found
    font = None
    italic_font = None
    font_paths_to_try = [
        "arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    italic_font_paths_to_try = [
        "timesi.ttf",
        "C:\\Windows\\Fonts\\timesi.ttf",
        "LiberationSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"
    ]
    
    for path in font_paths_to_try:
        try:
            font = ImageFont.truetype(path, 24)
            break
        except IOError:
            continue
            
    for path in italic_font_paths_to_try:
        try:
            italic_font = ImageFont.truetype(path, 26)
            break
        except IOError:
            continue
            
    if font is None:
        font = ImageFont.load_default()
    if italic_font is None:
        italic_font = ImageFont.load_default()
        
    # 1. Printed style sample
    img_printed = Image.new("RGB", (600, 100), color="white")
    draw = ImageDraw.Draw(img_printed)
    draw.text((25, 35), "This is a clean sample of printed text.", fill="black", font=font)
    img_printed.save("samples/printed_sample.png")
    
    # 2. Handwritten/Italic style sample
    img_handwritten = Image.new("RGB", (600, 100), color="#FAF7F2")
    draw = ImageDraw.Draw(img_handwritten)
    draw.text((25, 35), "Solve the problem using deep learning.", fill="#2B3A67", font=italic_font)
    img_handwritten.save("samples/handwritten_sample.png")
    print("[*] Sample images generated in 'samples/'.", flush=True)

def analyze_document(image, model_type, progress=gr.Progress(track_tqdm=True)):
    """
    Runs PaddleOCR to detect layout and recognize printed text in all segments.
    Crops each detected region, runs TrOCR in batches to recognize handwriting,
    and returns annotated image and summary table.
    """
    if image is None:
        return None, pd.DataFrame(), [], gr.update(choices=[]), "Error: Please upload an image first."
        
    try:
        progress(0.0, desc="Converting image...")
        orig_w, orig_h = image.size
        print(f"[*] Original image size: {orig_w}x{orig_h}", flush=True)
        
        # Resize to a max dimension of 1000px for layout analysis speed on CPU
        max_dim = 1000
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            print(f"[*] Resized image to: {new_w}x{new_h} (scale factor: {scale:.4f})", flush=True)
        else:
            resized_img = image
            scale = 1.0
            print("[*] Image is under 1000px; no resizing needed.", flush=True)
            
        # Convert resized image to numpy array for PaddleOCR
        img_np = np.array(resized_img)
        
        progress(0.1, desc="Initializing PaddleOCR...")
        # Initialize PaddleOCR on startup (lazy loading)
        from paddleocr import PaddleOCR
        print("[*] Initializing PaddleOCR...", flush=True)
        paddle_ocr = PaddleOCR(
            lang='en',
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        
        progress(0.2, desc="Running layout analysis (PaddleOCR)...")
        print("[*] Running PaddleOCR detection and recognition...", flush=True)
        result = paddle_ocr.predict(img_np)
        
        if not result or not isinstance(result, list) or len(result) == 0:
            status = "No text boxes detected in the image."
            print(f"[*] {status}", flush=True)
            return image, pd.DataFrame(columns=["Box ID", "Printed OCR", "Handwritten OCR (TrOCR)", "Confirmed Text"]), [], gr.update(choices=[]), status
            
        data = result[0]
        polys = data.get('dt_polys', [])
        texts = data.get('rec_texts', [])
        
        if not polys:
            status = "No text boxes detected in the image."
            print(f"[*] {status}", flush=True)
            return image, pd.DataFrame(columns=["Box ID", "Printed OCR", "Handwritten OCR (TrOCR)", "Confirmed Text"]), [], gr.update(choices=[]), status
            
        progress(0.3, desc=f"Detected {len(polys)} text regions. Initializing TrOCR...")
        # Initialize TrOCR (lazy loading/caching)
        processor, model = get_model_and_processor(model_type)
        
        boxes_data = []
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        try:
            # Scale the font size relative to the original image dimensions
            font_size = max(16, int(max(orig_w, orig_h) * 0.008))
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
        print(f"[*] Preprocessing crops for {len(polys)} text boxes...", flush=True)
        progress(0.4, desc="Cropping text regions...")
        
        valid_crops = []
        valid_indices = []
        box_coords = []
        
        for idx, poly in enumerate(polys):
            # Scale coordinates back to original resolution for high-quality crops
            orig_poly = poly / scale
            
            x_coords = [int(p[0]) for p in orig_poly]
            y_coords = [int(p[1]) for p in orig_poly]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            
            # Ensure dimensions are positive and within original image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(image.width, xmax)
            ymax = min(image.height, ymax)
            
            box_coords.append((xmin, ymin, xmax, ymax))
            
            if xmax > xmin and ymax > ymin:
                crop_img = image.crop((xmin, ymin, xmax, ymax))
                valid_crops.append(crop_img)
                valid_indices.append(idx)
                
        # Batched TrOCR inference
        trocr_texts = [""] * len(polys)
        if valid_crops:
            batch_size = 8
            num_batches = int(np.ceil(len(valid_crops) / batch_size))
            print(f"[*] Running batched TrOCR inference on {len(valid_crops)} crops (batch size: {batch_size})...", flush=True)
            
            for i in range(0, len(valid_crops), batch_size):
                batch = valid_crops[i:i + batch_size]
                indices = valid_indices[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                progress_pct = 0.4 + 0.5 * (batch_num / num_batches)
                progress(progress_pct, desc=f"Running TrOCR (Batch {batch_num}/{num_batches})...")
                
                try:
                    # Preprocess batch
                    inputs = processor(images=batch, return_tensors="pt", padding=True)
                    pixel_values = inputs.pixel_values.to(device)
                    
                    # Generate transcriptions
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values)
                        
                    # Decode transcriptions
                    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Store results
                    for idx_in_batch, decoded_text in enumerate(decoded):
                        global_idx = indices[idx_in_batch]
                        trocr_texts[global_idx] = decoded_text.strip()
                except Exception as e:
                    print(f"[!] Error in TrOCR batch {batch_num}: {e}", flush=True)
                    for global_idx in indices:
                        trocr_texts[global_idx] = "[OCR Error]"
                        
        progress(0.95, desc="Rendering results...")
        df_rows = []
        choices = []
        
        for idx in range(len(polys)):
            box_id = idx + 1
            xmin, ymin, xmax, ymax = box_coords[idx]
            printed_clean = texts[idx].strip() if idx < len(texts) else ""
            trocr_clean = trocr_texts[idx].strip()
            
            crop_img = None
            if xmax > xmin and ymax > ymin:
                crop_img = image.crop((xmin, ymin, xmax, ymax))
                
            confirmed_text = printed_clean
            
            boxes_data.append({
                "id": box_id,
                "box": [xmin, ymin, xmax, ymax],
                "printed": printed_clean,
                "handwritten": trocr_clean,
                "confirmed": confirmed_text,
                "crop": crop_img
            })
            
            # Scale line width of bounding boxes based on original image dimensions
            line_width = max(2, int(max(orig_w, orig_h) * 0.002))
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=line_width)
            
            # Scale ID tag background and text placement
            tag_height = int(font_size * 1.25)
            tag_width = int(font_size * 1.5)
            draw.rectangle([xmin, ymin - tag_height, xmin + tag_width, ymin], fill="red")
            draw.text((xmin + 4, ymin - tag_height + 2), str(box_id), fill="white", font=font)
            
            df_rows.append([box_id, printed_clean, trocr_clean, confirmed_text])
            choices.append(f"Box {box_id}")
            
        df = pd.DataFrame(df_rows, columns=["Box ID", "Printed OCR", "Handwritten OCR (TrOCR)", "Confirmed Text"])
        status = f"Analysis complete! Detected {len(boxes_data)} text regions."
        print(f"[*] {status}", flush=True)
        
        progress(1.0, desc="Complete!")
        return annotated_img, df, boxes_data, gr.update(choices=choices, value=choices[0] if choices else None), status
        
    except Exception as e:
        import traceback
        err_msg = f"Error in analyze_document:\n{str(e)}\n\n{traceback.format_exc()}"
        print(f"[!] {err_msg}", flush=True)
        return image, pd.DataFrame(), [], gr.update(choices=[]), err_msg

def load_box_details(selected_box_name, boxes_data):
    """
    Triggered when dropdown box ID changes. Loads corresponding crop and text.
    """
    if not selected_box_name or not boxes_data:
        return None, "", "", ""
        
    try:
        box_id = int(selected_box_name.replace("Box ", ""))
        box_item = next((item for item in boxes_data if item["id"] == box_id), None)
        if box_item:
            return box_item["crop"], box_item["printed"], box_item["handwritten"], box_item["confirmed"]
    except Exception as e:
        print(f"[!] Error loading box details: {e}", flush=True)
        
    return None, "", "", ""

def update_box_transcription(selected_box_name, new_val, boxes_data):
    """
    Updates the confirmed text for the active box and refreshes the summary table.
    """
    if not selected_box_name or not boxes_data:
        return pd.DataFrame(), boxes_data
        
    try:
        box_id = int(selected_box_name.replace("Box ", ""))
        for item in boxes_data:
            if item["id"] == box_id:
                item["confirmed"] = new_val
                break
                
        # Rebuild DataFrame
        df_rows = []
        for item in boxes_data:
            df_rows.append([item["id"], item["printed"], item["handwritten"], item["confirmed"]])
            
        df = pd.DataFrame(df_rows, columns=["Box ID", "Printed OCR", "Handwritten OCR (TrOCR)", "Confirmed Text"])
        return df, boxes_data
    except Exception as e:
        print(f"[!] Error updating box: {e}", flush=True)
        return pd.DataFrame(), boxes_data

def export_to_csv(boxes_data):
    """
    Exports the verified text table to a CSV file.
    """
    if not boxes_data:
        return None
    try:
        df_rows = []
        for item in boxes_data:
            df_rows.append([item["id"], item["printed"], item["handwritten"], item["confirmed"]])
        df = pd.DataFrame(df_rows, columns=["Box ID", "Printed OCR", "Handwritten OCR (TrOCR)", "Confirmed Text"])
        
        file_path = "ocr_results_export.csv"
        df.to_csv(file_path, index=False)
        print(f"[*] Exported results to {file_path}", flush=True)
        return file_path
    except Exception as e:
        print(f"[!] Error exporting CSV: {e}", flush=True)
        return None

# Build the Gradio interface
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
).set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    block_border_width="1px",
    block_border_color="*neutral_200",
    button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_600, *secondary_600)",
    button_primary_text_color="white"
)

custom_css = """
.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
    padding-top: 2rem;
}
.header-area {
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border-color-primary);
    padding-bottom: 1.5rem;
}
"""

with gr.Blocks(title="Layout-Aware Hybrid OCR Detector") as demo:
    # Header Section
    gr.HTML(
        """
        <div class="header-area" style="text-align: center;">
            <h1 style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;">
                Layout-Aware Segmented OCR
            </h1>
            <p style="color: #64748b; font-size: 1.15rem; max-width: 750px; margin: 0 auto; line-height: 1.6;">
                Designed for complex documents and mixed-layout medical charts. 
                Runs <b>PaddleOCR</b> to segment text lines/cells and transcribe printed labels, 
                then runs <b>TrOCR</b> on each segment to transcribe handwriting.
            </p>
        </div>
        """
    )
    
    # Session State
    boxes_store = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown("### 📷 Upload Document")
            input_image = gr.Image(type="pil", label="Upload chart, form, or document")
            
            with gr.Row():
                model_type = gr.Radio(
                    choices=["Printed", "Handwritten"],
                    value="Handwritten",
                    label="TrOCR Model",
                    info="Select target model for TrOCR evaluation on cropped regions."
                )
                
            analyze_btn = gr.Button("Analyze Document", variant="primary")
            status_output = gr.Textbox(label="Status / Log Output", interactive=False)
            
        with gr.Column(scale=12):
            gr.Markdown("### 🔍 Annotated Document Overview")
            annotated_image_display = gr.Image(label="Segmented Regions (Red Bounding Boxes)", type="pil", interactive=False)
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Overview Table")
            summary_table = gr.Dataframe(
                headers=["Box ID", "Printed OCR", "Handwritten OCR (TrOCR)", "Confirmed Text"],
                datatype=["number", "str", "str", "str"],
                label="All Detected Text Regions",
                interactive=False
            )
            
            with gr.Row():
                export_btn = gr.Button("Export to CSV", variant="secondary")
                download_file = gr.File(label="Download CSV Results File", interactive=False)

    with gr.Row(variant="panel"):
        with gr.Column(scale=8):
            gr.Markdown("### 🔬 Region Validator & Inspector")
            box_selector = gr.Dropdown(
                choices=[],
                label="Select Box ID to Inspect",
                info="Choose a detected text box number from the dropdown to zoom in and correct it."
            )
            
            crop_display = gr.Image(
                label="Cropped Region Snippet",
                type="pil",
                interactive=False,
                height=120
            )
            
        with gr.Column(scale=12):
            gr.Markdown("### 📝 Edit & Validate Text")
            with gr.Row():
                printed_suggestion = gr.Textbox(label="Printed OCR Suggestion (Paddle)", interactive=False)
                handwritten_suggestion = gr.Textbox(label="Handwritten OCR Suggestion (TrOCR)", interactive=False)
                
            confirmed_textbox = gr.Textbox(
                label="Edit Confirmed Value", 
                placeholder="Modify the value here..."
            )
            
            with gr.Row():
                use_printed_btn = gr.Button("Accept Printed", size="sm")
                use_handwritten_btn = gr.Button("Accept Handwritten", size="sm")
                save_correction_btn = gr.Button("Save Correction", variant="primary", size="sm")

    # Set up sample images
    generate_sample_images()
    
    gr.Markdown("### 💡 Quick Examples")
    gr.Examples(
        examples=[
            ["samples/printed_sample.png", "Printed"],
            ["samples/handwritten_sample.png", "Handwritten"]
        ],
        inputs=[input_image, model_type],
        outputs=[annotated_image_display, summary_table, boxes_store, box_selector, status_output],
        fn=analyze_document,
        cache_examples=False
    )
    
    # Event Listeners
    analyze_btn.click(
        fn=analyze_document,
        inputs=[input_image, model_type],
        outputs=[annotated_image_display, summary_table, boxes_store, box_selector, status_output]
    )
    
    box_selector.change(
        fn=load_box_details,
        inputs=[box_selector, boxes_store],
        outputs=[crop_display, printed_suggestion, handwritten_suggestion, confirmed_textbox]
    )
    
    use_printed_btn.click(
        fn=lambda printed: printed,
        inputs=[printed_suggestion],
        outputs=[confirmed_textbox]
    ).then(
        fn=update_box_transcription,
        inputs=[box_selector, confirmed_textbox, boxes_store],
        outputs=[summary_table, boxes_store]
    )
    
    use_handwritten_btn.click(
        fn=lambda handwritten: handwritten,
        inputs=[handwritten_suggestion],
        outputs=[confirmed_textbox]
    ).then(
        fn=update_box_transcription,
        inputs=[box_selector, confirmed_textbox, boxes_store],
        outputs=[summary_table, boxes_store]
    )
    
    save_correction_btn.click(
        fn=update_box_transcription,
        inputs=[box_selector, confirmed_textbox, boxes_store],
        outputs=[summary_table, boxes_store]
    )
    
    export_btn.click(
        fn=export_to_csv,
        inputs=[boxes_store],
        outputs=[download_file]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme, css=custom_css)
