import os
import sys
import torch
import pandas as pd
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import base64
import requests
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()

# Setup device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Running on device: {device}", flush=True)

# In-memory cache for loaded processors and models
models_cache = {}

def load_custom_env():
    """
    Manually parses the .env file to support JSON-like formats
    such as '"ApiKey": "value",' or standard dotenv format,
    while grouping Azure CS credentials.
    """
    env_vars = {}
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                in_azure_cs = False
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "Azure CS" in line:
                        in_azure_cs = True
                        continue
                    
                    key, val = None, None
                    if "=" in line:
                        parts = line.split("=", 1)
                        key = parts[0].strip().strip('"').strip("'")
                        val = parts[1].strip().strip('"').strip("'")
                    elif ":" in line:
                        parts = line.split(":", 1)
                        key = parts[0].strip().strip('"').strip("'")
                        val = parts[1].strip().rstrip(",").strip().strip('"').strip("'")
                    
                    if key and val:
                        if in_azure_cs:
                            env_vars[f"azure_{key}"] = val
                        else:
                            env_vars[key] = val
        except Exception as e:
            print(f"[!] Error reading .env manually: {e}", flush=True)
    return env_vars

def call_azure_computer_vision_ocr(img, endpoint, api_key):
    """
    Performs OCR using Azure Computer Vision.
    First tries the synchronous v4.0 Image Analysis API.
    If that fails or returns 404, falls back to the asynchronous v3.2 Read API.
    """
    import io
    import time
    
    img_byte_arr = io.BytesIO()
    img.convert("RGB").save(img_byte_arr, format='JPEG', quality=90)
    image_bytes = img_byte_arr.getvalue()
    
    base_url = endpoint.strip().rstrip("/")
    
    # Try Azure AI Vision v4.0 (Synchronous)
    v4_url = f"{base_url}/computervision/imageanalysis:analyze?api-version=2023-10-01&features=read"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/octet-stream"
    }
    
    print(f"[*] Trying Azure Computer Vision v4.0 Image Analysis API...", flush=True)
    try:
        response = requests.post(v4_url, headers=headers, data=image_bytes, timeout=30)
        if response.status_code == 200:
            res_json = response.json()
            lines = []
            if "readResult" in res_json and "blocks" in res_json["readResult"]:
                for block in res_json["readResult"]["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "text" in line:
                                lines.append(line["text"])
            if lines:
                return "\n".join(lines).strip()
            print("[*] v4.0 returned success but no text blocks found.", flush=True)
        else:
            print(f"[!] v4.0 API failed with status {response.status_code}: {response.text}", flush=True)
    except Exception as e:
        print(f"[!] Exception calling v4.0 API: {e}", flush=True)

    # Fallback to Azure Computer Vision v3.2 Read API (Asynchronous)
    print(f"[*] Falling back to Azure Computer Vision v3.2 Read API...", flush=True)
    v3_url = f"{base_url}/vision/v3.2/read/analyze"
    try:
        response = requests.post(v3_url, headers=headers, data=image_bytes, timeout=30)
        if response.status_code != 202:
            return f"[Azure API Error: HTTP {response.status_code} - {response.text}]"
            
        operation_url = response.headers.get("Operation-Location")
        if not operation_url:
            return "[Azure API Error: No Operation-Location header found in response]"
            
        headers_get = {"Ocp-Apim-Subscription-Key": api_key}
        for attempt in range(20):
            time.sleep(1.0)
            get_res = requests.get(operation_url, headers=headers_get, timeout=10)
            if get_res.status_code != 200:
                continue
            get_json = get_res.json()
            status = get_json.get("status")
            if status == "succeeded":
                lines = []
                analyze_result = get_json.get("analyzeResult", {})
                read_results = analyze_result.get("readResults", [])
                for result in read_results:
                    for line in result.get("lines", []):
                        lines.append(line.get("text", ""))
                return "\n".join(lines).strip()
            elif status == "failed":
                return f"[Azure OCR v3.2 failed: {get_json}]"
        return "[Azure OCR v3.2 timeout]"
    except Exception as e:
        print(f"[!] Exception calling v3.2 API: {e}", flush=True)
        return f"[Azure OCR Error: {str(e)}]"

def call_openai_api(prompt, api_url, api_key, model_name="gpt-5"):
    """
    Sends a chat completion request to the custom OpenAI API endpoint.
    Supports both standard chat completions and the new Responses API.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    url = api_url.strip()
    is_responses_api = "/responses" in url
    
    if is_responses_api:
        payload = {
            "model": model_name,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    else:
        # Fallback to standard chat completions
        if "/chat/completions" not in url:
            url = url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        
    print(f"[*] Sending request to OpenAI API (URL: {url}, Model: {model_name}, ResponsesAPI: {is_responses_api})...", flush=True)
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    res_json = response.json()
    
    if is_responses_api and "output" in res_json:
        for out in res_json["output"]:
            if out.get("type") == "message" and "content" in out:
                for c in out["content"]:
                    if c.get("type") == "output_text" and "text" in c:
                        return c["text"]
        for out in res_json["output"]:
            if "content" in out and isinstance(out["content"], list):
                for c in out["content"]:
                    if isinstance(c, dict) and "text" in c:
                        return c["text"]
                        
    if "choices" in res_json and len(res_json["choices"]) > 0:
        choice = res_json["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
            
    if "content" in res_json:
        return res_json["content"]
        
    return str(res_json)

def analyze_full_page_azure(file_paths, progress=gr.Progress(track_tqdm=True)):
    if not file_paths:
        return [], None, pd.DataFrame(columns=["Page ID", "Transcribed Text"]), gr.update(choices=[]), "Error: Please upload image files first."
        
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
        
    try:
        progress(0.05, desc="Loading environment credentials...")
        custom_env = load_custom_env()
        azure_url = custom_env.get("azure_url")
        azure_key = custom_env.get("azure_key")
        
        if not azure_url or not azure_key:
            return [], None, pd.DataFrame(columns=["Page ID", "Transcribed Text"]), gr.update(choices=[]), "Error: Azure CS url or key not found in .env under 'Azure CS:'"
            
        azure_pages_data = []
        df_rows = []
        choices = []
        total_pages = len(file_paths)
        
        for idx, file in enumerate(file_paths):
            page_num = idx + 1
            progress_pct = 0.1 + 0.8 * (idx / total_pages)
            progress(progress_pct, desc=f"Processing page {page_num}/{total_pages} with Azure CV...")
            
            if isinstance(file, str):
                file_path = file
            elif hasattr(file, "name") and file.name:
                file_path = file.name
            elif isinstance(file, dict) and "name" in file and file["name"]:
                file_path = file["name"]
            elif hasattr(file, "path") and file.path:
                file_path = file.path
            elif isinstance(file, dict) and "path" in file and file["path"]:
                file_path = file["path"]
            else:
                file_path = str(file)
                
            print(f"[*] Loading page {page_num} for Azure OCR: {file_path}", flush=True)
            image = Image.open(file_path).convert("RGB")
            
            max_dim = 2000
            if max(image.width, image.height) > max_dim:
                scale = max_dim / max(image.width, image.height)
                resized_img = image.resize(
                    (int(image.width * scale), int(image.height * scale)), 
                    Image.Resampling.BILINEAR
                )
            else:
                resized_img = image
                
            print(f"[*] Sending page {page_num} to Azure Computer Vision OCR...", flush=True)
            text = call_azure_computer_vision_ocr(resized_img, azure_url, azure_key)
            
            azure_pages_data.append({
                "id": page_num,
                "image": image,
                "transcription": text
            })
            
            df_rows.append([page_num, text])
            choices.append(f"Page {page_num}")
            
            if idx < total_pages - 1:
                import time
                time.sleep(1.0)
                
        df = pd.DataFrame(df_rows, columns=["Page ID", "Transcribed Text"])
        status = f"Analysis complete! Processed {len(azure_pages_data)} pages using Azure Computer Vision OCR."
        progress(1.0, desc="Complete!")
        return (
            azure_pages_data,
            azure_pages_data[0]["image"] if azure_pages_data else None,
            df,
            gr.update(choices=choices, value=choices[0] if choices else None),
            status
        )
    except Exception as e:
        import traceback
        err_msg = f"Error during Azure OCR:\n{str(e)}\n\n{traceback.format_exc()}"
        print(f"[!] {err_msg}", flush=True)
        return [], None, pd.DataFrame(columns=["Page ID", "Transcribed Text"]), gr.update(choices=[]), err_msg

def generate_discharge_card_with_gpt5(pages_data, progress=gr.Progress(track_tqdm=True)):
    if not pages_data:
        return "Error: No transcribed pages found. Please run Azure OCR on some documents first."
        
    try:
        progress(0.1, desc="Formatting OCR data...")
        combined_text_list = []
        for item in pages_data:
            combined_text_list.append(f"--- Page {item['id']} ---\n{item['transcription']}")
        combined_ocr_text = "\n\n".join(combined_text_list)
        
        progress(0.2, desc="Loading environment credentials...")
        custom_env = load_custom_env()
        api_key = custom_env.get("ApiKey")
        api_url = custom_env.get("ApiUrl", "https://api.openai.com/v1/responses")
        model_name = custom_env.get("Model", "gpt-5")
        
        if not api_key:
            return "Error: 'ApiKey' not found in .env file."
            
        progress(0.3, desc="Formulating clinical prompt...")
        prompt = f"""You are an expert clinical medical documentation assistant.
Your task is to take raw OCR transcription from hospital chart pages, clinical notes, and reports, and synthesize a professional, highly readable Discharge Card (Discharge Summary).

Provide the output in a clean Markdown format with the following sections:
1. **Patient Information** (Name, Age, Gender, Patient ID/Reg No if found)
2. **Admission & Discharge Details** (Dates if found)
3. **Primary Diagnosis & Comorbidities**
4. **Brief Clinical History & Course in Hospital**
5. **Key Investigation Findings** (Lab results, imaging)
6. **Treatment Given** (Procedures, surgeries, key medications in hospital)
7. **Discharge Medications** (Names, dosage, frequency, duration)
8. **Follow-up Instructions & Warning Signs** (When to return to the ER)

Here is the raw OCR text extracted from the document:
---
{combined_ocr_text}
---

Generate the structured Discharge Card in clear, well-formatted Markdown. Use bullet points and bold headers for clarity. If any information is missing or not mentioned in the OCR text, omit that sub-item or mark it as 'Not Specified'. Do not hallucinate any patient names or clinical values."""

        progress(0.5, desc="Calling GPT-5 for Discharge Card generation...")
        card_content = call_openai_api(prompt, api_url, api_key, model_name=model_name)
        progress(1.0, desc="Complete!")
        return card_content
    except Exception as e:
        import traceback
        err_msg = f"Error generating discharge card: {str(e)}\n\n{traceback.format_exc()}"
        print(f"[!] {err_msg}", flush=True)
        return err_msg

def load_azure_page_details(selected_page, azure_pages_data):
    if not selected_page or not azure_pages_data:
        return None, ""
    try:
        page_num = int(selected_page.replace("Page ", ""))
        item = next((p for p in azure_pages_data if p["id"] == page_num), None)
        if item:
            return item["image"], item["transcription"]
    except Exception as e:
        print(f"[!] Error loading Azure page details: {e}", flush=True)
    return None, ""

def update_azure_page_transcription(selected_page, new_val, azure_pages_data):
    if not selected_page or not azure_pages_data:
        return pd.DataFrame(columns=["Page ID", "Transcribed Text"]), azure_pages_data
    try:
        page_num = int(selected_page.replace("Page ", ""))
        for item in azure_pages_data:
            if item["id"] == page_num:
                item["transcription"] = new_val
                break
        df_rows = []
        for item in azure_pages_data:
            df_rows.append([item["id"], item["transcription"]])
        df = pd.DataFrame(df_rows, columns=["Page ID", "Transcribed Text"])
        return df, azure_pages_data
    except Exception as e:
        print(f"[!] Error updating Azure page transcription: {e}", flush=True)
        return pd.DataFrame(columns=["Page ID", "Transcribed Text"]), azure_pages_data

def export_azure_pages_to_csv(azure_pages_data):
    if not azure_pages_data:
        return None
    try:
        df_rows = []
        for item in azure_pages_data:
            df_rows.append([item["id"], item["transcription"]])
        df = pd.DataFrame(df_rows, columns=["Page ID", "Transcribed Text"])
        file_path = "azure_full_page_results.csv"
        df.to_csv(file_path, index=False)
        print(f"[*] Exported Azure page results to {file_path}", flush=True)
        return file_path
    except Exception as e:
        print(f"[!] Error exporting Azure CSV: {e}", flush=True)
        return None

def call_mistral_ocr(crop_img, api_url, api_key):
    """
    Performs OCR on an image using the Mistral Document AI API on Azure.
    """
    try:
        import io
        img_byte_arr = io.BytesIO()
        crop_img.convert("RGB").save(img_byte_arr, format='JPEG', quality=90)
        image_bytes = img_byte_arr.getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "mistral-document-ai-2505",
            "document": {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        
        res_json = response.json()
        if "pages" in res_json and len(res_json["pages"]) > 0:
            text = res_json["pages"][0].get("markdown", "").strip()
            return text
        elif "markdown" in res_json:
            return res_json["markdown"].strip()
        elif "text" in res_json:
            return res_json["text"].strip()
        else:
            if isinstance(res_json, str):
                return res_json.strip()
            return str(res_json)
    except Exception as e:
        print(f"[!] Error in call_mistral_ocr: {e}", flush=True)
        return f"[OCR Error: {str(e)}]"

def analyze_full_page_mistral(file_paths, progress=gr.Progress(track_tqdm=True)):
    if not file_paths:
        return [], None, pd.DataFrame(columns=["Page ID", "Transcribed Text"]), gr.update(choices=[]), "Error: Please upload image files first."
        
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
        
    try:
        progress(0.05, desc="Loading environment credentials...")
        custom_env = load_custom_env()
        mistral_url = custom_env.get("url")
        mistral_key = custom_env.get("key")
        
        if not mistral_url or not mistral_key:
            return [], None, pd.DataFrame(columns=["Page ID", "Transcribed Text"]), gr.update(choices=[]), "Error: Mistral url or key not found in .env"
            
        full_pages_data = []
        df_rows = []
        choices = []
        total_pages = len(file_paths)
        
        for idx, file in enumerate(file_paths):
            page_num = idx + 1
            progress_pct = 0.1 + 0.8 * (idx / total_pages)
            progress(progress_pct, desc=f"Processing page {page_num}/{total_pages} (one by one)...")
            
            # Extract file path
            if isinstance(file, str):
                file_path = file
            elif hasattr(file, "name") and file.name:
                file_path = file.name
            elif isinstance(file, dict) and "name" in file and file["name"]:
                file_path = file["name"]
            elif hasattr(file, "path") and file.path:
                file_path = file.path
            elif isinstance(file, dict) and "path" in file and file["path"]:
                file_path = file["path"]
            else:
                file_path = str(file)
                
            print(f"[*] Loading page {page_num}: {file_path}", flush=True)
            image = Image.open(file_path).convert("RGB")
            
            max_dim = 2000
            if max(image.width, image.height) > max_dim:
                scale = max_dim / max(image.width, image.height)
                resized_img = image.resize(
                    (int(image.width * scale), int(image.height * scale)), 
                    Image.Resampling.BILINEAR
                )
            else:
                resized_img = image
                
            print(f"[*] Sending page {page_num} to Mistral OCR API...", flush=True)
            text = call_mistral_ocr(resized_img, mistral_url, mistral_key)
            
            full_pages_data.append({
                "id": page_num,
                "image": image,
                "transcription": text
            })
            
            df_rows.append([page_num, text])
            choices.append(f"Page {page_num}")
            
            if idx < total_pages - 1:
                import time
                print("[*] Sleeping 3 seconds to avoid rate limiting...", flush=True)
                time.sleep(3.0)
                
        df = pd.DataFrame(df_rows, columns=["Page ID", "Transcribed Text"])
        status = f"Analysis complete! Processed {len(full_pages_data)} pages sequentially."
        progress(1.0, desc="Complete!")
        return (
            full_pages_data,
            full_pages_data[0]["image"] if full_pages_data else None,
            df,
            gr.update(choices=choices, value=choices[0] if choices else None),
            status
        )
    except Exception as e:
        import traceback
        err_msg = f"Error during Mistral raw page OCR:\n{str(e)}\n\n{traceback.format_exc()}"
        print(f"[!] {err_msg}", flush=True)
        return [], None, pd.DataFrame(columns=["Page ID", "Transcribed Text"]), gr.update(choices=[]), err_msg

def load_full_page_details(selected_page, full_pages_data):
    if not selected_page or not full_pages_data:
        return None, ""
    try:
        page_num = int(selected_page.replace("Page ", ""))
        item = next((p for p in full_pages_data if p["id"] == page_num), None)
        if item:
            return item["image"], item["transcription"]
    except Exception as e:
        print(f"[!] Error loading page details: {e}", flush=True)
    return None, ""

def update_full_page_transcription(selected_page, new_val, full_pages_data):
    if not selected_page or not full_pages_data:
        return pd.DataFrame(columns=["Page ID", "Transcribed Text"]), full_pages_data
    try:
        page_num = int(selected_page.replace("Page ", ""))
        for item in full_pages_data:
            if item["id"] == page_num:
                item["transcription"] = new_val
                break
        df_rows = []
        for item in full_pages_data:
            df_rows.append([item["id"], item["transcription"]])
        df = pd.DataFrame(df_rows, columns=["Page ID", "Transcribed Text"])
        return df, full_pages_data
    except Exception as e:
        print(f"[!] Error updating page transcription: {e}", flush=True)
        return pd.DataFrame(columns=["Page ID", "Transcribed Text"]), full_pages_data

def export_full_pages_to_csv(full_pages_data):
    if not full_pages_data:
        return None
    try:
        df_rows = []
        for item in full_pages_data:
            df_rows.append([item["id"], item["transcription"]])
        df = pd.DataFrame(df_rows, columns=["Page ID", "Transcribed Text"])
        file_path = "mistral_full_page_results.csv"
        df.to_csv(file_path, index=False)
        print(f"[*] Exported full page results to {file_path}", flush=True)
        return file_path
    except Exception as e:
        print(f"[!] Error exporting full page CSV: {e}", flush=True)
        return None

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
    
    with gr.Tabs():
        with gr.Tab("Segmented OCR"):
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

        with gr.Tab("Full-Page Mistral OCR"):
            gr.HTML(
                """
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h2 style="font-size: 2rem; font-weight: 700; color: #4f46e5;">Raw Full-Page Mistral OCR</h2>
                    <p style="color: #64748b; font-size: 1rem; max-width: 600px; margin: 0 auto;">
                        Bypasses segmentation (PaddleOCR) and character cropping. 
                        Sends raw full images directly to Azure AI's Mistral Document AI OCR API sequentially.
                    </p>
                </div>
                """
            )
            
            full_pages_store = gr.State([])
            
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("### 📷 Upload Document Pages")
                    input_files = gr.File(file_count="multiple", file_types=["image"], label="Upload Document Pages (Images)", type="filepath")
                    run_btn = gr.Button("Run Mistral OCR", variant="primary")
                    mistral_status = gr.Textbox(label="Status / Log Output", interactive=False)
                    
                with gr.Column(scale=12):
                    gr.Markdown("### 📄 Document Preview")
                    selected_page_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select Page to View",
                        info="Choose a page to see its raw image and transcribed text."
                    )
                    full_image_display = gr.Image(label="Raw Page Image", type="pil", interactive=False)
                    
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Transcribed Document Pages")
                    mistral_summary_table = gr.Dataframe(
                        headers=["Page ID", "Transcribed Text"],
                        datatype=["number", "str"],
                        label="All Page Transcriptions",
                        interactive=False
                    )
                    
                    with gr.Row():
                        mistral_export_btn = gr.Button("Export to CSV", variant="secondary")
                        mistral_download_file = gr.File(label="Download CSV Results File", interactive=False)
            
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown("### 📝 Edit & Validate Text")
                    mistral_confirmed_textbox = gr.Textbox(
                        label="Edit Confirmed Page Transcription", 
                        placeholder="Modify the transcription here...",
                        lines=10
                    )
                    mistral_save_correction_btn = gr.Button("Save Page Correction", variant="primary")

        with gr.Tab("Azure CV Full-Page OCR"):
            gr.HTML(
                """
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h2 style="font-size: 2rem; font-weight: 700; color: #0284c7;">Azure CV Full-Page OCR & Discharge Card</h2>
                    <p style="color: #64748b; font-size: 1rem; max-width: 600px; margin: 0 auto;">
                        Performs OCR on raw full images using Microsoft Azure Computer Vision.
                        Includes option to generate a medical Discharge Card using GPT-5.
                    </p>
                </div>
                """
            )
            
            azure_pages_store = gr.State([])
            
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("### 📷 Upload Document Pages")
                    azure_input_files = gr.File(file_count="multiple", file_types=["image"], label="Upload Document Pages (Images)", type="filepath")
                    azure_run_btn = gr.Button("Run Azure OCR", variant="primary")
                    azure_status = gr.Textbox(label="Status / Log Output", interactive=False)
                    
                with gr.Column(scale=12):
                    gr.Markdown("### 📄 Document Preview")
                    azure_selected_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select Page to View",
                        info="Choose a page to see its raw image and transcribed text."
                    )
                    azure_image_display = gr.Image(label="Raw Page Image", type="pil", interactive=False)
                    
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Transcribed Document Pages")
                    azure_summary_table = gr.Dataframe(
                        headers=["Page ID", "Transcribed Text"],
                        datatype=["number", "str"],
                        label="All Page Transcriptions",
                        interactive=False
                    )
                    
                    with gr.Row():
                        azure_export_btn = gr.Button("Export to CSV", variant="secondary")
                        azure_download_file = gr.File(label="Download CSV Results File", interactive=False)
            
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown("### 📝 Edit & Validate Text")
                    azure_confirmed_textbox = gr.Textbox(
                        label="Edit Confirmed Page Transcription", 
                        placeholder="Modify the transcription here...",
                        lines=8
                    )
                    azure_save_correction_btn = gr.Button("Save Page Correction", variant="primary")
                    
            with gr.Row(variant="panel"):
                with gr.Column(scale=8):
                    gr.Markdown("### 📋 Clinical Discharge Card Generator")
                    generate_card_btn = gr.Button("Generate Discharge Card via GPT-5", variant="primary")
                with gr.Column(scale=12):
                    gr.Markdown("### 📄 Discharge Card Output")
                    discharge_card_output = gr.Markdown(value="*No discharge card generated yet.*")

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
    
    # Event Listeners for Segmented OCR
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

    # Event Listeners for Full-Page Mistral OCR
    run_btn.click(
        fn=analyze_full_page_mistral,
        inputs=[input_files],
        outputs=[full_pages_store, full_image_display, mistral_summary_table, selected_page_dropdown, mistral_status]
    )
    
    selected_page_dropdown.change(
        fn=load_full_page_details,
        inputs=[selected_page_dropdown, full_pages_store],
        outputs=[full_image_display, mistral_confirmed_textbox]
    )
    
    mistral_save_correction_btn.click(
        fn=update_full_page_transcription,
        inputs=[selected_page_dropdown, mistral_confirmed_textbox, full_pages_store],
        outputs=[mistral_summary_table, full_pages_store]
    )
    
    mistral_export_btn.click(
        fn=export_full_pages_to_csv,
        inputs=[full_pages_store],
        outputs=[mistral_download_file]
    )

    # Event Listeners for Azure CV Full-Page OCR
    azure_run_btn.click(
        fn=analyze_full_page_azure,
        inputs=[azure_input_files],
        outputs=[azure_pages_store, azure_image_display, azure_summary_table, azure_selected_dropdown, azure_status]
    )
    
    azure_selected_dropdown.change(
        fn=load_azure_page_details,
        inputs=[azure_selected_dropdown, azure_pages_store],
        outputs=[azure_image_display, azure_confirmed_textbox]
    )
    
    azure_save_correction_btn.click(
        fn=update_azure_page_transcription,
        inputs=[azure_selected_dropdown, azure_confirmed_textbox, azure_pages_store],
        outputs=[azure_summary_table, azure_pages_store]
    )
    
    azure_export_btn.click(
        fn=export_azure_pages_to_csv,
        inputs=[azure_pages_store],
        outputs=[azure_download_file]
    )
    
    generate_card_btn.click(
        fn=generate_discharge_card_with_gpt5,
        inputs=[azure_pages_store],
        outputs=[discharge_card_output]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme, css=custom_css)
