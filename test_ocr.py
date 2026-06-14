import sys
import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running test on device: {device}", flush=True)
    
    # Path to test image
    image_path = "samples/printed_sample.png"
    if not os.path.exists(image_path):
        print(f"[-] Image {image_path} does not exist. Please run app.py first to generate it.", flush=True)
        sys.exit(1)
        
    image = Image.open(image_path)
    
    # Test printed model
    model_name = "microsoft/trocr-base-printed"
    print(f"[*] Loading processor for {model_name}...", flush=True)
    processor = TrOCRProcessor.from_pretrained(model_name)
    print(f"[*] Loading model for {model_name}...", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    
    # Run prediction
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    print(f"[*] Generating transcription...", flush=True)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"[+] Transcription result: '{transcription}'", flush=True)
    
    if "printed text" in transcription.lower():
        print("[+] Test PASSED!", flush=True)
    else:
        print("[-] Test FAILED: Transcription did not match expected output.", flush=True)

if __name__ == "__main__":
    test()
