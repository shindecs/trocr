import time
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

def main():
    print("[*] Initializing PaddleOCR...", flush=True)
    start = time.time()
    ocr = PaddleOCR(
        lang='en', 
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print(f"[*] Initialized in {time.time() - start:.2f} seconds.", flush=True)
    
    img_path = 'samples/mix.JPG'
    print(f"[*] Loading original image {img_path}...", flush=True)
    orig_img = Image.open(img_path)
    orig_w, orig_h = orig_img.size
    print(f"[*] Original size: {orig_w}x{orig_h}", flush=True)
    
    # Resize to a max dimension of 1000px
    max_dim = 1000
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized_img = orig_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        print(f"[*] Resized to: {new_w}x{new_h} (scale: {scale:.4f})", flush=True)
    else:
        resized_img = orig_img
        scale = 1.0
        
    img_np = np.array(resized_img)
    
    print(f"[*] Running PaddleOCR on resized image...", flush=True)
    start = time.time()
    res = ocr.ocr(img_np)
    print(f"[*] Completed OCR in {time.time() - start:.2f} seconds.", flush=True)
    
    if res and res[0]:
        data = res[0]
        polys = data.get('dt_polys', [])
        texts = data.get('rec_texts', [])
        print(f"[+] Found {len(polys)} text regions.", flush=True)
        if len(polys) > 0:
            # Show original vs scaled box coordinates for box 1
            poly = polys[0]
            orig_poly = poly / scale
            print(f"[+] Box 1 (Resized coords): {poly.tolist()}", flush=True)
            print(f"[+] Box 1 (Mapped to Original coords): {orig_poly.tolist()}", flush=True)
            print(f"[+] Box 1 Text: '{texts[0]}'", flush=True)
    else:
        print("[-] No text found.", flush=True)

if __name__ == "__main__":
    main()
