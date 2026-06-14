import time
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
    print(f"[*] Running detection-only OCR on {img_path}...", flush=True)
    start = time.time()
    res = ocr.ocr(img_path, rec=False)
    print(f"[*] Completed detection-only OCR in {time.time() - start:.2f} seconds.", flush=True)
    print("RESULT:", res)

if __name__ == "__main__":
    main()
