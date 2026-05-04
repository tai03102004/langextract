from PIL import Image
from paddleocr import PaddleOCR
import torchvision.transforms.functional as TF
from dataclasses import dataclass

@dataclass
class Cell:
    row_idx: int
    col_idx: int
    x: int; y: int; w: int; h: int
    is_span: bool = False
    text: str = ""

ocr = PaddleOCR(
    use_textline_orientation=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    lang="en"
)

def prepare_image_tensor(image_path, img_size=384):
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    scale = img_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    draft = orig_img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    padded.paste(draft, (0, 0))
    img_t = TF.to_tensor(padded).unsqueeze(0)
    return img_t, orig_w, orig_h, new_w, new_h