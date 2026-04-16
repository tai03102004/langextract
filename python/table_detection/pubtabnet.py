import json
import random
from pathlib import Path
from tqdm import tqdm


class PubTabNetDownloader:
    """Download PubTabNet dataset từ Hugging Face"""
    
    def __init__(self, data_dir: str = "../../data/pubtabnet"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, num_samples: int = 100, split: str = "train", skip: int = 0):
        """Download samples từ Hugging Face"""
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ Cần cài đặt: pip install datasets")
            return False
        
        print(f"📥 Downloading PubTabNet...")
        print(f"   Split: {split}, Samples: {num_samples}")
        
        # ✅ Dùng streaming=True để không download toàn bộ 12GB
        dataset = load_dataset(
            "apoidea/pubtabnet-html", 
            split=split,
            streaming=True, 
            trust_remote_code=True
        )
        
        existing_files = len(list(self.images_dir.glob("*.png")))
        # Lấy num_samples đầu tiên
        count = 0
        skipped = 0

        for sample in tqdm(dataset, desc="Downloading", total=skip + num_samples):

            if skipped < skip:
                skipped += 1
                continue

            if count >= num_samples:
                break

            file_index = existing_files + count + 1
            filename = f"sample_{file_index:05d}"
                        
            # Lưu ảnh
            image = sample.get('image')
            if image:
                image.save(self.images_dir / f"{filename}.png")
            
            # Lưu annotation
            annotation = {
                'filename': f"{filename}.png",
                'html': sample.get('html', ''),
            }
            with open(self.annotations_dir / f"{filename}.json", 'w') as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)
            
            count += 1
        
        print(f"✅ Saved {count} samples to {self.data_dir}")
        return True
    
    def load_samples(self, max_samples: int = None):
        """Load samples đã download"""
        samples = []
        image_files = sorted(self.images_dir.glob("*.png"))
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        for img_path in image_files:
            anno_path = self.annotations_dir / f"{img_path.stem}.json"
            annotation = None
            
            if anno_path.exists():
                with open(anno_path) as f:
                    annotation = json.load(f)
            
            samples.append({
                'image_path': str(img_path),
                'annotation': annotation
            })
        
        return samples
    
    def get_stats(self):
        """Thống kê dataset"""
        images = len(list(self.images_dir.glob("*.png")))
        annotations = len(list(self.annotations_dir.glob("*.json")))
        
        return {
            'images': images,
            'annotations': annotations,
            'status': 'ready' if images > 0 else 'empty'
        }


if __name__ == "__main__":
    downloader = PubTabNetDownloader()
    
    # ⚠️ Chỉ download 100 samples để test trước
    downloader.download(num_samples=100, skip=500677)
    
    stats = downloader.get_stats()
    print(f"\n📊 Stats: {stats}")