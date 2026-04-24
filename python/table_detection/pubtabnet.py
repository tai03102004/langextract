import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


class PubTabNetDownloader:
    def __init__(self, data_dir: str = "../../data/PaddleOCR"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def download_random(self, num_samples: int = 20000, split: str = "train"):
        """
        Random chuẩn bằng Reservoir Sampling (uniform over stream)
        """
        print(f"📥 Tải RANDOM CHUẨN {num_samples} ảnh từ PubTabNet ({split})...")

        dataset = load_dataset(
            "apoidea/pubtabnet-html",
            split=split,
            streaming=True,
            trust_remote_code=True
        )

        reservoir = []
        iterator = iter(dataset)

        print("🔄 Đang sampling (có thể hơi lâu vì phải duyệt stream)...")

        for i, sample in enumerate(tqdm(iterator, desc="Sampling")):
            if i < num_samples:
                reservoir.append(sample)
            else:
                j = random.randint(0, i)
                if j < num_samples:
                    reservoir[j] = sample

        print(f"✅ Sampling xong {len(reservoir)} samples")

        # Lưu dữ liệu
        existing = len(list(self.images_dir.glob("*.png")))

        for idx, sample in enumerate(tqdm(reservoir, desc="💾 Đang lưu")):
            file_id = existing + idx + 1
            filename = f"sample_{file_id:06d}"

            # Save image
            image = sample.get("image")
            if image:
                image.save(self.images_dir / f"{filename}.png")

            # Save annotation
            annotation = {
                "filename": f"{filename}.png",
                "html": sample.get("html", ""),
                "source": f"pubtabnet_{split}_random"
            }

            with open(self.annotations_dir / f"{filename}.json", "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)

        print(f"🎉 Done! Đã lưu {len(reservoir)} ảnh vào {self.data_dir}")
        return len(reservoir)

    def get_stats(self):
        return {
            "images": len(list(self.images_dir.glob("*.png"))),
            "annotations": len(list(self.annotations_dir.glob("*.json"))),
        }


if __name__ == "__main__":
    downloader = PubTabNetDownloader()

    downloader.download_random(num_samples=20000, split="train")

    stats = downloader.get_stats()
    print(f"\n📊 Stats: {stats}")