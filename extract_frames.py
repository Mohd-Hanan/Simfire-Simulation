import os
from PIL import Image

def extract(gif_path, prefix, num_frames=4):
    if not os.path.exists(gif_path):
        print(f"File not found: {gif_path}")
        return
    img = Image.open(gif_path)
    total_frames = img.n_frames
    step = max(1, total_frames // num_frames)
    for i in range(num_frames):
        frame_idx = min(i * step, total_frames - 1)
        img.seek(frame_idx)
        out_path = f"{prefix}_{i+1}.png"
        img.convert("RGB").save(out_path)
        print(f"Saved {out_path}")

os.makedirs("pdf_assets", exist_ok=True)
extract("/home/hari/.gemini/antigravity/brain/03666627-d24e-4417-8dc9-325260f3fbc9/550_iter_watch.gif", "pdf_assets/128map", 4)
extract("/home/hari/.gemini/antigravity/brain/03666627-d24e-4417-8dc9-325260f3fbc9/1091_iter_watch.gif", "pdf_assets/32map", 4)
