from PIL import Image

for i in range(1, 5):
    path = f"32map_{i}.png"
    img = Image.open(path)
    # The original was 128x128 because I already scaled by 4. So I will scale by another 6 (to get ~768)
    new_img = img.resize((img.width * 6, img.height * 6), Image.NEAREST)
    new_img.save(path)
    print(f"Resized {path} to {new_img.width}x{new_img.height}")
