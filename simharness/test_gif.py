import pygame
from PIL import Image

def test_memory():
    pygame.init()
    screen_size = (128, 128)
    screen = pygame.Surface(screen_size)
    
    frames = []
    
    for i in range(250):
        screen.fill((i % 255, 0, 0))
        screen_bytes = pygame.image.tostring(screen, "RGB")
        screen_im = Image.frombuffer(
            "RGB", screen_size, screen_bytes
        )
        frames.append(screen_im)
        
    print("Frames collected. Now saving GIF...")
    frames[0].save(
        "test.gif",
        save_all=True,
        duration=100,
        loop=0,
        append_images=frames[1:],
    )
    print("Done saving!")

if __name__ == "__main__":
    test_memory()
