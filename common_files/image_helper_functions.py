from PIL import Image
import numpy as np

def preprocess_image(img):
    image = Image.fromarray(img)
    # Resize the image
    image_resized = image.resize((84, 110))
    # Convert to grayscale
    image_gray = image_resized.convert('L')

    # Calculate the coordinates for cropping
    left = 0
    top = 110 - 84
    right = 84
    bottom = 110

    cropped_image = image_gray.crop((left, top, right, bottom))

    # Convert PIL Image back to NumPy array (if needed)
    image_array = np.array(cropped_image, dtype=np.float32)
    return image_array