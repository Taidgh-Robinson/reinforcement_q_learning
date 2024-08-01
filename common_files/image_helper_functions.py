from PIL import Image
import numpy as np

def preprocess_image(img):
    image = Image.fromarray(img)
    # Resize the image
    image_resized = image.resize((84, 110))
    # Convert to grayscale
    image_gray = image_resized.convert('L')
    image_gray.show()
    # Convert PIL Image back to NumPy array (if needed)
    gray_array = np.array(image_gray)
