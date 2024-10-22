from PIL import Image
import numpy as np
import torch 
import torchvision.transforms as T
from PIL import ImageEnhance

def preprocess_image(img):
    image = Image.fromarray(img)
    # Resize the image
    image_resized = image.resize((84, 110))
    # Convert to grayscale
    image_gray = image_resized.convert('L')
    enhancer = ImageEnhance.Contrast(image_gray)
    image_contrast = enhancer.enhance(4)
    # Calculate the coordinates for cropping
    left = 0
    top = 16
    right = 84
    bottom = 16+84

    cropped_image = image_contrast.crop((left, top, right, bottom))

    # Convert PIL Image back to NumPy array (if needed)
    image_array = np.array(cropped_image, dtype=np.float32) / 255
    return image_array


def crop_screen(screen):
    bbox = [34,0,160,160] #(x,y,delta_x,delta_y)
    screen = screen[:, bbox[0]:bbox[2]+bbox[0], bbox[1]:bbox[3]+bbox[1]] #BZX:(CHW)
    return screen

def transform_screen_data(screen):
    # Convert to float, rescale, convert to tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = crop_screen(screen)
    # Use torchvision package to compose image transforms
    resize = T.Compose([
            T.ToPILImage()
            , T.Grayscale()
            , T.Resize((84, 84)) #BZX: the original paper's settings:(110,84), however, for simplicty we...
            , T.ToTensor()
    ])
    # add a batch dimension (BCHW)
    screen = resize(screen)

    return screen.unsqueeze(0)   # BZX: Pay attention to the shape here. should be [1,1,84,84]

def get_processed_screen(screen_state):
    screen = screen_state.transpose((2, 0, 1))  # PyTorch expects CHW
    screen = crop_screen(screen)
    return transform_screen_data(screen)    
