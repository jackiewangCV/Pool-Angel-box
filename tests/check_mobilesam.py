

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from PIL import Image
import numpy as np

# torch.set_num_threads(1)

IMAGE_PATH = '/data2/Freelancer/PoolAngel/Data/PoolAngel2/vid2-001.jpg'
image = Image.open(IMAGE_PATH)

model_type = "vit_t"
sam_checkpoint = "./data/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

from mobile_sam import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(np.array(image))

print(masks)
