import onnxruntime as ort
import numpy as np
import cv2
from typing import Tuple

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
):
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for m in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**m))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (neww, newh)

def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...], target_length=1024) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_length
    )
    coords = coords.copy().astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

class MobileSAM_pooldetection:
    def __init__(self) -> None:
        model_type = "vit_t"
        sam_checkpoint = "./data/mobile_sam.pt"

        self.ort_sess_img_encoder = None #ort.InferenceSession("./data/mobile_sam_Oct23_gelu_approx.onnx")
        self.ort_sess_mask_decoder =None #ort.InferenceSession("./data/sam_mask_decoder_single.onnx")
        self.target_length=1024
    
    def segment(self, IMAGE_PATH):
        image_org=cv2.imread(IMAGE_PATH)
        org_size=image_org.shape[:2]

        print("read image")
        
        target_size = get_preprocess_shape(image_org.shape[0], image_org.shape[1], self.target_length)

        image=cv2.resize(image_org, target_size)

        im_size=image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("defining image encoder")
        ort_sess_img_encoder=ort.InferenceSession("./data/mobile_sam_Oct23_gelu_approx_no_cf.onnx")

        outputs = ort_sess_img_encoder.run(None, {"input_image": image.astype(np.float32)})

        del ort_sess_img_encoder

        print("extarcted embedding")


        image_embedding=outputs[0]
        
        point_grids=build_all_layer_point_grids(8, 0, 1)

        print("built point grid")

        points_scale = np.array(org_size)[None, ::-1]
        points_for_image = point_grids[0] * points_scale

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        cnt=0
        masks=[]
        print("defining mask decoder")

        ort_sess_mask_decoder=ort.InferenceSession("./data/mobile_sam_opset11.onnx")
        for pt in points_for_image:
            # print(pt)
            onnx_coord=apply_coords(np.array(pt), org_size)
            onnx_label = np.array([1]*len(onnx_coord), dtype=np.float32)
            ort_inputs = {
                    "image_embeddings": image_embedding,
                    "point_coords": onnx_coord[None, None, :].astype(np.float32),
                    "point_labels": onnx_label[None,:],
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(org_size, dtype=np.float32)
                }
            msk, _, _ = ort_sess_mask_decoder.run(None, ort_inputs)
            masks.append(msk[0,0,:,:]>0.0)

        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2HSV)
        mask=self.select_mask(masks, image)
        rle=mask2rle(mask)
        return mask, rle, mask.shape
    
    def select_mask(self, masks, image):
        lower_range= np.array([180//2,50,50])
        upper_range= np.array([210//2,255,255])
        mask_in =cv2.inRange(image, lower_range, upper_range)
        mask=[]
        overlap=0.0
        cnt=1
        
        for i in range(len(masks)):
            mask = masks[i]
            olp= np.sum(mask_in*mask)

            if olp>overlap:
                final_mask=mask.copy()
                
                print("detected one of the mask")
                overlap=olp
                # np.save(f"./data/mask_{cnt}",final_mask)
                cnt+=1
                #break
        return final_mask


import sys
if __name__ == "__main__": 
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    print(image_path, mask_path)
    model=MobileSAM_pooldetection()
    print("Model defined")
    mask,_,_=model.segment(image_path)
    mask=np.uint8(mask*255)
    cv2.imwrite(mask_path, mask)

