from segmentation.SAM import SAM_pooldetection
import cv2
import os



def write_masks_to_folder(masks,path) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))


model=SAM_pooldetection('vit_h', 'data/models/sam_vit_h_4b8939.pth')

name='vid2-005'
# name='img3_0002'
# name='vid2-006'
# image = cv2.imread(f'data/images/{name}.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask,_=model.segment(f'data/images/{name}.jpg')
print(mask)
cv2.imwrite(f'data/{name}.png', mask * 255)

#write_masks_to_folder(mask, f'data/masks_{name}.png')


