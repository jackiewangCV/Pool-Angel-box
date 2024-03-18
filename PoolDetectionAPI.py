# from segmentation.SAM import SAM_pooldetection
from segmentation.fast_SAM import fastSAM_pooldetection
from fastapi import FastAPI
import os
from fastapi import File, UploadFile


app = FastAPI()
# model=SAM_pooldetection('vit_h', 'data/models/sam_vit_h_4b8939.pth')
model=fastSAM_pooldetection()

###################################

@app.get("/")
async def root():
    return {"message": "This is the pool detection API"}

@app.post('/detect_pool')
def detect_pool(file: UploadFile = File(...)):
        try:
                contents = file.file.read()
                with open(file.filename, 'wb') as f:
                        f.write(contents)
        except Exception:
                return {"message": "There was an error uploading the file"}
        finally:
                file.file.close()

        _,rle, im_size=model.segment(file.filename)
        os.remove(file.filename)
        return {"message": {"rle":rle, 'size': im_size}, "status": 200}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)