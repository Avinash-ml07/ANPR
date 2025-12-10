from fastapi import FastAPI, UploadFile
import cv2, numpy as np
from core import extract_plate
from vehicle_db import check_status

app = FastAPI()

@app.post("/recognise")
async def recognise(file: UploadFile):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    plates = extract_plate(img)

    for p in plates:
        p["status"] = check_status(p["text"])

    return {"plates": plates}
