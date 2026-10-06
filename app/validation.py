from fastapi import HTTPException, UploadFile

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}


async def validate_image_upload(file: UploadFile, max_bytes: int) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds max size of {max_bytes} bytes.")
    return data
