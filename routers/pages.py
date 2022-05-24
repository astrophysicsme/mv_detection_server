from fastapi import APIRouter

from controllers import inspections

router = APIRouter()


@router.get("/")
async def root():
    return {"root route"}


@router.post("/inspect")
async def inspect_images():
    return inspections.inspect_single_image()
