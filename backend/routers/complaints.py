import os
import json
import uuid
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.complaint import Complaint
from backend.routers.auth import get_current_user
from backend.models.user import User
from backend.services.complaint_service import process_new_complaint, verify_live_detection

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_complaint(
    background_tasks: BackgroundTasks,
    description: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    location_source: str = Form(...),
    images: List[UploadFile] = File(...),
    captured_at: Optional[str] = Form(None),
    live_detection: bool = Form(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if len(images) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images allowed")

    saved_images = []
    first_image_bytes = None
    for image in images:
        ext = image.filename.split(".")[-1] if "." in image.filename else "jpg"
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            content = await image.read()
            f.write(content)
        if first_image_bytes is None:
            first_image_bytes = content
        saved_images.append(f"/uploads/{filename}")

    complaint = process_new_complaint(
        db=db,
        user_id=current_user.id,
        description=description,
        images=saved_images,
        lat=latitude,
        lon=longitude,
        location_source=location_source,
        captured_at=captured_at
    )

    # Re-verify the on-device detection server-side rather than trusting the
    # client. Runs after the response: inference is slow on small CPU hosts.
    if live_detection and first_image_bytes:
        background_tasks.add_task(verify_live_detection, complaint.id, first_image_bytes)

    return {
        "ticket_id": complaint.ticket_id,
        "category": complaint.category,
        "department": complaint.department,
        "priority": complaint.priority,
        "status": complaint.status,
        "master_ticket_id": complaint.master_ticket_id,
        "message": "Complaint registered successfully"
    }

@router.get("/track/{ticket_id}")
def track_complaint(ticket_id: str, db: Session = Depends(get_db)):
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
        
    return {
        "ticket_id": complaint.ticket_id,
        "description": complaint.description,
        "category": complaint.category,
        "department": complaint.department,
        "priority": complaint.priority,
        "status": complaint.status,
        "created_at": complaint.created_at.isoformat(),
        "updated_at": complaint.updated_at.isoformat(),
        "images": json.loads(complaint.images),
        "location": {"lat": complaint.lat, "lon": complaint.lon},
        "master_ticket_id": complaint.master_ticket_id
    }

@router.get("/my")
def get_my_complaints(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    complaints = db.query(Complaint).filter(Complaint.user_id == current_user.id).all()
    return [
        {
            "ticket_id": c.ticket_id,
            "description": c.description,
            "status": c.status,
            "priority": c.priority,
            "category": c.category,
            "created_at": c.created_at.isoformat()
        } for c in complaints
    ]
