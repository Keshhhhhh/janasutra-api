import json
import uuid
from backend.models.complaint import Complaint
from backend.models.master_ticket import MasterTicket
from backend.services.ai_service import analyze_complaint
from backend.services.routing import get_department
from backend.services.priority import calculate_priority, boost_priority
from backend.services.deduplication import find_master_ticket
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.services.pothole_detector import detect_potholes

def process_new_complaint(db: Session, user_id: str, description: str, images: list[str], lat: float, lon: float, location_source: str, captured_at: str | None = None) -> Complaint:
    # 1. AI Analysis — classify from user description (+ image path for future use)
    img_path = images[0] if images else None
    ai_result = analyze_complaint(description=description, image_path=img_path)

    if captured_at:
        ai_result["captured_at"] = captured_at

    category = ai_result.get("category", "other")
    severity = ai_result.get("severity", 2)
    is_auth = ai_result.get("is_authentic", True)
    
    # 2. Routing
    department = get_department(category)
    
    # 3. Priority Calculation
    # Assuming urgency is 2 by default, can be tweaked
    urgency = 2
    priority = calculate_priority(urgency, severity)
    
    # 4. Deduplication
    master_ticket = find_master_ticket(db, description, lat, lon, category)
    
    if master_ticket:
        # Attach as child
        master_ticket.child_count += 1
        
        # Recalculate centroid (simple average)
        n = master_ticket.child_count
        master_ticket.centroid_lat = ((master_ticket.centroid_lat * (n - 1)) + lat) / n
        master_ticket.centroid_lon = ((master_ticket.centroid_lon * (n - 1)) + lon) / n
        
        # Boost priority
        master_ticket.priority = boost_priority(master_ticket.priority, master_ticket.child_count)
        master_id = master_ticket.id
    else:
        # Create new master ticket
        new_master = MasterTicket(
            canonical_description=description,
            category=category,
            department=department,
            priority=priority,
            centroid_lat=lat,
            centroid_lon=lon
        )
        db.add(new_master)
        db.flush() # get ID
        master_id = new_master.id

    # Generate ticket ID JS-{YEAR}-{SEQUENTIAL_5_DIGIT} (safe against deleted tickets)
    year = 2026
    all_tickets = db.query(Complaint.ticket_id).filter(Complaint.ticket_id.like(f"JS-{year}-%")).all()
    max_num = 0
    for (t_id,) in all_tickets:
        try:
            num = int(t_id.split("-")[-1])
            if num > max_num:
                max_num = num
        except (ValueError, IndexError):
            pass
    ticket_id = f"JS-{year}-{str(max_num + 1).zfill(5)}"
    
    complaint = Complaint(
        ticket_id=ticket_id,
        user_id=user_id,
        description=description,
        images=json.dumps(images),
        lat=lat,
        lon=lon,
        location_source=location_source,
        category=category,
        department=department,
        priority=priority,
        urgency=urgency,
        severity=severity,
        ai_analysis=json.dumps(ai_result),
        is_authentic=is_auth,
        master_ticket_id=master_id
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    return complaint


def verify_live_detection(complaint_id: str, image_bytes: bytes) -> None:
    """Background task: confirm a live-camera pothole report with the server model.

    A confirmed detection overrides the text classifier (pothole, severity 3,
    BBMP Roads); the result is stored in ai_analysis either way so admins can
    see whether the on-device detection held up.
    """
    try:
        detection = detect_potholes(image_bytes)
    except Exception as exc:
        print(f"[verify_live_detection] {complaint_id}: detection failed: {exc}")
        return

    db = SessionLocal()
    try:
        complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
        if not complaint:
            return
        ai_result = json.loads(complaint.ai_analysis)
        ai_result["camera_detection"] = detection
        ai_result["camera_detection_verified"] = detection["detected"]
        if detection["detected"]:
            ai_result["category"] = "pothole"
            ai_result["severity"] = 3
            ai_result["description"] = "Detected: a road surface pothole (live camera)."
            complaint.category = "pothole"
            complaint.severity = 3
            complaint.department = get_department("pothole")
        complaint.ai_analysis = json.dumps(ai_result)
        db.commit()
    finally:
        db.close()
