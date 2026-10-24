import math
from backend.models.master_ticket import MasterTicket
from backend.services.ai_service import get_text_embedding, compute_similarity

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine formula to calculate distance in meters between two points."""
    R = 6371e3 # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def find_master_ticket(db, description: str, lat: float, lon: float, category: str):
    """
    Find matching master ticket.
    Match if: text similarity > 0.85 AND geographic distance < 500 meters AND same category
    """
    open_masters = db.query(MasterTicket).filter(MasterTicket.status != "resolved", MasterTicket.category == category).all()
    
    new_embedding = get_text_embedding(description)
    
    best_match = None
    best_score = 0
    
    for master in open_masters:
        dist = calculate_distance(lat, lon, master.centroid_lat, master.centroid_lon)
        if dist < 500:
            master_embedding = get_text_embedding(master.canonical_description)
            sim = compute_similarity(new_embedding, master_embedding)
            if sim > 0.85 and sim > best_score:
                best_score = sim
                best_match = master
                
    return best_match
