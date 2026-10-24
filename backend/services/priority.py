def calculate_priority(urgency: int, severity: int) -> str:
    """
    ITIL Priority Matrix
    Urgency(1-3) + Severity(1-3) - 1
    3=High, 2=Medium, 1=Low
    Formula yields 1-5, mapped to P5-P1
    Value 5 -> P1
    Value 4 -> P2
    Value 3 -> P3
    Value 2 -> P4
    Value 1 -> P5
    """
    # Clip values to 1-3 range
    u = max(1, min(3, urgency))
    s = max(1, min(3, severity))
    
    score = u + s - 1
    # 5->P1, 4->P2, 3->P3, 2->P4, 1->P5
    # Priority = 6 - score
    p_level = 6 - score
    return f"P{p_level}"

def boost_priority(current_priority: str, child_count: int) -> str:
    """Every 5 children -> priority escalates by 1 level"""
    if not current_priority.startswith("P"):
        return current_priority
    
    level = int(current_priority[1:])
    boosts = child_count // 5
    
    new_level = max(1, level - boosts)
    return f"P{new_level}"
