"""
Static Demo Resources
=======================
Fixed demo inventory for the Resources page. It is NOT connected to any
resource system; every response is labelled data_origin = "static_demo".
"""

NOTE = "Not connected to any resource system."

RESOURCES = [
    {"id": "RES-001", "type": "Rescue Boats", "total": 85, "available": 12, "assigned_area": "Chennai", "status": "Deploying"},
    {"id": "RES-002", "type": "Water Pumps", "total": 150, "available": 42, "assigned_area": "Mumbai", "status": "Active"},
    {"id": "RES-003", "type": "Helicopters", "total": 12, "available": 3, "assigned_area": "Assam", "status": "Standby"},
    {"id": "RES-004", "type": "Food Packets", "total": 50000, "available": 12000, "assigned_area": "Multiple", "status": "Distributing"},
    {"id": "RES-005", "type": "Medical Kits", "total": 20000, "available": 5400, "assigned_area": "Multiple", "status": "Available"},
    {"id": "RES-006", "type": "Rescue Personnel", "total": 1200, "available": 150, "assigned_area": "Kerala", "status": "Active"},
]
