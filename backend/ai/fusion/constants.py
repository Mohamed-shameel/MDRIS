"""
Fusion Module — Constants (single source of truth)
====================================================
Every value here is a documented design choice, not a calibrated parameter
(docs/MDRIS_CONTEXT.md sections 5.1.1, 5.2.3 and 8.1).

training/common.py imports the label definitions from this file, so the
training scripts and the backend can never disagree.
"""

# ── Label definitions (shared with training/common.py) ──────────────────────

# Canonical vision classes; list index == YOLO class id
CLASS_NAMES = ["person_in_water", "person_in_vehicle", "person", "vehicle"]

# COCO-pretrained yolov8n.pt class id -> canonical class (D-37). Other COCO classes are ignored.
# It can never output person_in_water or person_in_vehicle.
COCO_TO_CANONICAL = {0: "person", 2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle"}

# Urgency labels; list index == DistilBERT label id
URGENCY_LABELS = ["Low", "Medium", "High"]

# HumAID humanitarian category -> urgency (section 5.2.3)
HUMAID_TO_URGENCY = {
    "requests_or_urgent_needs":               "High",
    "injured_or_dead_people":                 "High",
    "missing_or_found_people":                "High",
    "displaced_people_and_evacuations":       "High",
    "infrastructure_and_utility_damage":      "Medium",
    "caution_and_advice":                     "Medium",
    "rescue_volunteering_or_donation_effort": "Medium",
    "sympathy_and_support":                   "Low",
    "other_relevant_information":             "Low",
    "not_humanitarian":                       "Low",
}

# ── Fusion constants (section 8.1) ──────────────────────────────────────────

VISION_WEIGHT = 0.6
NLP_WEIGHT    = 0.4

CLASS_WEIGHTS = {            # w_c: how much one confident detection of this class implies danger
    "person_in_water":   1.00,
    "person_in_vehicle": 0.85,
    "person":            0.45,
    "vehicle":           0.15,
}

URGENCY_VALUES = {"Low": 0.15, "Medium": 0.50, "High": 0.90}

VISION_SOURCE_RELIABILITY = {
    "officer_photo":  1.00,  # trained responder, known location
    "drone":          0.90,  # operator-controlled, may be oblique
    "cctv":           0.85,  # fixed camera, low resolution, night issues
    "citizen_upload": 0.60,  # unverified, could be old or from elsewhere
}
TEXT_SOURCE_RELIABILITY = {
    "officer_report":    1.00,
    "control_room_call": 0.80,  # transcribed, second-hand
    "citizen_sms":       0.60,
    "social_media":      0.50,  # unverified, may be rumour or a repost
}

NO_DETECTION_CONFIDENCE = 0.50   # confidence assigned to "image present, nothing detected"
DETECTION_CONF_THRESHOLD = 0.25  # YOLO inference threshold
CONFLICT_THRESHOLD = 0.50        # |S_v - S_n| above this -> conflict
RISK_BANDS = [(75, "Critical"), (55, "High"), (35, "Moderate"), (0, "Low")]
EVIDENCE_LEVELS = [(0.70, "Strong"), (0.40, "Fair"), (0.0, "Weak")]

# Used when a caller does not say where an input came from: treat it as unverified
DEFAULT_VISION_SOURCE = "citizen_upload"
DEFAULT_TEXT_SOURCE = "citizen_sms"
