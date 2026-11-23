"""
Fusion Module — Action Map
============================
Suggested action keyed on dominant vision class x risk band
(docs/MDRIS_CONTEXT.md section 8.5). Every action traces to a model output.
"""

ACTION_MAP = {
    "person_in_water": {
        "Critical": "Dispatch rescue boat and NDRF swift-water team immediately",
        "High":     "Dispatch rescue boat; alert nearest NDRF unit",
        "Moderate": "Send field officer to verify; boat on standby",
        "Low":      "Log and monitor",
    },
    "person_in_vehicle": {
        "Critical": "Dispatch rescue team for vehicle extraction; close the road",
        "High":     "Dispatch rescue team; issue road closure",
        "Moderate": "Traffic police to divert; verify occupants",
        "Low":      "Log; issue traffic advisory",
    },
    "person": {
        "Critical": "Evacuate to nearest relief camp; send boat if access is cut",
        "High":     "Organise evacuation; send relief supplies",
        "Moderate": "Field officer check; alert relief camp",
        "Low":      "Monitor",
    },
    "vehicle": {
        "Critical": "Close road, divert traffic, check for occupants",
        "High":     "Close road and divert traffic",
        "Moderate": "Issue traffic advisory",
        "Low":      "Monitor",
    },
    "text_only": {
        "Critical": "Call back reporter; dispatch verification team now",
        "High":     "Call back reporter; send field officer",
        "Moderate": "Verify through field officer",
        "Low":      "Log",
    },
}

CONFLICT_SUFFIX = "Signals disagree: verify on ground before committing resources."


def suggest_action(dominant_class: str, risk_band: str, conflict: bool) -> str:
    """Look up the action for (dominant_class, risk_band); append the conflict note if needed."""
    action = ACTION_MAP[dominant_class][risk_band]
    if conflict:
        action = f"{action}. {CONFLICT_SUFFIX}"
    return action
