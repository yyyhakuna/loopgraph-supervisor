from __future__ import annotations

import json
import sys

context = json.load(sys.stdin)
completed = context.get("output", {}).get("completed") is True
json.dump(
    {
        "score": 1.0 if completed else 0.0,
        "passed": completed,
        "dimensions": {"business_completion": 1.0 if completed else 0.0},
        "hard_constraints": {"completed": completed},
        "confidence": 1.0,
        "evidence": context.get("output", {}).get("evidence", []),
        "feedback": [] if completed else ["The business task is not complete."],
        "retryable": not completed,
    },
    sys.stdout,
)
