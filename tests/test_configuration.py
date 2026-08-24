from __future__ import annotations

import sys

from loopgraph_supervisor.config import Settings, build_runtime


def test_settings_build_harness_and_grader_registries_without_framework_coupling(tmp_path) -> None:
    settings = Settings.model_validate(
        {
            "database_url": f"sqlite+aiosqlite:///{tmp_path / 'configured.db'}",
            "max_concurrency": 3,
            "harnesses": [
                {
                    "type": "command",
                    "name": "custom-agent",
                    "argv": [sys.executable, "-c", "print()"],
                }
            ],
            "graders": [
                {
                    "type": "rules",
                    "id": "business-goal",
                    "threshold": 1,
                    "rules": [
                        {
                            "id": "done",
                            "path": "done",
                            "operator": "equals",
                            "expected": True,
                        }
                    ],
                }
            ],
            "supervisor_agent": {
                "harness_id": "custom-agent",
                "bundle": {
                    "name": "supervisor",
                    "system_prompt": "Return structured directives.",
                },
                "observation_policy": {"after_tool_errors": 3},
            },
        }
    )

    runtime = build_runtime(settings)

    assert runtime.harnesses.names() == ("custom-agent",)
    assert sorted(runtime.graders) == ["business-goal"]
    assert runtime.tasks.max_concurrency == 3
    assert runtime.engine.advisor is not None
    assert runtime.engine.observation_policy.after_tool_errors == 3
