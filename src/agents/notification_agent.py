"""
AGENT 5 of 6 -- NotificationAgent

ROLE:   Tell the outside world when a part FAILS, so a human or a machine can act:
        light a tower lamp, push the part off the line, or alert an operator. This
        is the "real-time feedback to operators and manufacturing systems" pillar.

WHY A SEPARATE AGENT: who/what to notify changes per site -- a console log in
development, an MQTT topic to a PLC in production, an email to a supervisor. By
isolating it, swapping the channel never touches the inspection logic.

PRODUCTION NOTE: the MQTT method below is the bridge to the factory floor
(PLC/MES/SCADA). In the demo it logs to the console so it runs with no broker.
See docs/08_manufacturing_integration.md for the full OPC-UA / MQTT design.

INTERACTS WITH: receives the final decision from the Orchestrator after storage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src import config


class NotificationAgent:
    name = "notification"

    def __init__(
        self,
        channel: str = "console",
        mqtt_host: str | None = None,
        mqtt_topic: str = "factory/quality/alerts",
    ):
        # channel: "console" (demo) or "mqtt" (production bridge to PLC/MES).
        self.channel = channel
        self.mqtt_host = mqtt_host
        self.mqtt_topic = mqtt_topic
        self._mqtt_client = None

    def notify(
        self, *, inspection_id: int, part_id: str, decision: dict[str, Any]
    ) -> None:
        """Send an alert ONLY for failures; passes stay silent to avoid noise."""
        if decision["verdict"] != "FAIL":
            return

        payload = {
            "event": "DEFECT_DETECTED",
            "inspection_id": inspection_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "line_id": config.LINE_ID,
            "station_id": config.STATION_ID,
            "part_id": part_id,
            "defect_type": decision.get("defect_type"),
            "severity": decision.get("severity"),
            "score": round(decision["score"], 4),
        }

        if self.channel == "mqtt":
            self._send_mqtt(payload)
        else:
            self._send_console(payload)

    # --- channels ------------------------------------------------------------
    def _send_console(self, payload: dict[str, Any]) -> None:
        print(f"[ALERT] {json.dumps(payload)}")

    def _send_mqtt(self, payload: dict[str, Any]) -> None:
        """
        Publish the alert to an MQTT topic that a PLC/MES subscribes to.
        Needs paho-mqtt (`pip install paho-mqtt`) and a running broker.
        """
        if self._mqtt_client is None:
            import paho.mqtt.client as mqtt  # imported lazily

            self._mqtt_client = mqtt.Client()
            self._mqtt_client.connect(self.mqtt_host or "localhost", 1883, 60)
            self._mqtt_client.loop_start()
        self._mqtt_client.publish(self.mqtt_topic, json.dumps(payload), qos=1)
