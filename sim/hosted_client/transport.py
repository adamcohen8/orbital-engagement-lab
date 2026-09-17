"""Black-box JSON transport used by the public Hosted OEL client."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Mapping


class HostedServiceError(RuntimeError):
    """A bounded service error returned by the configured hosted transport."""


class SubprocessJsonTransport:
    def __init__(self, profile: Mapping[str, Any]) -> None:
        transport = dict(profile["transport"])
        self.command = tuple(map(str, transport["command"]))
        self.timeout_s = float(transport["timeout_s"])

    def call(self, action: str, arguments: Mapping[str, Any]) -> Any:
        request = {
            "schema": "oel.hosted_control_request.v1",
            "action": str(action),
            "arguments": dict(arguments),
        }
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise HostedServiceError("Hosted OEL service request timed out.") from exc
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise HostedServiceError("Hosted OEL service returned a malformed response.") from exc
        if not isinstance(response, Mapping):
            raise HostedServiceError("Hosted OEL service returned a non-object response.")
        if completed.returncode != 0 or response.get("outcome") != "ok":
            error = response.get("error", {})
            error_type = str(error.get("type", "service_error")) if isinstance(error, Mapping) else "service_error"
            raise HostedServiceError(
                f"Hosted OEL service rejected action {action!r} ({error_type})."
            )
        return response.get("result")


__all__ = ["HostedServiceError", "SubprocessJsonTransport"]
