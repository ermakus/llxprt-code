#!/usr/bin/env python3
"""
Simple Python client for communicating with the LLxprt A2A Server.

This client demonstrates how to:
1. Fetch the agent card (capabilities)
2. Create a new task
3. Send messages and receive streaming responses

Usage:
    python a2a_client.py [--host HOST] [--port PORT] [--message MESSAGE]

Example:
    python a2a_client.py --message "Write a hello world function in Python"
"""

import argparse
import json
import sys
import uuid
from typing import Generator

import requests


class A2AClient:
    """Client for communicating with the A2A server."""

    def __init__(self, host: str = "localhost", port: int = 41242):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()

    def get_agent_card(self) -> dict:
        """Fetch the agent card with capabilities."""
        response = self.session.get(
            f"{self.base_url}/.well-known/agent-card.json"
        )
        response.raise_for_status()
        return response.json()

    def create_task(
        self,
        context_id: str | None = None,
        agent_settings: dict | None = None,
    ) -> str:
        """Create a new task and return the task ID."""
        payload = {}
        if context_id:
            payload["contextId"] = context_id
        if agent_settings:
            payload["agentSettings"] = agent_settings

        response = self.session.post(
            f"{self.base_url}/tasks",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def get_task_metadata(self, task_id: str) -> dict:
        """Get metadata for a specific task."""
        response = self.session.get(
            f"{self.base_url}/tasks/{task_id}/metadata"
        )
        response.raise_for_status()
        return response.json()

    def send_message(
        self, task_id: str, message: str
    ) -> Generator[dict, None, None]:
        """
        Send a message to the agent and stream the response.

        Uses JSON-RPC over HTTP with SSE streaming.
        Yields parsed SSE events as dictionaries.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/stream",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "messageId": str(uuid.uuid4()),
                },
            },
        }

        response = self.session.post(
            self.base_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            stream=True,
        )
        response.raise_for_status()

        # Parse SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        print(f"[Warning] Could not parse: {data}", file=sys.stderr)

    def send_message_sync(
        self, task_id: str, message: str
    ) -> dict:
        """
        Send a message to the agent and get a non-streaming response.

        Uses JSON-RPC over HTTP.
        Returns the response as a dictionary.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "messageId": str(uuid.uuid4()),
                },
            },
        }

        response = self.session.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def get_task(self, task_id: str, history_length: int | None = None) -> dict:
        """
        Get task details by ID.

        Uses JSON-RPC over HTTP.
        """
        params = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = history_length

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/get",
            "params": params,
        }

        response = self.session.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def resubscribe_task(self, task_id: str) -> Generator[dict, None, None]:
        """
        Resubscribe to a task's event stream.

        Uses JSON-RPC over HTTP with SSE streaming.
        Yields parsed SSE events as dictionaries.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/resubscribe",
            "params": {"id": task_id},
        }

        response = self.session.post(
            self.base_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        print(f"[Warning] Could not parse: {data}", file=sys.stderr)


def print_event(event: dict, verbose: bool = False) -> None:
    """Print an SSE event in a readable format."""
    if verbose:
        print(json.dumps(event, indent=2))
        return

    # Extract useful information from the event
    if "result" in event:
        result = event["result"]
        if "status" in result:
            status = result["status"]
            state = status.get("state", "unknown")
            print(f"[State: {state}]")

            # Print message content if available
            message = status.get("message", {})
            parts = message.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    text = part.get("text", "")
                    if text:
                        print(text)

        # Handle artifacts (generated files, etc.)
        if "artifact" in result:
            artifact = result["artifact"]
            print(f"\n[Artifact: {artifact.get('name', 'unnamed')}]")
            parts = artifact.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    print(part.get("text", ""))

    elif "error" in event:
        error = event["error"]
        print(f"[Error] {error.get('message', 'Unknown error')}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="A2A Client for LLxprt Code Server"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="A2A server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=41242,
        help="A2A server port (default: 41242)",
    )
    parser.add_argument(
        "--message",
        "-m",
        default="Write a simple hello world function in Python",
        help="Message to send to the agent",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full JSON responses",
    )
    parser.add_argument(
        "--agent-card",
        action="store_true",
        help="Only fetch and display the agent card",
    )

    args = parser.parse_args()

    client = A2AClient(host=args.host, port=args.port)

    try:
        # Fetch and display agent card
        print("=" * 60)
        print("Fetching Agent Card...")
        print("=" * 60)
        agent_card = client.get_agent_card()
        print(f"Agent: {agent_card.get('name', 'Unknown')}")
        print(f"Description: {agent_card.get('description', 'N/A')}")
        print(f"Version: {agent_card.get('version', 'N/A')}")
        print(f"Protocol Version: {agent_card.get('protocolVersion', 'N/A')}")

        capabilities = agent_card.get("capabilities", {})
        print(f"Streaming: {capabilities.get('streaming', False)}")

        skills = agent_card.get("skills", [])
        if skills:
            print("\nSkills:")
            for skill in skills:
                print(f"  - {skill.get('name', 'Unknown')}: {skill.get('description', 'N/A')}")

        if args.agent_card:
            return

        # Create a new task
        print("\n" + "=" * 60)
        print("Creating Task...")
        print("=" * 60)
        task_id = client.create_task()
        print(f"Task ID: {task_id}")

        # Get task metadata
        metadata = client.get_task_metadata(task_id)
        print(f"Task State: {metadata.get('metadata', {}).get('taskState', 'unknown')}")

        # Send message and stream response
        print("\n" + "=" * 60)
        print(f"Sending Message: {args.message}")
        print("=" * 60 + "\n")

        for event in client.send_message(task_id, args.message):
            print_event(event, verbose=args.verbose)

        print("\n" + "=" * 60)
        print("Task Complete")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print(
            f"Error: Could not connect to A2A server at {args.host}:{args.port}",
            file=sys.stderr,
        )
        print("Make sure the server is running.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
