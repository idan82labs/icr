"""
Stdio transport implementation for IC-MCP.

This module provides the stdio transport for local Claude Code integration,
handling JSON-RPC messages over stdin/stdout.
"""

import asyncio
import json
import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ic_mcp.server import ICRMCPServer

logger = logging.getLogger(__name__)


class StdioTransport:
    """
    Stdio transport for MCP server.

    Handles reading JSON-RPC messages from stdin and writing responses to stdout.
    Follows the MCP protocol specification for message framing.
    """

    def __init__(self, server: "ICRMCPServer") -> None:
        """
        Initialize the stdio transport.

        Args:
            server: The MCP server instance to handle messages
        """
        self.server = server
        self._running = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        """Start the transport, reading from stdin and writing to stdout."""
        self._running = True

        # Set up async stdin/stdout
        loop = asyncio.get_event_loop()

        # Create stream reader for stdin
        self._reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self._reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        # Create stream writer for stdout
        transport, _ = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        self._writer = asyncio.StreamWriter(transport, protocol, self._reader, loop)

        logger.info("Stdio transport started")

        try:
            await self._read_loop()
        except Exception as e:
            logger.error(f"Transport error: {e}")
            raise
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the transport."""
        self._running = False
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()

    async def _read_loop(self) -> None:
        """Main read loop for processing incoming messages."""
        assert self._reader is not None

        buffer = b""

        while self._running:
            try:
                # Read data from stdin
                chunk = await self._reader.read(4096)
                if not chunk:
                    logger.info("EOF received, shutting down")
                    break

                buffer += chunk

                # Process complete messages from buffer
                while True:
                    message, buffer = self._extract_message(buffer)
                    if message is None:
                        break

                    # Process message asynchronously
                    asyncio.create_task(self._handle_message(message))

            except asyncio.CancelledError:
                logger.info("Transport cancelled")
                break
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                break

    def _extract_message(self, buffer: bytes) -> tuple[dict[str, Any] | None, bytes]:
        """
        Extract a complete JSON-RPC message from the buffer.

        MCP uses newline-delimited JSON messages.

        Args:
            buffer: Current buffer contents

        Returns:
            Tuple of (message dict or None, remaining buffer)
        """
        # Look for newline-delimited JSON
        newline_pos = buffer.find(b"\n")
        if newline_pos == -1:
            return None, buffer

        line = buffer[:newline_pos]
        remaining = buffer[newline_pos + 1:]

        if not line.strip():
            return None, remaining

        try:
            message = json.loads(line.decode("utf-8"))
            return message, remaining
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
            return None, remaining

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """
        Handle a single JSON-RPC message.

        Args:
            message: The parsed JSON-RPC message
        """
        try:
            # Dispatch to server
            response = await self.server.handle_message(message)

            if response is not None:
                await self._send_response(response)

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            # Send error response if we have a request ID
            if "id" in message:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {e}",
                    },
                }
                await self._send_response(error_response)

    async def _send_response(self, response: dict[str, Any]) -> None:
        """
        Send a JSON-RPC response.

        Args:
            response: The response to send
        """
        if self._writer is None:
            logger.error("Cannot send response: writer not initialized")
            return

        try:
            response_json = json.dumps(response)
            response_bytes = (response_json + "\n").encode("utf-8")

            self._writer.write(response_bytes)
            await self._writer.drain()

            logger.debug(f"Sent response: {response.get('id', 'notification')}")

        except Exception as e:
            logger.error(f"Error sending response: {e}")


async def run_stdio_server(server: "ICRMCPServer") -> None:
    """
    Run the MCP server with stdio transport.

    This is the main entry point for running the server in stdio mode,
    which is the default for Claude Code integration.

    Args:
        server: The MCP server instance to run
    """
    transport = StdioTransport(server)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        asyncio.create_task(transport.stop())

    # Register signal handlers (Unix only)
    try:
        import signal

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except (ImportError, NotImplementedError):
        # Windows doesn't support add_signal_handler
        pass

    try:
        await transport.start()
    finally:
        await transport.stop()
