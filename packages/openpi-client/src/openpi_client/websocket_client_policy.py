import logging
import os
import time
from typing import Dict, Optional, Tuple

import msgpack
from typing_extensions import override
import websockets.exceptions
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {value!r}") from exc


def _timeout_arg(timeout_sec: float) -> Optional[float]:
    return None if timeout_sec <= 0 else timeout_sec


def _unpack_msgpack_frame(frame: bytes, label: str) -> Dict:
    try:
        return msgpack_numpy.unpackb(frame)
    except msgpack.ExtraData:
        unpacker = msgpack_numpy.Unpacker()
        unpacker.feed(frame)
        objects = list(unpacker)
        if not objects:
            raise
        logging.warning(
            "Received %s websocket frame with %d concatenated msgpack objects; using the first object.",
            label,
            len(objects),
        )
        return objects[0]


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        request_timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
    ) -> None:
        if host.startswith("wss://") or host.startswith("ws://"):
            self._uri = host
            port = None
        elif host.startswith("https://"):
            self._uri = "wss://" + host[len("https://") :]
            port = None
        elif host.startswith("http://"):
            self._uri = "ws://" + host[len("http://") :]
            port = None
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._request_timeout = (
            request_timeout if request_timeout is not None else _float_env("OPENPI_INFERENCE_TIMEOUT_SEC", 300.0)
        )
        self._connect_timeout = (
            connect_timeout if connect_timeout is not None else _float_env("OPENPI_CONNECT_TIMEOUT_SEC", 30.0)
        )
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=_timeout_arg(self._connect_timeout),
                )
                metadata = _unpack_msgpack_frame(conn.recv(timeout=_timeout_arg(self._connect_timeout)), "metadata")
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
        except websockets.exceptions.ConnectionClosed as exc:
            raise RuntimeError(
                "Policy server websocket closed before sending an inference request. "
                "Inspect the policy server logs; it may have crashed, restarted, or the proxy may have closed the socket."
            ) from exc
        start_time = time.monotonic()
        try:
            response = self._ws.recv(timeout=_timeout_arg(self._request_timeout))
        except TimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for policy inference response "
                f"after {self._request_timeout:.1f}s. The policy server may still be compiling JAX/XLA "
                "for the first inference, or it may be stuck. Wait for the server compile to finish, "
                "or increase OPENPI_INFERENCE_TIMEOUT_SEC."
            ) from exc
        except websockets.exceptions.ConnectionClosed as exc:
            raise RuntimeError(
                "Policy server websocket closed while waiting for an inference response. "
                "Inspect the policy server logs for the server-side traceback."
            ) from exc
        logging.info("Received policy inference response in %.1f ms", (time.monotonic() - start_time) * 1000)
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return _unpack_msgpack_frame(response, "inference response")

    @override
    def reset(self) -> None:
        pass
