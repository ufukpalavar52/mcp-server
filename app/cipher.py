"""
Opens the credentials that arrive with a definition.

This service holds no keys. A definition carries its model's API key as ciphertext plus
the id of the key that sealed it; reading one means asking mcp-cipher. The consequence is
the point — a key entered once in the panel is the key used here, instead of a second copy
living in this process's environment where it has to be kept in step by hand.

Values are cached for the life of the process. A definition is planned many times and the
key does not change between calls; asking the cipher on every one would add a round trip
to every plan for no benefit. The cache is keyed by the ciphertext, so a rotated or
replaced key is a different entry rather than a stale hit.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc

from app.models import SealedSecret
from app.pb import cipher_pb2, cipher_pb2_grpc

logger = logging.getLogger(__name__)

TOKEN_HEADER = "x-cipher-token"


class CipherUnavailable(Exception):
    """The cipher could not be reached, or refused."""


class CipherClient:
    """Decrypts sealed values through mcp-cipher."""

    def __init__(self, address: str, token: str = "", timeout: float = 5.0) -> None:
        self._address = address
        self._token = token
        self._timeout = timeout
        self._channel: Any = None
        self._stub: Any = None
        self._cache: dict[bytes, str] = {}

    @property
    def configured(self) -> bool:
        return bool(self._address)

    def open(self, sealed: SealedSecret) -> str:
        """
        Decrypts one value.

        Raises rather than returning an empty string: a caller that silently proceeded with
        no key would produce a 401 from the model provider, three steps from the cause.
        """
        if not self.configured:
            raise CipherUnavailable("No cipher configured; set MCP_CIPHER_ADDRESS")

        if sealed.ciphertext in self._cache:
            return self._cache[sealed.ciphertext]

        try:
            response = self._call().Decrypt(
                cipher_pb2.DecryptRequest(
                    ciphertext=sealed.ciphertext,
                    context=sealed.context,
                    key_id=sealed.key_id,
                ),
                timeout=self._timeout,
                metadata=self._metadata(),
            )
        except grpc.RpcError as exc:
            # The cipher is deliberately vague about why; passing its message through keeps
            # it that way rather than inventing a more specific explanation here.
            raise CipherUnavailable(
                f"The credential could not be opened: {exc.code().name}"
            ) from exc

        plaintext = response.plaintext.decode()
        self._cache[sealed.ciphertext] = plaintext
        return plaintext

    def reachable(self) -> bool:
        """Whether the cipher answers, for the start up log and /health."""
        if not self.configured:
            return False

        try:
            self._call().Keys(
                cipher_pb2.KeysRequest(), timeout=self._timeout, metadata=self._metadata()
            )
            return True
        except grpc.RpcError:
            return False

    def _call(self) -> Any:
        # Built lazily so importing this module does not open a socket, which matters for
        # the tests that never use it.
        if self._stub is None:
            self._channel = grpc.insecure_channel(self._address)
            self._stub = cipher_pb2_grpc.CipherServiceStub(self._channel)
        return self._stub

    def _metadata(self) -> list[tuple[str, str]]:
        return [(TOKEN_HEADER, self._token)] if self._token else []

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
