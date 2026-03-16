"""Async UDP client for the CL1 stimulation interface.

Sends binary stim packets to CL1_HOST:STIM_PORT and receives spike packets
on LISTEN_HOST:SPIKE_PORT.  Socket recv runs in a thread-pool executor so it
doesn't block the asyncio event loop during the artifact-rejection wait.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import List

import config
from cl1 import protocol

logger = logging.getLogger(__name__)


class CL1Interface:
    """Async context manager wrapping a UDP socket pair for CL1 communication.

    Usage:
        async with CL1Interface() as cl1:
            spikes = await cl1.stimulate(frequencies, amplitudes)
    """

    def __init__(
        self,
        cl1_host: str = config.CL1_HOST,
        stim_port: int = config.STIM_PORT,
        listen_host: str = config.LISTEN_HOST,
        spike_port: int = config.SPIKE_PORT,
    ):
        self._stim_addr = (cl1_host, stim_port)      # neural interface — stim destination
        self._listen_addr = (listen_host, spike_port) # our local bind — spike receive
        self._sock: socket.socket | None = None
        self._seq = 0

    async def __aenter__(self) -> "CL1Interface":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(self._listen_addr)
        self._sock.setblocking(False)  # required for loop.sock_recvfrom()

        # Suppress spurious ConnectionResetError on Windows when the remote
        # port is briefly unreachable (same fix as the reference test client).
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            try:
                self._sock.ioctl(socket.SIO_UDP_CONNRESET, False)
            except OSError:
                pass

        logger.info(
            "CL1 UDP socket ready: stim -> %s:%d  spikes <- %s:%d  (device: %s)",
            *self._stim_addr,
            *self._listen_addr,
            config.CL1_DEVICE,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None
            logger.info("CL1 UDP socket closed")

    async def stimulate(
        self, frequencies: List[float], amplitudes: List[float]
    ) -> List[float]:
        """Send a stim packet and return spike counts for all TOTAL_CHANNELS channels.

        Waits SPIKE_ARTIFACT_WAIT_S after transmitting before reading the spike
        response, to avoid contamination by stimulation artifacts.

        Returns a list of length TOTAL_CHANNELS (dead channels will be 0.0).
        """
        assert self._sock is not None, "CL1Interface used outside of async context"

        packet = protocol.pack_stim(frequencies, amplitudes)
        send_ts = protocol.now_us()
        self._sock.sendto(packet, self._stim_addr)
        self._seq += 1

        # Artifact rejection window
        await asyncio.sleep(config.SPIKE_ARTIFACT_WAIT_S)

        # Non-blocking coroutine recv — cancellable, no stale executor threads
        loop = asyncio.get_event_loop()
        raw, _ = await asyncio.wait_for(
            loop.sock_recvfrom(self._sock, protocol.SPIKE_PACKET_SIZE),
            timeout=config.UDP_TIMEOUT_S,
        )

        spike_ts, spike_counts = protocol.unpack_spike(raw)
        logger.debug(
            "seq=%d rtt=%.1f ms spike_ts_latency=%.1f ms spikes_mean=%.3f",
            self._seq,
            protocol.latency_ms(send_ts),
            protocol.latency_ms(spike_ts),
            sum(spike_counts) / len(spike_counts),
        )
        return spike_counts
