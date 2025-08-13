import errno
import time
import os
from typing import List, Callable

import numpy as np
import scipy.io

import websockets
import asyncio


def load_dataset(mat_filepath):
    mat = scipy.io.loadmat(mat_filepath)
    csi_data = mat["csi"].T
    sample_counts = mat["nsamples"].flatten()
    input_shape = mat["dim"].flatten()
    class_labels = list(map(lambda s: s.strip().title(), mat["classnames"]))

    labels = []
    for class_idx in range(len(class_labels)):
        labels.extend([class_idx] * sample_counts[class_idx])

    return csi_data, np.array(labels), sample_counts, class_labels, input_shape


def read_nonblocking(file_path, buffer_size=100, timeout=0.1) -> List[str]:
    result_lines = []
    try:
        pipe_fd = os.open(file_path, os.O_RDONLY | os.O_NONBLOCK)
        grace_period = True

        while True:
            try:
                buffer = os.read(pipe_fd, buffer_size)
                if not buffer:
                    break
                decoded = buffer.decode("utf-8")
                lines = decoded.split("\n")
                result_lines.extend(lines)
            except OSError as e:
                if e.errno == 11 and grace_period:
                    time.sleep(timeout)
                    grace_period = False
                else:
                    break
    except OSError as e:
        if e.errno == errno.ENOENT:
            pipe_fd = None
        else:
            raise e

    if pipe_fd is not None:
        os.close(pipe_fd)

    return result_lines


class WebsocketBroadcastServer:
    CONNECTED_CLIENTS = set()

    def __init__(self, host: str, port: int, message_generator: Callable[[], str], broadcast_frequency: float) -> None:
        self.host = host
        self.port = port
        self.message_generator = message_generator
        self.broadcast_frequency = broadcast_frequency

    async def run(self):
        async with websockets.serve(self._handle_client, self.host, self.port):
            await self._broadcast_loop()

    async def _handle_client(self, websocket):
        self.CONNECTED_CLIENTS.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.CONNECTED_CLIENTS.remove(websocket)

    async def _send_to_client(self, websocket, message):
        try:
            await websocket.send(message)
        except websockets.ConnectionClosed:
            pass

    async def _broadcast(self, message):
        if message is None:
            return
        for websocket in self.CONNECTED_CLIENTS:
            asyncio.create_task(self._send_to_client(websocket, message))

    async def _broadcast_loop(self):
        while True:
            await asyncio.gather(
                self._broadcast(self.message_generator()),
                asyncio.sleep(1.0 / self.broadcast_frequency),
            )
