"""Loopback HTTP policy service; one inference at a time, no waiting queue."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
import time

import numpy as np

from .protocol import MAX_BYTES, VERSION, actions50, observation, pack, unpack


class PolicyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, policy, metadata):
        self.policy, self.metadata = policy, {'version': VERSION, **metadata}
        self.inference_lock = threading.Lock()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, message, *args):
        logging.info(message, *args)

    def send(self, status, data, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/healthz':
            self.send(200, json.dumps(self.server.metadata).encode())
        else:
            self.send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != '/infer':
            self.send(404, b'{"error":"not found"}')
            return
        acquired = False
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BYTES or self.headers.get('Transfer-Encoding'):
                raise ValueError('请求体长度错误')
            if self.headers.get('Content-Type') != 'application/x-npz':
                raise ValueError('Content-Type 必须为 application/x-npz')
            payload = self.rfile.read(size)
            if len(payload) != size:
                raise ValueError('请求体不完整')
            arrays, metadata = unpack(payload)
            obs = observation(arrays, metadata)
            acquired = self.server.inference_lock.acquire(blocking=False)
            if not acquired:
                self.send(503, b'{"error":"inference busy; retry with a new observation"}')
                return
            start = time.monotonic()
            actions = actions50(self.server.policy.infer(obs)['actions'])
            reply = {**self.server.metadata, 'request_id': metadata['request_id'],
                     'infer_s': time.monotonic() - start}
            self.send(200, pack({'actions': actions}, reply), 'application/x-npz')
        except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
            self.send(400, json.dumps({'error': str(error)}, ensure_ascii=False).encode())
        except Exception:
            logging.exception('模型推理失败')
            self.send(500, b'{"error":"policy inference failed; see server log"}')
        finally:
            if acquired:
                self.server.inference_lock.release()


class MockPolicy:
    def infer(self, obs):
        return {'actions': np.repeat(obs['state'][None], 50, axis=0)}
