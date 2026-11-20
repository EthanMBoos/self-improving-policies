"""openpi's scripts/serve_policy.py with websocket keepalive pings turned off. Same arguments.

openpi runs inference inside the server's event loop, so while JAX compiles or 8 clients queue up, the loop cannot
process pongs and the server closes every connection with "1011 keepalive ping timeout". Clients are local, so
keepalive is not needed.

  python serve.py --port 8000 policy:checkpoint --policy.config=pi05_libero --policy.dir=...  (run from openpi/)
"""
import functools
import runpy
import sys

import websockets.asyncio.server as _server

_server.serve = functools.partial(_server.serve, ping_interval=None)
sys.argv[0] = "scripts/serve_policy.py"
runpy.run_path("scripts/serve_policy.py", run_name="__main__")
