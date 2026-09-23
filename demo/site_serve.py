# -*- coding: utf-8 -*-
"""Serve demo/site so the agent has something to drive. python demo/site_serve.py"""
import functools
import http.server
import os
import socketserver

PORT = int(os.environ.get("JOBE_DEMO_PORT", "8824"))
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
    print("serving %s at http://localhost:%d  (ctrl-c to stop)" % (root, PORT))
    httpd.serve_forever()
