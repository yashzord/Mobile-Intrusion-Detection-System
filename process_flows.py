#!/usr/bin/env python3
"""
process_flows.py

Extracts HTTP flows captured by mitmproxy from the file flows.mitm (located in secure_traffic_data/)
and saves the processed flows as CSV and JSON files. Optionally, an output filename can be provided.
Usage: python3 process_flows.py [output_csv]
"""

import os
import sys
import json
import pandas as pd
from mitmproxy.io import FlowReader
from mitmproxy.http import HTTPFlow

def safe_headers(headers):
    return {str(k): str(v) for k, v in headers.items()}

def safe_json(obj):
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    return str(obj)

base_dir = os.path.dirname(__file__)
flows_mitm_path = os.path.join(base_dir, "secure_traffic_data", "flows.mitm")

# Determine output CSV filename
if len(sys.argv) > 1:
    csv_filename = sys.argv[1]
    csv_path = os.path.join(base_dir, "secure_traffic_data", csv_filename)
else:
    csv_path = os.path.join(base_dir, "secure_traffic_data", "flows.csv")

flows = []
with open(flows_mitm_path, "rb") as f:
    reader = FlowReader(f)
    for flow in reader.stream():
        if isinstance(flow, HTTPFlow):
            if flow.request.timestamp_start is None or flow.request.timestamp_end is None:
                continue
            flows.append({
                "timestamp_start": flow.request.timestamp_start,
                "timestamp_end": flow.request.timestamp_end,
                "duration": flow.request.timestamp_end - flow.request.timestamp_start,
                "url": flow.request.pretty_url,
                "method": flow.request.method,
                "http_version": flow.request.http_version,
                "request_headers": safe_headers(flow.request.headers),
                "request_content_length": int(flow.request.headers.get("content-length", 0)),
                "user_agent": flow.request.headers.get("user-agent", "unknown"),
                "status_code": float(flow.response.status_code) if flow.response else None,
                "response_headers": safe_headers(flow.response.headers) if flow.response else {},
                "response_content_length": int(flow.response.headers.get("content-length", 0)) if flow.response else 0,
                "content_type": flow.response.headers.get("content-type", "") if flow.response else "",
                "content_encoding": flow.response.headers.get("content-encoding", "") if flow.response else "",
                "client_ip": flow.client_conn.peername[0],
                "client_port": flow.client_conn.peername[1],
                "server_ip": flow.server_conn.peername[0] if flow.server_conn and flow.server_conn.peername else None,
                "server_port": flow.server_conn.peername[1] if flow.server_conn and flow.server_conn.peername else None,
                "tls_established": safe_json(getattr(flow.server_conn, "tls_established", None)),
                "tls_cipher": safe_json(getattr(flow.server_conn, "cipher_name", None)),
                "alpn": safe_json(getattr(flow.server_conn, "alpn", None)),
                "sni": safe_json(getattr(flow.server_conn, "sni", None)),
            })

pd.DataFrame(flows).to_csv(csv_path, index=False)
print(f"Extracted {len(flows)} HTTP flows and saved CSV at {csv_path}")

json_path = os.path.join(base_dir, "secure_traffic_data", "flows.json")
with open(json_path, "w") as f:
    json.dump(flows, f, indent=2, default=safe_json)
print(f"Saved flows data to JSON at {json_path}")
