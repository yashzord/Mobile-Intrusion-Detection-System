#!/usr/bin/env python3
from mitmproxy.io import FlowReader
from mitmproxy.http import HTTPFlow
import json
import pandas as pd

def safe_headers(headers):
    return {str(k): str(v) for k, v in headers.items()}

def safe_json(obj):
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    return str(obj)

flows = []

with open("/root/secure_traffic_data/flows.mitm", "rb") as f:
    reader = FlowReader(f)
    for flow in reader.stream():
        if isinstance(flow, HTTPFlow):
            # Skip flows that don't have both timestamps
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

with open("/root/secure_traffic_data/flows.json", "w") as f:
    json.dump(flows, f, indent=2, default=safe_json)

df = pd.DataFrame(flows)
df.to_csv("/root/secure_traffic_data/flows.csv", index=False)
print(f"✅ Extracted {len(df)} HTTP flows and saved to CSV/JSON.")

# Preview sample rows
print("\n🔎 Sample rows:")
print(df[["url", "method", "status_code", "content_type", "user_agent", "tls_cipher", "sni"]].sample(5))
