#!/usr/bin/env bash
# =============================================================================
#  FRONTEND TLS terminator — run this ON the frontend VPS.
#  Puts an nginx HTTPS listener (self-signed) in front of the split-serving
#  proxy so the slot's on-chain endpoint can be https://.
#
#  Topology:
#     validators ── https:$PUBLIC_PORT ──▶ nginx (this) ──▶ 127.0.0.1:$BACKEND_PORT
#                                                            (neurons.miner proxy,
#                                                             started via --port $BACKEND_PORT)
#
#  The miner registers --endpoint https://$PUBLIC_IP:$PUBLIC_PORT and binds its
#  proxy on --port $BACKEND_PORT. Validators use verify=False, so self-signed is
#  fine. A self-signed cert for an IP host needs an IP SAN (set below).
# =============================================================================
set -euo pipefail

PUBLIC_IP="${PUBLIC_IP:?set PUBLIC_IP, e.g. 203.0.113.10}"
PUBLIC_PORT="${PUBLIC_PORT:-19101}"   # public https port = registered endpoint port
BACKEND_PORT="${BACKEND_PORT:-28082}" # internal proxy port = miner --port
CERT_DIR="${CERT_DIR:-/etc/nginx/certs}"

command -v nginx >/dev/null || { apt-get update -qq && apt-get install -y -qq nginx; }

mkdir -p "$CERT_DIR"
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout "$CERT_DIR/slot.key" -out "$CERT_DIR/slot.crt" \
  -days 825 -subj "/CN=$PUBLIC_IP" -addext "subjectAltName=IP:$PUBLIC_IP"
chmod 600 "$CERT_DIR/slot.key"

cat > /etc/nginx/conf.d/slot.conf <<CONF
server {
    listen $PUBLIC_PORT ssl;
    server_name $PUBLIC_IP;

    ssl_certificate     $CERT_DIR/slot.crt;
    ssl_certificate_key $CERT_DIR/slot.key;
    ssl_protocols       TLSv1.2 TLSv1.3;

    client_max_body_size 64m;

    location / {
        proxy_pass http://127.0.0.1:$BACKEND_PORT;
        proxy_http_version 1.1;
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Connection        "";
        # streaming (SSE) inference: no buffering, long timeouts
        proxy_buffering     off;
        proxy_cache         off;
        proxy_read_timeout  3600s;
        proxy_send_timeout  3600s;
    }
}
CONF

nginx -t
systemctl reload nginx || systemctl restart nginx

echo "TLS terminator up on https://$PUBLIC_IP:$PUBLIC_PORT -> 127.0.0.1:$BACKEND_PORT"
echo "Register the slot with --endpoint https://$PUBLIC_IP:$PUBLIC_PORT and run the"
echo "miner proxy with --port $BACKEND_PORT (see run-split-slot.example.sh)."
