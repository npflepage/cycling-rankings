#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.r2"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found."
  echo "Copy .env.r2.template to .env.r2 and fill in your credentials."
  exit 1
fi

# Load R2 credentials (skip comments/blank lines, handle CRLF and = in values)
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line//$'\r'/}"                   # strip Windows carriage returns
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line// }" ]] && continue
  key="${line%%=*}"
  value="${line#*=}"
  export "$key=$value"
done < "$ENV_FILE"

# Validate required vars
for var in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY CF_ACCOUNT_ID R2_BUCKET_NAME; do
  if [[ -z "${!var:-}" ]]; then
    echo "Error: $var is not set in $ENV_FILE"
    exit 1
  fi
done

cd "$ROOT_DIR"

echo "==> Running model export..."
uv run python model/export_for_web.py \
  --relabel-iters 4 \
  --data-dir ./data \
  --out-dir ./docs/outputs \
  --min-races 2

echo "==> Syncing to Cloudflare R2..."
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
AWS_DEFAULT_REGION=auto \
AWS_ENDPOINT_URL="https://${CF_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  aws s3 sync docs/outputs/ "s3://${R2_BUCKET_NAME}/" \
    --content-type application/json \
    --cache-control "public, max-age=3600" \
    --no-progress

echo "==> Done. Commit and push docs/ to update Cloudflare Pages."
