# PROJECT-SUZKA

## Local Runbook

- Start the API with `just api`; it serves FastAPI on the `api.host` and `api.port` values from `config.yaml`.
- Run backend tests with `uv run pytest`.
- Run frontend tests from `frontend/` with `nix develop --command npm test`.
- Build the frontend from `frontend/` with `nix develop --command npm run build`.

## Admin Access

- Normal chat remains public at `POST /api/chat`.
- Debug, memory inspection, sleep, and adapter endpoints require `X-SUZKA-Admin-Token`.
- The expected token is read from the env var named by `api.admin_token_env`; the default is `SUZKA_ADMIN_TOKEN`.
- Frontend admin pages call the Next.js `/admin-proxy/*` route, which injects `SUZKA_ADMIN_TOKEN` server-side; the token is not included in browser bundles.

## Model Provider

- `config.yaml` defaults to the safe `dummy` provider.
- Real model smoke should use the Transformers provider and model IDs from `config.yaml` only.
- External LLM providers such as Ollama, OpenAI, Gemini API, and Claude API are intentionally unsupported.

## Release Checklist

- `uv run pytest`
- `nix develop --command npm test` from `frontend/`
- `nix develop --command npm run build` from `frontend/`
- `timeout 5s just api || test $? -eq 124 -o $? -eq 143`
- Search for forbidden provider implementation paths.
- Verify normal API/UI responses do not expose `hidden_thought`, raw prompts, retrieved memory, or `<think>` tags.

## Private Deployment

PROJECT-SUZKA is intended to run as a private/local application, not as a public website. The deployment target is a single Linux host with FastAPI bound to `127.0.0.1:8000`, Next.js bound to `127.0.0.1:3000`, and nginx or Caddy bound to loopback for local or SSH-tunnel access.

### 1. Prepare Host

Create a service user and install the required runtime tools:

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin suzka
sudo mkdir -p /opt/project-suzka /etc/project-suzka
sudo chown -R suzka:suzka /opt/project-suzka /etc/project-suzka
```

Install `git`, `uv`, `nodejs` 22, `npm`, and either `nginx` or `caddy` using your OS package manager or Nix profile. For the real Transformers provider, verify NVIDIA drivers/CUDA before switching away from the default `dummy` provider.

### 2. Install Application

```bash
sudo -u suzka git clone <repo-url> /opt/project-suzka
cd /opt/project-suzka
sudo -u suzka git switch develop
sudo -u suzka uv sync
cd frontend
sudo -u suzka npm ci
```

### 3. Configure Environment

```bash
sudo cp deploy/env/backend.env.example /etc/project-suzka/backend.env
sudo cp deploy/env/frontend.env.example /etc/project-suzka/frontend.env
sudo chmod 600 /etc/project-suzka/*.env
sudo chown suzka:suzka /etc/project-suzka/*.env
```

Edit both env files and set the same long random `SUZKA_ADMIN_TOKEN`. Set `NEXT_PUBLIC_API_BASE_URL` to the browser-visible private origin and keep `SUZKA_BACKEND_URL` pointed at the private FastAPI listener.

For SSH-tunnel access, run `ssh -L 18080:127.0.0.1:8080 user@host`, set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:18080`, rebuild the frontend, and open `http://127.0.0.1:18080` locally.

Admin warning: the frontend no longer exposes the admin token to browser bundles, but admin pages still do not have user login/session handling. Keep this listener bound to loopback, or put it behind VPN, SSO, basic auth, or another access-control layer.

Build the frontend after the env file is configured because `NEXT_PUBLIC_API_BASE_URL` is embedded at build time:

```bash
sudo -u suzka bash -lc 'set -a; source /etc/project-suzka/frontend.env; set +a; cd /opt/project-suzka/frontend && npm run build'
```

### 4. Install Services

```bash
sudo cp deploy/systemd/suzka-api.service /etc/systemd/system/suzka-api.service
sudo cp deploy/systemd/suzka-frontend.service /etc/systemd/system/suzka-frontend.service
sudo systemctl daemon-reload
sudo systemctl enable --now suzka-api suzka-frontend
sudo systemctl status suzka-api suzka-frontend
```

### 5. Configure Reverse Proxy

For nginx:

```bash
sudo cp deploy/nginx/suzka.conf /etc/nginx/sites-available/suzka.conf
sudo ln -s /etc/nginx/sites-available/suzka.conf /etc/nginx/sites-enabled/suzka.conf
sudo nginx -t
sudo systemctl reload nginx
```

For Caddy, copy `deploy/caddy/Caddyfile` into your Caddy config path. The provided example binds to `127.0.0.1:8080` and is meant for local or SSH-tunnel access.

### 6. Verify Deployment

```bash
SUZKA_ADMIN_TOKEN=replace-with-long-random-token scripts/smoke-private-deploy.sh http://127.0.0.1:8080
```

The smoke script verifies `/health`, public `/api/chat`, direct admin API rejection without a token, direct admin API success with `X-SUZKA-Admin-Token`, and `/admin-proxy/*` forwarding through the frontend. Set `CHECK_ADMIN_PROXY=0` if you are checking only the FastAPI reverse proxy without the frontend service.

Normal chat is unauthenticated on the private listener at `POST /api/chat`; direct debug, memory, sleep, and adapter APIs require the admin token header. Frontend admin pages use `/admin-proxy/*` and should remain behind your private access boundary.
