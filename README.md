# PROJECT-SUZKA

## Research Philosophy

- [PROJECT-SUZKA — 設計思想・研究原則](PHILOSOPHY.md): research purpose, human-like cognition, imperfection-versus-defect criteria, and architectural decision boundaries.

## Local Runbook

- Start the API with `just api`; it serves FastAPI on the `api.host` and `api.port` values from `config.yaml`.
- Run backend tests with `uv run pytest`.
- Run frontend tests from `frontend/` with `nix develop --command npm test`.
- Build the frontend from `frontend/` with `nix develop --command npm run build`.

## Admin Access

- Normal chat remains public at `POST /api/chat`.
- There is no application-level authentication layer. Debug, memory inspection, sleep, adapter, Context, Value, and R16 endpoints are open to whoever can reach the listener; deploy on a private LAN/VPN or behind your own access boundary (state-changing routes have no Origin/Host check outside `/api/r18/*`, so do not browse untrusted pages from a machine that can reach the listener).
- The R18 approval API (`/api/r18/*`) keeps its own localhost/trusted-network, Host/Origin/header, JSON, rate-limit, and typed-code guards.
- Frontend admin pages call the Next.js `/admin-proxy/*` route, which forwards an allowlisted set of paths to the backend without credentials and never reaches `/api/r18/*`.
- A legacy `api.admin_token_env` key in an existing config file is still accepted but ignored (one deprecation warning at startup); a stray `X-SUZKA-Admin-Token` header is ignored.

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

Edit both env files. Set `NEXT_PUBLIC_API_BASE_URL` to the browser-visible private origin and keep `SUZKA_BACKEND_URL` pointed at the private FastAPI listener.

For SSH-tunnel access, run `ssh -L 18080:127.0.0.1:8080 user@host`, set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:18080`, rebuild the frontend, and open `http://127.0.0.1:18080` locally.

Admin warning: neither the backend nor the admin pages have authentication or user login/session handling. Keep this listener bound to loopback, or put it behind VPN, SSO, basic auth, or another access-control layer.

Build the frontend after the env file is configured because `NEXT_PUBLIC_API_BASE_URL` is embedded at build time:

```bash
sudo -u suzka bash -lc 'set -a; source /etc/project-suzka/frontend.env; set +a; cd /opt/project-suzka/frontend && npm run build'
```

### SubjectHistory (one-way adoption)

`config.yaml` enables `subject_history` (IP-10b, #376 H2). The first clean
startup after updating to it adopts the SubjectHistory store
(`/opt/project-suzka/.suzka/private/subject_history/subject_history.sqlite3`)
into the EventJournal **irreversibly**:

- Back up the whole `/opt/project-suzka/.suzka/` directory before the first
  startup on a version that enables it.
- After adoption the store and the Journal are inseparable. Never delete
  `.suzka/` (no `git clean -fdx`, no fresh re-clone in place), and never
  restore one without the other.
- Do not go back to a commit with `subject_history.enabled: false` after
  adoption; startup then fails closed.
- Do not run the test suite inside `/opt/project-suzka`.
- Rows are never deleted; once `max_page_count` (65536 pages, 256 MiB) is
  reached, later Decisions HOLD with `HISTORY_CAPACITY`.

### Experiments never run against production

Chat turns become the subject's permanent experience, and after SubjectHistory
adoption they cannot be removed. Run real-model experiments on a throwaway copy
of the production state instead:

```bash
scripts/lab-from-production.sh --timeout 3600 -- python3 my_experiment.py
```

The script copies `/opt/project-suzka/.suzka/` and `/etc/project-suzka/config.yaml`
into `/var/lib/suzka/lab/` and starts the deployed code there as a transient
systemd unit (`suzka-lab-*`) on `127.0.0.1:8101`. It then runs the command with
`SUZKA_LAB_URL` set, and on exit stops the unit and deletes the copy (`--keep`
keeps the copy). It loads the model a second time, so it needs that much free
GPU memory (about 7.7 GB for gemma-4-12B-it).

It refuses to run when the parsed settings contain any path that is absolute,
starts with `~` or leaves the copy, when the lab port is the production port,
and when the port is already in use. If the script itself is killed with
SIGKILL, stop the lab with `sudo systemctl stop 'suzka-lab-*'`.

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
scripts/smoke-private-deploy.sh http://127.0.0.1:8080
```

The smoke script verifies `/health`, public `/api/chat`, direct admin API access without any credential, and `/admin-proxy/*` forwarding through the frontend. Set `CHECK_ADMIN_PROXY=0` if you are checking only the FastAPI reverse proxy without the frontend service.

Normal chat is unauthenticated on the private listener at `POST /api/chat`; direct debug, memory, sleep, and adapter APIs are likewise unauthenticated. Keep the listener behind your private access boundary (LAN/VPN/SSH tunnel).
