# Deployment

This repository is now split for separate frontend and backend deployment.

## Services

- Backend: FastAPI app served by `uvicorn api.main:app`
- Frontend: Vite static build from `frontend/`

## Local Development

Backend:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server proxies `/api` requests to `BACKEND_BASE_URL` from the root `.env`. If `BACKEND_BASE_URL` is unset, it falls back to `http://localhost:8000`.

## Required Environment Variables

Backend:

- `BACKEND_BASE_URL`: Public backend origin without `/api` suffix. Example: `https://gammashield-api.onrender.com`
- `FRONTEND_BASE_URL`: Public frontend origin. Example: `https://gammashield-client.onrender.com`
- `CORS_ALLOWED_ORIGINS`: Comma-separated allowed origins for browser calls. Usually set this to the frontend URL.
- `CORS_ALLOW_CREDENTIALS`: Optional. Defaults to `false`.

Frontend:

- `VITE_API_BASE_URL`: Public backend API base URL including `/api`. Example: `https://gammashield-api.onrender.com/api`

## Auth Flow

- Frontend sends users to `${VITE_API_BASE_URL}/auth/login`
- Backend redirects Kite OAuth callbacks to `${BACKEND_BASE_URL}/api/auth/callback`
- Backend sends successful auth back to `${FRONTEND_BASE_URL}/login?auth=success`

## Render

`render.yaml` defines:

- `gammashield-api` as a Python web service
- `gammashield-client` as a static site built from `frontend/dist`

After creating both services, set:

- On `gammashield-api`: `BACKEND_BASE_URL`, `FRONTEND_BASE_URL`, `CORS_ALLOWED_ORIGINS`
- On `gammashield-client`: `VITE_API_BASE_URL`
