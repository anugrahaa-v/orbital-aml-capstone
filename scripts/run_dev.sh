#!/usr/bin/env bash
# Terminal 1: API.  Terminal 2: `cd frontend && npm install --legacy-peer-deps && npm run dev` (http://localhost:5173)
# Or just open http://localhost:8000 : the API serves the prebuilt frontend/dist.
uvicorn backend.main:app --reload --port 8000
