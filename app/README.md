
#  App Setup Guide

This guide will help you set up and run the UI application, which consists of a FastAPI backend and a React/TypeScript frontend.

---

## 1. Configure API Keys

Go into the `backend` directory and add your API keys to the `.env` file. Example `.env` contents:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_HOST=your_pinecone_host
OPENAI_API_KEY=your_openai_api_key
````

Make sure the `.env` file is saved before running the app.

---

## 2. Start the App

Make sure the `start` script is executable:

```bash
chmod +x ./start
```

Then run the script:

```bash
./start
```

What this script does:

1. Activates the Python virtual environment for the backend.
2. Installs backend dependencies from `requirements.txt`.
3. Checks if port **8000** is in use, kills any process if necessary, and starts the FastAPI backend.
4. Sets up the frontend dependencies (`node_modules`) if missing.
5. Starts the frontend server.

**Notes:**

* Backend runs on port **8000**.
* Frontend runs on port **3000**.
* To run only the backend, use:

```bash
./startb
```

---

## 3. App Structure

```
app/
│
├─ backend/           # FastAPI backend
│   ├─ main.py        # Main API entrypoint
│   ├─ routers/       # API route definitions (e.g., Pinecone, ChatGPT)
│   ├─ support/       # Helper modules (metadata)
│   └─ venv/          # Python virtual environment (created automatically)
│
├─ src/               # Frontend source code (TypeScript / React)
├─ public/            # Frontend assets
│
├─ start              # Bash script to start both backend & frontend
└─ startb             # Bash script to start backend only
```

**Backend:** Handles all APIs, Pinecone queries, OpenAI requests, and case law processing.
**Frontend:** React/TypeScript app connecting to the backend via API calls. The UI is available at `http://localhost:3000`.

---

## 4. Accessing the App

* Frontend: Open your browser at [http://localhost:3000](http://localhost:3000)
* Backend APIs: Available at [http://localhost:8000](http://localhost:8000)

---

## 5. Troubleshooting

1. **Port 8000 already in use:** The start script kills the existing process. If you encounter errors, manually check using:

```bash
lsof -i :8000
```

and kill any leftover process:

```bash
kill -9 <PID>
```

2. **Frontend dependencies conflict:** Use legacy peer deps if necessary:

```bash
npm install --legacy-peer-deps
```

3. **Backend virtual environment missing:** The script will create it automatically, but you can manually create it with:

```bash
python3 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt
```


