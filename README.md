# MedLingua

MedLingua extracts English text from uploaded PDFs and images and stores each result under the authenticated account.

## Run locally

1. Start PostgreSQL: `docker compose up -d postgres`.
2. Copy `backend/.env.example` to `backend/.env` and replace `JWT_SECRET_KEY`.
3. Install backend dependencies: `cd backend` then `python -m pip install -r requirements.txt`.
4. Apply migrations: `alembic upgrade head`.
5. Start the API from CMD: `python -m uvicorn app.main:app --reload --port 8000`. Using `python -m` avoids Windows policies that may block the `uvicorn.exe` launcher.
6. Copy `frontend/.env.local.example` to `frontend/.env.local`.
7. Install and start the frontend: `cd frontend`, `npm install`, then `npm run dev`.

Open `http://localhost:3000`. The first PaddleOCR run downloads its English model files. If Windows blocks PaddleOCR's native dependency, the backend automatically falls back to Tesseract OCR. Install it from CMD with `winget install --id UB-Mannheim.TesseractOCR -e`, then restart the backend. If it is installed outside PATH, set `TESSERACT_CMD` in `backend/.env`, for example `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`.

On Windows CMD, paste only the commands inside a code block; do not paste the Markdown fence characters (the lines containing three backticks).

Uploads are stored in `backend/uploads` by default. Configure limits and paths through `backend/.env`.

PDF OCR ordering is layout-aware: when PaddleOCR detects two populated columns, it reads the complete left column from top to bottom and then the complete right column instead of interleaving both columns line-by-line.

The `demo` folder contains `demo.tex`, a clean single-column sample patient document. Install MiKTeX or TeX Live and compile it from CMD with:

```cmd
cd /d "C:\Users\tanis\Documents\VSC Projects\Python Projects\MedLingua\demo"
pdflatex demo.tex
```
