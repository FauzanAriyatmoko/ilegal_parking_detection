import uvicorn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print("Menjalankan API Server Backend...")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
