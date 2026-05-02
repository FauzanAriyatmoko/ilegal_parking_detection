import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from backend.app.ai_manager import AIManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
ai_manager = AIManager()

@app.on_event("startup")
async def startup_event():
    logger.info("Backend Server Started")

@app.on_event("shutdown")
async def shutdown_event():
    ai_manager.stop_all()

@app.post("/camera/{camera_id}/start")
async def start_camera(camera_id: str):
    success = ai_manager.start_camera(camera_id)
    return {"status": "started" if success else "failed or already running"}

@app.post("/camera/{camera_id}/stop")
async def stop_camera(camera_id: str):
    success = ai_manager.stop_camera(camera_id)
    return {"status": "stopped" if success else "not running"}

@app.get("/camera/{camera_id}/status")
async def get_status(camera_id: str):
    return {"is_running": ai_manager.is_running(camera_id)}

@app.get("/camera/{camera_id}/stream")
async def video_stream(camera_id: str):
    return StreamingResponse(ai_manager.generate_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")
