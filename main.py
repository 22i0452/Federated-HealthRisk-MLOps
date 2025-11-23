import uvicorn
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Health Risk Prediction API server...")
    
    is_production = os.getenv("REPLIT_DEPLOYMENT") == "1"
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=5000,
        reload=not is_production,
        log_level="info"
    )
