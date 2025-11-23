MLOps Health Risk Prediction System (Federated Learning + FastAPI)

This project builds a complete MLOps pipeline for predicting health risk levels using wearable activity data, heart-rate data, sleep data, weather, and air-quality indicators. The goal is to simulate a real public-health scenario where data stays at different hospitals or cities, so the model is trained using Federated Learning instead of combining all data in one place.

The system covers data ingestion, preprocessing, federated training on three nodes, FedAvg aggregation, drift detection, global model evaluation, FastAPI deployment, and minimal CI/CD automation. The deployed API also includes a simple browser dashboard for manual predictions and visual analytics.

Main work included:
- merging activity, sleep, heart-rate, pollution, and weather data
- generating a health-risk score and converting it into Low/Medium/High classes
- training three separate MLP models on three simulated nodes
- fixing layer-shape mismatches and performing FedAvg to create the global model
- evaluating the global model with accuracy, confusion matrix, and classification report
- logging basic experiments (accuracies and confusion matrix) using MLflow
- checking for data drift with a simple z-score method
- deploying the global model with FastAPI
- creating a dashboard page using HTML, JS, and a simple form interface
- setting up Dockerfile and a GitHub Actions workflow for automated checks

Input features used by the model:
Total steps, distance, active minutes, calories, average heart-rate, total sleep minutes, PM25, PM10, temperature, humidity.

API summary:
POST /predict takes the ten input values as JSON and returns both the risk code (0,1,2) and the risk label (Low, Medium, High).
GET /dashboard shows a simple UI for manual risk prediction and small visual graphs.

To run locally:
pip install -r requirements.txt
uvicorn fastapi_app:app --host 0.0.0.0 --port 5000
Dashboard runs at /dashboard.

Deployment was done on Replit with autoscale enabled. The Dockerfile allows containerizing the project, and the GitHub workflow installs dependencies, loads the global model, and performs a test prediction.

Node-level accuracy is around 0.86–0.90. The global model performs around 0.90 accuracy. Confusion matrix, error plots, feature–risk plots, and drift charts are saved in the images folder.

Prepared for MLOps course final project.
