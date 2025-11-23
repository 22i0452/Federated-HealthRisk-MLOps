from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import sys
import joblib
import numpy as np
from numpy.random import MT19937
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load("global_model.pkl")
    logger.info(f"Model loaded successfully - Type: {type(model).__name__}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

labels = ["Low", "Medium", "High"]

app = FastAPI(
    title="Health Risk Prediction API",
    description="MLOps deployment for health risk assessment",
    version="1.0.0"
)

class HealthInput(BaseModel):
    TotalSteps: float
    TotalDistance: float
    VeryActiveMinutes: float
    Calories: float
    AvgHeartRate: float
    TotalSleepMinutes: float
    PM25: float
    PM10: float
    Temperature: float
    Humidity: float

    class Config:
        json_schema_extra = {
            "example": {
                "TotalSteps": 10000,
                "TotalDistance": 7.5,
                "VeryActiveMinutes": 30,
                "Calories": 2200,
                "AvgHeartRate": 75,
                "TotalSleepMinutes": 420,
                "PM25": 85,
                "PM10": 150,
                "Temperature": 28,
                "Humidity": 60
            }
        }

@app.get("/")
async def root():
    return {
        "message": "Health Risk Prediction API",
        "status": "online",
        "endpoints": {
            "dashboard": "/dashboard",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.post("/predict")
async def predict(data: HealthInput):
    try:
        x = np.array([
            data.TotalSteps,
            data.TotalDistance,
            data.VeryActiveMinutes,
            data.Calories,
            data.AvgHeartRate,
            data.TotalSleepMinutes,
            data.PM25,
            data.PM10,
            data.Temperature,
            data.Humidity
        ]).reshape(1, -1)

        prediction = model.predict(x)[0]
        risk_code = int(prediction)
        risk_level = labels[risk_code]

        logger.info(f"Prediction made: {risk_level} (code: {risk_code})")

        return {
            "risk_code": risk_code,
            "risk_level": risk_level,
            "success": True
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Assessment Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            justify-content: center;
        }
        
        .tab-button {
            padding: 12px 30px;
            background: rgba(255, 255, 255, 0.2);
            border: 2px solid white;
            color: white;
            cursor: pointer;
            border-radius: 25px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .tab-button:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        
        .tab-button.active {
            background: white;
            color: #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
            font-size: 14px;
        }
        
        input {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
            width: 100%;
            max-width: 300px;
            margin: 20px auto;
            display: block;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .result-box {
            margin-top: 30px;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            display: none;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-box.low {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            border: 3px solid #28a745;
        }
        
        .result-box.medium {
            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
            border: 3px solid #ff9800;
        }
        
        .result-box.high {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border: 3px solid #dc3545;
        }
        
        .result-title {
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .result-message {
            font-size: 1.2em;
            margin-top: 10px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 1em;
            opacity: 0.9;
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .alert {
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #dc3545;
            background: #f8d7da;
            color: #721c24;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏥 Health Risk Assessment Dashboard</h1>
            <p class="subtitle">MLOps Deployment System for Predictive Health Analytics</p>
        </header>

        <div class="tabs">
            <button class="tab-button active" onclick="switchTab('citizen')">Citizen Assessment</button>
            <button class="tab-button" onclick="switchTab('analytics')">Health Authority Analytics</button>
        </div>

        <div id="citizen-tab" class="tab-content active">
            <div class="card">
                <h2 style="margin-bottom: 20px; color: #333;">📋 Enter Your Health Data</h2>
                <form id="healthForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="steps">Total Steps</label>
                            <input type="number" id="steps" name="TotalSteps" value="10000" step="100" required>
                        </div>
                        <div class="form-group">
                            <label for="distance">Total Distance (km)</label>
                            <input type="number" id="distance" name="TotalDistance" value="7.5" step="0.1" required>
                        </div>
                        <div class="form-group">
                            <label for="active">Very Active Minutes</label>
                            <input type="number" id="active" name="VeryActiveMinutes" value="30" step="1" required>
                        </div>
                        <div class="form-group">
                            <label for="calories">Calories Burned</label>
                            <input type="number" id="calories" name="Calories" value="2200" step="10" required>
                        </div>
                        <div class="form-group">
                            <label for="hr">Average Heart Rate (bpm)</label>
                            <input type="number" id="hr" name="AvgHeartRate" value="75" step="1" required>
                        </div>
                        <div class="form-group">
                            <label for="sleep">Total Sleep (minutes)</label>
                            <input type="number" id="sleep" name="TotalSleepMinutes" value="420" step="5" required>
                        </div>
                        <div class="form-group">
                            <label for="pm25">PM2.5 Level (μg/m³)</label>
                            <input type="number" id="pm25" name="PM25" value="85" step="0.1" required>
                        </div>
                        <div class="form-group">
                            <label for="pm10">PM10 Level (μg/m³)</label>
                            <input type="number" id="pm10" name="PM10" value="150" step="0.1" required>
                        </div>
                        <div class="form-group">
                            <label for="temp">Temperature (°C)</label>
                            <input type="number" id="temp" name="Temperature" value="28" step="0.1" required>
                        </div>
                        <div class="form-group">
                            <label for="humidity">Humidity (%)</label>
                            <input type="number" id="humidity" name="Humidity" value="60" step="1" required>
                        </div>
                    </div>
                    <button type="submit" class="btn">🔍 Assess Health Risk</button>
                </form>
                
                <div id="resultBox" class="result-box">
                    <div class="result-title" id="resultTitle"></div>
                    <div class="result-message" id="resultMessage"></div>
                </div>
            </div>
        </div>

        <div id="analytics-tab" class="tab-content">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="totalPredictions">0</div>
                    <div class="stat-label">Total Assessments</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="lowRisk">0</div>
                    <div class="stat-label">Low Risk</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="mediumRisk">0</div>
                    <div class="stat-label">Medium Risk</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="highRisk">0</div>
                    <div class="stat-label">High Risk</div>
                </div>
            </div>

            <div class="chart-container">
                <h3 style="margin-bottom: 20px;">Risk Distribution</h3>
                <canvas id="riskChart"></canvas>
            </div>

            <div id="alertsSection"></div>
        </div>
    </div>

    <script>
        let predictions = JSON.parse(localStorage.getItem('predictions') || '[]');
        let chart = null;

        function switchTab(tab) {
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            if (tab === 'citizen') {
                document.querySelector('.tab-button:nth-child(1)').classList.add('active');
                document.getElementById('citizen-tab').classList.add('active');
            } else {
                document.querySelector('.tab-button:nth-child(2)').classList.add('active');
                document.getElementById('analytics-tab').classList.add('active');
                updateAnalytics();
            }
        }

        document.getElementById('healthForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });

            const resultBox = document.getElementById('resultBox');
            resultBox.style.display = 'block';
            resultBox.className = 'result-box';
            resultBox.innerHTML = '<div class="spinner"></div><p style="margin-top: 15px;">Analyzing health data...</p>';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                
                const riskLevel = result.risk_level.toLowerCase();
                resultBox.className = `result-box ${riskLevel}`;
                
                let icon = '';
                let message = '';
                
                if (riskLevel === 'low') {
                    icon = '✅';
                    message = 'Your health indicators are within normal range. Keep up the good work!';
                } else if (riskLevel === 'medium') {
                    icon = '⚠️';
                    message = 'Some health indicators need attention. Consider consulting a healthcare professional.';
                } else {
                    icon = '🚨';
                    message = 'Several health indicators are concerning. Please seek medical advice promptly.';
                }
                
                resultBox.innerHTML = `
                    <div class="result-title">${icon} ${result.risk_level} Risk</div>
                    <div class="result-message">${message}</div>
                `;

                predictions.push({
                    timestamp: new Date().toISOString(),
                    risk_level: result.risk_level,
                    risk_code: result.risk_code,
                    data: data
                });
                localStorage.setItem('predictions', JSON.stringify(predictions));

            } catch (error) {
                resultBox.className = 'result-box high';
                resultBox.innerHTML = `
                    <div class="result-title">❌ Error</div>
                    <div class="result-message">Failed to get prediction: ${error.message}</div>
                `;
            }
        });

        function updateAnalytics() {
            const stats = {
                total: predictions.length,
                low: predictions.filter(p => p.risk_code === 0).length,
                medium: predictions.filter(p => p.risk_code === 1).length,
                high: predictions.filter(p => p.risk_code === 2).length
            };

            document.getElementById('totalPredictions').textContent = stats.total;
            document.getElementById('lowRisk').textContent = stats.low;
            document.getElementById('mediumRisk').textContent = stats.medium;
            document.getElementById('highRisk').textContent = stats.high;

            if (chart) {
                chart.destroy();
            }

            const ctx = document.getElementById('riskChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                    datasets: [{
                        data: [stats.low, stats.medium, stats.high],
                        backgroundColor: [
                            'rgba(40, 167, 69, 0.8)',
                            'rgba(255, 152, 0, 0.8)',
                            'rgba(220, 53, 69, 0.8)'
                        ],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                font: {
                                    size: 14
                                }
                            }
                        }
                    }
                }
            });

            const alertsSection = document.getElementById('alertsSection');
            if (stats.high > 0) {
                alertsSection.innerHTML = `
                    <div class="alert">
                        <strong>⚠️ Alert:</strong> ${stats.high} high-risk case(s) detected. Immediate attention may be required.
                    </div>
                `;
            } else {
                alertsSection.innerHTML = '';
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)
