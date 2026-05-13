from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import random

app = FastAPI(title="TrackMate API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

STATIONS = [
    "Nagasandra", "Dasarahalli", "Jalahalli",
    "Peenya Industry", "Peenya", "Yeshwanthpur",
    "Sandal Soap Factory", "Mahalakshmi",
    "Rajajinagar", "Kuvempu Road", "Srirampura",
    "Mantri Square Sampige Road", "Majestic (KSR)",
    "Cubbon Park", "MG Road", "Trinity",
    "Halasuru", "Indiranagar", "Swami Vivekananda Road",
    "Baiyappanahalli"
]

@app.get("/")
def home():
    return {
        "app": "TrackMate",
        "description": "Namma Metro Passenger Density Estimator",
        "version": "1.0",
        "status": "running"
    }

@app.get("/density/{train_id}")
def get_density(train_id: str):
    try:
        # Change to project root directory
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root)
        sys.path.insert(0, root)

        from model.infer import get_all_coaches
        coaches = get_all_coaches()
        return {
            "train_id": train_id,
            "coaches": coaches,
            "total_coaches": len(coaches)
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }

@app.get("/eta")
def get_eta(from_station: str, to_station: str):
    try:
        from_idx = STATIONS.index(from_station)
        to_idx = STATIONS.index(to_station)
        stops = abs(to_idx - from_idx)
        eta_minutes = round(stops * 2.5)
        direction = "eastbound" if to_idx > from_idx else "westbound"
        return {
            "from": from_station,
            "to": to_station,
            "stops": stops,
            "eta_minutes": eta_minutes,
            "direction": direction,
            "status": "success"
        }
    except ValueError as e:
        return {
            "error": f"Station not found: {str(e)}",
            "status": "failed"
        }

@app.get("/stations")
def get_stations():
    return {
        "stations": STATIONS,
        "total": len(STATIONS)
    }

@app.get("/trains")
def get_trains(station: str):
    trains = []
    for i in range(1, 4):
        eta = random.randint(2, 20)
        trains.append({
            "train_id": f"420{i}",
            "eta_minutes": eta,
            "status": "on time",
            "direction": "towards Baiyappanahalli"
        })
    trains.sort(key=lambda x: x["eta_minutes"])
    return {
        "station": station,
        "upcoming_trains": trains
    }