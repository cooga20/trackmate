from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys, os, random, glob, math, json

app = FastAPI(title="TrackMate API", version="2.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
sys.path.insert(0, root)

PURPLE_LINE = [
    "Challaghatta","Kengeri","Kengeri Bus Terminal","Pattanagere",
    "Jnanabharathi","Rajarajeshwari Nagar","Nayandahalli","Mysuru Road",
    "Deepanjali Nagar","Attiguppe","Vijayanagar","Hosahalli",
    "Magadi Road","City Railway Station","Majestic (KSR)","Sir M Visvesvaraya",
    "Vidhana Soudha","Cubbon Park","MG Road","Trinity","Halasuru",
    "Indiranagar","Swami Vivekananda Road","Baiyappanahalli",
    "Benniganahalli","KR Pura","Singayyanapalya","Garudacharapalya",
    "Hoodi","Seetharampalya","Kundalahalli","Nallurhalli",
    "Sri Sathya Sai Hospital","Pattandur Agrahara","Kadugodi Tree Park",
    "Hopefarm Channasandra","Whitefield (Kadugodi)"
]

GREEN_LINE = [
    "Madavara","Chikkabidarakallu","Manjunatha Nagar","Nagasandra",
    "Dasarahalli","Jalahalli","Peenya Industry","Peenya",
    "Goraguntepalya","Yeshwanthpur","Sandal Soap Factory","Mahalakshmi",
    "Rajajinagar","Kuvempu Road","Srirampura","Mantri Square Sampige Road",
    "Majestic (KSR)","Chickpete","Krishna Rajendra Market","National College",
    "Lalbagh","Rashtreeya Vidyalaya Road","South End Circle","Jayanagar",
    "Jayanagar 4th Block","Yelachenahalli","Doddakallasandra",
    "Konankunte Cross","Vajrahalli","Thalaghattapura","Silk Institute"
]

YELLOW_LINE = [
    "Rashtreeya Vidyalaya Road","Ragigudda","Jayadeva Hospital","BTM Layout",
    "Central Silk Board","Bommanahalli","Hongasandra","Kudlu Gate",
    "Singasandra","Hosa Road","Beratena Agrahara","Electronic City",
    "Konappana Agrahara","Huskur Road","Biocon Hebbagodi","Bommasandra"
]

STATION_COORDS = {
    "Challaghatta":{"lat":12.9022,"lng":77.4921,"line":"purple","parking":True},
    "Kengeri":{"lat":12.9107,"lng":77.4872,"line":"purple","parking":True},
    "Kengeri Bus Terminal":{"lat":12.9127,"lng":77.4892,"line":"purple","parking":False},
    "Pattanagere":{"lat":12.9156,"lng":77.4921,"line":"purple","parking":False},
    "Jnanabharathi":{"lat":12.9178,"lng":77.4968,"line":"purple","parking":True},
    "Rajarajeshwari Nagar":{"lat":12.9228,"lng":77.5012,"line":"purple","parking":False},
    "Nayandahalli":{"lat":12.9285,"lng":77.5089,"line":"purple","parking":False},
    "Mysuru Road":{"lat":12.9342,"lng":77.5156,"line":"purple","parking":True},
    "Deepanjali Nagar":{"lat":12.9398,"lng":77.5198,"line":"purple","parking":False},
    "Attiguppe":{"lat":12.9445,"lng":77.5234,"line":"purple","parking":False},
    "Vijayanagar":{"lat":12.9512,"lng":77.5289,"line":"purple","parking":False},
    "Hosahalli":{"lat":12.9556,"lng":77.5345,"line":"purple","parking":False},
    "Magadi Road":{"lat":12.9601,"lng":77.5398,"line":"purple","parking":False},
    "City Railway Station":{"lat":12.9768,"lng":77.5724,"line":"purple","parking":False},
    "Majestic (KSR)":{"lat":12.9766,"lng":77.5713,"line":"both","parking":False},
    "Sir M Visvesvaraya":{"lat":12.9775,"lng":77.5739,"line":"purple","parking":False},
    "Vidhana Soudha":{"lat":12.9789,"lng":77.5751,"line":"purple","parking":False},
    "Cubbon Park":{"lat":12.9806,"lng":77.5787,"line":"purple","parking":False},
    "MG Road":{"lat":12.9761,"lng":77.6059,"line":"purple","parking":False},
    "Trinity":{"lat":12.9734,"lng":77.6089,"line":"purple","parking":False},
    "Halasuru":{"lat":12.9712,"lng":77.6123,"line":"purple","parking":False},
    "Indiranagar":{"lat":12.9784,"lng":77.6408,"line":"purple","parking":True},
    "Swami Vivekananda Road":{"lat":12.9834,"lng":77.6467,"line":"purple","parking":False},
    "Baiyappanahalli":{"lat":12.9886,"lng":77.6526,"line":"purple","parking":True},
    "Benniganahalli":{"lat":12.9912,"lng":77.6612,"line":"purple","parking":False},
    "KR Pura":{"lat":13.0012,"lng":77.6789,"line":"purple","parking":True},
    "Singayyanapalya":{"lat":13.0056,"lng":77.6834,"line":"purple","parking":False},
    "Garudacharapalya":{"lat":13.0089,"lng":77.6867,"line":"purple","parking":False},
    "Hoodi":{"lat":13.0123,"lng":77.6934,"line":"purple","parking":True},
    "Seetharampalya":{"lat":13.0156,"lng":77.7012,"line":"purple","parking":False},
    "Kundalahalli":{"lat":13.0189,"lng":77.7067,"line":"purple","parking":False},
    "Nallurhalli":{"lat":13.0234,"lng":77.7123,"line":"purple","parking":False},
    "Sri Sathya Sai Hospital":{"lat":13.0267,"lng":77.7178,"line":"purple","parking":False},
    "Pattandur Agrahara":{"lat":13.0312,"lng":77.7234,"line":"purple","parking":False},
    "Kadugodi Tree Park":{"lat":13.0345,"lng":77.7289,"line":"purple","parking":False},
    "Hopefarm Channasandra":{"lat":13.0378,"lng":77.7345,"line":"purple","parking":False},
    "Whitefield (Kadugodi)":{"lat":13.0423,"lng":77.7467,"line":"purple","parking":True},
    "Madavara":{"lat":13.1089,"lng":77.5234,"line":"green","parking":True},
    "Chikkabidarakallu":{"lat":13.0934,"lng":77.5278,"line":"green","parking":False},
    "Manjunatha Nagar":{"lat":13.0789,"lng":77.5312,"line":"green","parking":False},
    "Nagasandra":{"lat":13.0634,"lng":77.5356,"line":"green","parking":True},
    "Dasarahalli":{"lat":13.0489,"lng":77.5389,"line":"green","parking":False},
    "Jalahalli":{"lat":13.0345,"lng":77.5412,"line":"green","parking":False},
    "Peenya Industry":{"lat":13.0234,"lng":77.5234,"line":"green","parking":False},
    "Peenya":{"lat":13.0178,"lng":77.5198,"line":"green","parking":True},
    "Goraguntepalya":{"lat":13.0089,"lng":77.5178,"line":"green","parking":False},
    "Yeshwanthpur":{"lat":13.0023,"lng":77.5289,"line":"green","parking":True},
    "Sandal Soap Factory":{"lat":12.9978,"lng":77.5434,"line":"green","parking":False},
    "Mahalakshmi":{"lat":12.9923,"lng":77.5512,"line":"green","parking":False},
    "Rajajinagar":{"lat":12.9867,"lng":77.5567,"line":"green","parking":False},
    "Kuvempu Road":{"lat":12.9812,"lng":77.5612,"line":"green","parking":False},
    "Srirampura":{"lat":12.9778,"lng":77.5645,"line":"green","parking":False},
    "Mantri Square Sampige Road":{"lat":12.9756,"lng":77.5689,"line":"green","parking":False},
    "Chickpete":{"lat":12.9712,"lng":77.5723,"line":"green","parking":False},
    "Krishna Rajendra Market":{"lat":12.9667,"lng":77.5734,"line":"green","parking":False},
    "National College":{"lat":12.9612,"lng":77.5756,"line":"green","parking":False},
    "Lalbagh":{"lat":12.9534,"lng":77.5801,"line":"green","parking":False},
    "Rashtreeya Vidyalaya Road":{"lat":12.9489,"lng":77.5834,"line":"green-yellow","parking":False},
    "South End Circle":{"lat":12.9423,"lng":77.5867,"line":"green","parking":False},
    "Jayanagar":{"lat":12.9345,"lng":77.5934,"line":"green","parking":False},
    "Jayanagar 4th Block":{"lat":12.9289,"lng":77.5978,"line":"green","parking":False},
    "Yelachenahalli":{"lat":12.9123,"lng":77.6045,"line":"green","parking":True},
    "Doddakallasandra":{"lat":12.9067,"lng":77.6089,"line":"green","parking":False},
    "Konankunte Cross":{"lat":12.8978,"lng":77.6123,"line":"green","parking":False},
    "Vajrahalli":{"lat":12.8845,"lng":77.6178,"line":"green","parking":False},
    "Thalaghattapura":{"lat":12.8712,"lng":77.6234,"line":"green","parking":False},
    "Silk Institute":{"lat":12.8578,"lng":77.6289,"line":"green","parking":True},
    "Ragigudda":{"lat":12.9345,"lng":77.5978,"line":"yellow","parking":False},
    "Jayadeva Hospital":{"lat":12.9186,"lng":77.5966,"line":"yellow","parking":False},
    "BTM Layout":{"lat":12.9153,"lng":77.6101,"line":"yellow","parking":False},
    "Central Silk Board":{"lat":12.9172,"lng":77.6228,"line":"yellow","parking":False},
    "Bommanahalli":{"lat":12.9068,"lng":77.6231,"line":"yellow","parking":False},
    "Hongasandra":{"lat":12.8967,"lng":77.6252,"line":"yellow","parking":False},
    "Kudlu Gate":{"lat":12.8886,"lng":77.6288,"line":"yellow","parking":False},
    "Singasandra":{"lat":12.8811,"lng":77.6317,"line":"yellow","parking":False},
    "Hosa Road":{"lat":12.8698,"lng":77.6379,"line":"yellow","parking":False},
    "Beratena Agrahara":{"lat":12.8567,"lng":77.6434,"line":"yellow","parking":False},
    "Electronic City":{"lat":12.8452,"lng":77.6602,"line":"yellow","parking":True},
    "Konappana Agrahara":{"lat":12.8398,"lng":77.6689,"line":"yellow","parking":False},
    "Huskur Road":{"lat":12.8323,"lng":77.6756,"line":"yellow","parking":False},
    "Biocon Hebbagodi":{"lat":12.8234,"lng":77.6823,"line":"yellow","parking":False},
    "Bommasandra":{"lat":12.8159,"lng":77.6889,"line":"yellow","parking":True},
}

def haversine(lat1,lng1,lat2,lng2):
    R=6371
    dlat=math.radians(lat2-lat1)
    dlng=math.radians(lng2-lng1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlng/2)**2
    return R*2*math.asin(math.sqrt(a))

def get_fare(km):
    if km<=2: return 10
    elif km<=4: return 15
    elif km<=6: return 20
    elif km<=8: return 25
    elif km<=12: return 30
    elif km<=16: return 35
    elif km<=20: return 40
    elif km<=24: return 45
    elif km<=28: return 50
    elif km<=32: return 55
    else: return 60

# ── PRE-COMPUTE density results at startup ──────────────────────
# Instead of running YOLO every request (30s), we run it ONCE
# for 10 different coach combinations and cache them.
# Each request just picks a random cached result — instant!

model = None
DENSITY_CACHE = []   # list of pre-computed coach results

def make_coach_result(coach_id, level, pct):
    return {
        "coach": f"C{coach_id}",
        "density_pct": pct,
        "level": level,
        "ladies_only": coach_id == 1
    }

def build_density_cache_with_model():
    """Run YOLO on images and cache 10 result sets"""
    from PIL import Image
    import shutil
    global DENSITY_CACHE

    all_val = glob.glob('data/processed/val/images/*.jpg')
    if not all_val:
        print("No val images found — using simulated cache")
        build_simulated_cache()
        return

    random.shuffle(all_val)

    # Find images by density level
    low_imgs, med_imgs, high_imgs = [], [], []
    print("Scanning images for cache (this runs once)...")
    for img_path in all_val[:200]:
        r = model(img_path, verbose=False, conf=0.25)
        count = len(r[0].boxes)
        if count < 15 and len(low_imgs) < 15:
            low_imgs.append((img_path, count))
        elif 15 <= count < 50 and len(med_imgs) < 15:
            med_imgs.append((img_path, count))
        elif count >= 50 and len(high_imgs) < 15:
            high_imgs.append((img_path, count))
        if len(low_imgs)>=15 and len(med_imgs)>=15 and len(high_imgs)>=15:
            break

    # Fallback if not enough
    while len(low_imgs) < 5:
        low_imgs.append((random.choice(all_val), 3))
    while len(med_imgs) < 5:
        med_imgs.append((random.choice(all_val), 25))
    while len(high_imgs) < 5:
        high_imgs.append((random.choice(all_val), 80))

    # Pre-run YOLO on chosen images and save annotated results
    RD = os.path.join(root, 'results')
    os.makedirs(RD, exist_ok=True)

    # Save annotated images once
    for img_path, count in (low_imgs[:3] + med_imgs[:3] + high_imgs[:3]):
        r = model(img_path, verbose=False, conf=0.25)
        count = len(r[0].boxes)
        if count < 15: level = "low"; pct = max(5, round((count/180)*100))
        elif count < 50: level = "medium"; pct = round((count/180)*100)
        else: level = "high"; pct = min(100, round((count/180)*100))
        ann = r[0].plot()
        sp = os.path.join(RD, f'cached_{level}_{pct}pct.jpg')
        Image.fromarray(ann).save(sp)

    # Build 10 different result combinations
    for _ in range(10):
        pool = [
            (random.choice(low_imgs)[1], "low"),
            (random.choice(low_imgs)[1], "low"),
            (random.choice(med_imgs)[1], "medium"),
            (random.choice(med_imgs)[1], "medium"),
            (random.choice(high_imgs)[1], "high"),
            (random.choice(high_imgs)[1], "high"),
        ]
        random.shuffle(pool)
        coaches = {}
        for i, (count, level) in enumerate(pool):
            cid = i + 1
            if level == "low": pct = max(5, round((count/180)*100))
            elif level == "medium": pct = round((count/180)*100)
            else: pct = min(100, round((count/180)*100))
            coaches[f'C{cid}'] = make_coach_result(cid, level, pct)
        DENSITY_CACHE.append(coaches)

    print(f"Density cache built: {len(DENSITY_CACHE)} combinations ready")

def build_simulated_cache():
    """Fallback: simulate realistic density without model"""
    global DENSITY_CACHE
    import random
    hour = __import__('datetime').datetime.now().hour
    is_peak = (7 <= hour <= 10) or (17 <= hour <= 20)

    for _ in range(10):
        levels_pool = ["low","low","medium","medium","high","high"]
        random.shuffle(levels_pool)
        coaches = {}
        for i, level in enumerate(levels_pool):
            cid = i + 1
            if level == "low":
                pct = random.randint(5, 35)
            elif level == "medium":
                pct = random.randint(45, 68) if not is_peak else random.randint(55, 70)
            else:
                pct = random.randint(75, 98) if is_peak else random.randint(70, 90)
            coaches[f'C{cid}'] = make_coach_result(cid, level, pct)
        DENSITY_CACHE.append(coaches)
    print(f"Simulated cache built: {len(DENSITY_CACHE)} combinations ready")

# ── Startup ─────────────────────────────────────────────────────
try:
    print("Loading YOLO model...")
    from ultralytics import YOLO
    model = YOLO('model/weights/best.pt')
    print("Model loaded! Building density cache...")
    build_density_cache_with_model()
except Exception as e:
    print(f"WARNING: Model load failed ({e})")
    print("Building simulated density cache as fallback...")
    build_simulated_cache()

print(f"Startup complete. Cache size: {len(DENSITY_CACHE)}")

# ── ENDPOINTS ────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "app": "TrackMate",
        "status": "running",
        "version": "2.0",
        "model_loaded": model is not None,
        "cache_size": len(DENSITY_CACHE)
    }

@app.get("/density/{train_id}")
def get_density(train_id: str):
    """Returns pre-computed density — instant response"""
    try:
        if not DENSITY_CACHE:
            build_simulated_cache()
        coaches = random.choice(DENSITY_CACHE)
        return {
            "train_id": train_id,
            "coaches": coaches,
            "total_coaches": 6,
            "model_used": model is not None
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/eta")
def get_eta(from_station: str, to_station: str):
    try:
        if from_station in PURPLE_LINE and to_station in PURPLE_LINE:
            fi = PURPLE_LINE.index(from_station)
            ti = PURPLE_LINE.index(to_station)
            stops = abs(ti - fi)
            km = round(stops * 1.17, 1)
            direction = "eastbound" if ti > fi else "westbound"
        elif from_station in GREEN_LINE and to_station in GREEN_LINE:
            fi = GREEN_LINE.index(from_station)
            ti = GREEN_LINE.index(to_station)
            stops = abs(ti - fi)
            km = round(stops * 0.77, 1)
            direction = "southbound" if ti > fi else "northbound"
        elif from_station in YELLOW_LINE and to_station in YELLOW_LINE:
            fi = YELLOW_LINE.index(from_station)
            ti = YELLOW_LINE.index(to_station)
            stops = abs(ti - fi)
            km = round(stops * 1.28, 1)
            direction = "southbound" if ti > fi else "northbound"
        else:
            stops = 12; km = 15.0
            direction = "via Majestic/RV Road interchange"
        fare = get_fare(km)
        return {
            "from": from_station, "to": to_station,
            "stops": stops,
            "eta_minutes": round(stops * 2.5),
            "distance_km": km,
            "fare": fare,
            "direction": direction,
            "status": "success"
        }
    except ValueError as e:
        return {"error": str(e), "status": "failed"}

@app.get("/stations")
def get_stations():
    return {"purple_line": PURPLE_LINE, "green_line": GREEN_LINE, "yellow_line": YELLOW_LINE}

@app.get("/trains")
def get_trains(station: str):
    trains = []
    for i in range(1, 4):
        trains.append({
            "train_id": f"420{i}",
            "eta_minutes": random.randint(2, 20),
            "status": "on time",
            "direction": "towards Baiyappanahalli"
        })
    trains.sort(key=lambda x: x["eta_minutes"])
    return {"station": station, "upcoming_trains": trains}
@app.get("/nearest_station")
def nearest_station(lat: float, lng: float):
    """Returns top 3 nearest stations with accurate distance"""
    results = []
    for name, info in STATION_COORDS.items():
        d = haversine(lat, lng, info["lat"], info["lng"])
        results.append({
            "station": name,
            "distance_km": round(d, 2),
            "line": info["line"],
            "has_parking": info["parking"],
            "lat": info["lat"],
            "lng": info["lng"]
        })
    # Sort by distance
    results.sort(key=lambda x: x["distance_km"])
    top3 = results[:3]

    # Travel mode for nearest
    for r in top3:
        d = r["distance_km"]
        if d < 0.5: r["travel_mode"] = "🚶 Walk (~" + str(round(d*12)) + " min)"
        elif d < 1.5: r["travel_mode"] = "🚶 Walk (~" + str(round(d*12)) + " min)"
        elif d < 3: r["travel_mode"] = "🛺 Auto (~₹" + str(round(d*15)) + ")"
        elif d < 8: r["travel_mode"] = "🛺 Auto/Cab (~₹" + str(round(d*12)) + ")"
        else: r["travel_mode"] = "🚌 Bus/Cab (~₹" + str(round(d*10)) + ")"

    return {
        "nearest": top3[0]["station"],
        "top3": top3,
        "user_lat": lat,
        "user_lng": lng
    }

@app.get("/fare")
def fare(from_station: str, to_station: str):
    try:
        if from_station in PURPLE_LINE and to_station in PURPLE_LINE:
            stops = abs(PURPLE_LINE.index(from_station) - PURPLE_LINE.index(to_station))
            km = round(stops * 1.17, 1)
        elif from_station in GREEN_LINE and to_station in GREEN_LINE:
            stops = abs(GREEN_LINE.index(from_station) - GREEN_LINE.index(to_station))
            km = round(stops * 0.77, 1)
        elif from_station in YELLOW_LINE and to_station in YELLOW_LINE:
            stops = abs(YELLOW_LINE.index(from_station) - YELLOW_LINE.index(to_station))
            km = round(stops * 1.28, 1)
        else:
            stops = 12; km = 15.0
        f = get_fare(km)
        return {
            "from": from_station, "to": to_station,
            "stops": stops, "distance_km": km,
            "fare_single": f, "fare_return": f * 2,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}
    
@app.get("/density_by_station/{train_id}/{station_idx}")
def get_density_by_station(train_id: str, station_idx: int):
    """Returns density that varies based on station index — simulates real passenger loading"""
    try:
        # Density pattern changes based on station:
        # Early stations (terminus) = less crowded
        # Middle stations (city centre) = most crowded
        # Late stations = medium
        import datetime
        hour = datetime.datetime.now().hour
        is_peak = (7 <= hour <= 10) or (17 <= hour <= 20)

        # Station index affects density pattern
        idx = station_idx % 10  # cycle through patterns

        patterns = {
            0: ["low","low","medium","low","low","medium"],
            1: ["low","medium","low","medium","low","low"],
            2: ["medium","low","medium","high","medium","low"],
            3: ["medium","medium","high","medium","low","medium"],
            4: ["high","medium","high","medium","high","medium"],
            5: ["high","high","medium","high","medium","high"],
            6: ["medium","high","high","medium","high","medium"],
            7: ["medium","medium","medium","high","medium","low"],
            8: ["low","medium","medium","low","medium","medium"],
            9: ["low","low","low","medium","low","low"],
        }

        level_pattern = patterns.get(idx, patterns[0])
        if is_peak:
            # Push everything one level up during peak
            upgrade = {"low":"medium","medium":"high","high":"high"}
            level_pattern = [upgrade[l] for l in level_pattern]

        random.shuffle(level_pattern)
        coaches = {}
        for i, level in enumerate(level_pattern):
            cid = i + 1
            if level == "low": pct = random.randint(5, 35)
            elif level == "medium": pct = random.randint(42, 68)
            else: pct = random.randint(73, 98)
            coaches[f'C{cid}'] = {
                "coach": f"C{cid}",
                "density_pct": pct,
                "level": level,
                "ladies_only": cid == 1
            }
        return {"train_id": train_id, "coaches": coaches, "station_idx": station_idx}
    except Exception as e:
        return {"error": str(e)}

@app.get("/open_results")
def open_results():
    try:
        import subprocess
        rp = os.path.join(root, 'results')
        os.makedirs(rp, exist_ok=True)
        subprocess.Popen(f'explorer "{rp}"')
        return {"status": "opened"}
    except:
        return {"status": "failed"}