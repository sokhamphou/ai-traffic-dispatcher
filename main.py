import os
import json
import time
import html
import base64
import re
import asyncio
from typing import Optional
from enum import Enum
from shapely.geometry import Point, shape

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(title="AI Traffic Dispatcher")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class IncidentStatus(str, Enum):
    PENDING = "Pending"
    SCENE = "Go to Scene"
    COMPLETE = "Complete"

class StatusUpdate(BaseModel):
    status: IncidentStatus

def verify_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return authorization

INCIDENTS_DB = []

# ==========================================
# AGENT 2: GeoJSON Spatial Router
# ==========================================
# A lightweight proof-of-concept GeoJSON for Phnom Penh boundaries
PHNOM_PENH_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"khan": "Khan Toul Kork"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[104.88, 11.57], [104.90, 11.57], [104.90, 11.55], [104.88, 11.55], [104.88, 11.57]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"khan": "Khan Daun Penh"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[104.90, 11.58], [104.93, 11.58], [104.93, 11.56], [104.90, 11.56], [104.90, 11.58]]]
            }
        }
    ]
}

def geo_route_incident(lat: float, lon: float, hazard_score: int, category: str):
    """Maps GPS to Khan using GeoJSON polygons and determines multi-agency routing."""
    point = Point(lon, lat) # Shapely uses (x, y) -> (lon, lat)
    detected_khan = "Unknown District (Out of Bounds)"
    
    # Point-in-Polygon check against official GIS boundaries
    for feature in PHNOM_PENH_GEOJSON["features"]:
        polygon = shape(feature["geometry"])
        if polygon.contains(point):
            detected_khan = feature["properties"]["khan"]
            break

    # Intelligent Multi-Agency Routing
    agency = "Traffic Police"
    if hazard_score > 7 or "injury" in category.lower():
        agency = "Emergency Medical Services (EMS) & Traffic Police"
    elif "flood" in category.lower() or "pothole" in category.lower() or "light" in category.lower():
        agency = "Department of Public Works"

    return detected_khan, agency

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")

@app.get("/api/incidents")
async def get_incidents(khan: Optional[str] = None, auth: str = Depends(verify_auth)):
    if khan and khan != "ALL":
        return [inc for inc in INCIDENTS_DB if inc["khan"] == khan]
    return INCIDENTS_DB

class GeminiExtraction(BaseModel):
    category: str = Field(description="Type of incident, e.g., Motorbike Collision, Flooding")
    extracted_lat: float = Field(description="Latitude coordinate")
    extracted_lon: float = Field(description="Longitude coordinate")
    obstruction_percent: int = Field(description="Percentage of road blocked (0-100)")
    injury_hazard_score: int = Field(description="Severity score from 1 to 10")
    ocr_text: str = Field(description="Text read from license plates or signs")
    detected_vehicles: str = Field(description="Types of vehicles seen in media")
    narrative_en: str = Field(description="Professional dispatch summary in English")
    narrative_km: str = Field(description="Professional dispatch summary in Khmer")

@app.post("/api/report")
async def create_report(
    gps: str = Form(...),
    voice_note: str = Form(""),
    demo_type: str = Form(""), # Restored missing parameter
    media: Optional[UploadFile] = File(None),
    voice_audio: Optional[UploadFile] = File(None)
):
    if not gps or not gps.strip():
        raise HTTPException(status_code=422, detail="GPS location is required.")

    safe_gps = html.escape(gps)
    api_key = os.environ.get("GEMINI_API_KEY")
    current_time = time.strftime("%I:%M %p").lstrip("0")
    result_data = {}
    
    media_preview_url = "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=800&q=80"
    file_bytes = None
    media_mime = "image/jpeg"

    if media:
        try:
            file_bytes = await media.read()
            media_mime = media.content_type or "image/jpeg"
            if file_bytes and len(file_bytes) < 5_000_000:
                b64_img = base64.b64encode(file_bytes).decode("utf-8")
                media_preview_url = f"data:{media_mime};base64,{b64_img}"
        except Exception:
            pass

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing.")

    # AGENT 1: Multimodal Intake with Strict Schema & Auto-Retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
            Analyze this incident report. Location text: "{safe_gps}". Voice note: "{voice_note}". Hint: "{demo_type}".
            
            Perform deep multimodal extraction:
            1. Perform OCR on any visible vehicle license plates or street signs in Khmer & English.
            2. Detect all vehicle types (e.g., tuk-tuk, motorbike, SUV, container truck) and estimate road obstruction percentage (0-100%).
            3. Assess the injury hazard urgency on a 1-10 scale.
            4. Extract the numeric latitude and longitude coordinates from the location text if present.
            """
            contents = [prompt]

            if file_bytes:
                contents.append(types.Part.from_bytes(data=file_bytes, mime_type=media_mime))
            if voice_audio:
                audio_bytes = await voice_audio.read()
                if audio_bytes:
                    contents.append(types.Part.from_bytes(data=audio_bytes, mime_type=voice_audio.content_type or "audio/webm"))

            # Force strict JSON schema matching our Pydantic model
            response = client.models.generate_content(
                model="gemini-3.8-flash", 
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiExtraction,
                )
            )
            
            result_data = json.loads(response.text)
            break  # Break out of the loop if successful
            
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                print(f"503 High Demand detected. Retrying... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(2)  # Wait 2 seconds before trying again
            else:
                print(f"Gemini AI Error: {e}")
                result_data = {
                    "narrative_en": f"System Alert: AI extraction failed. Officer dispatch required to verify manual report at {safe_gps}.",
                    "narrative_km": "ការជូនដំណឹងពីប្រព័ន្ធ៖ បរាជ័យក្នុងការវិភាគ AI។ សូមបញ្ជូនមន្ត្រីចុះត្រួតពិនិត្យ។",
                    "category": "Unverified Incident"
                }

    # Bulletproof data extraction (Prevents crashes if Gemini returns text or null instead of numbers)
    try:
        lat = float(result_data.get("extracted_lat") or 11.5681)
    except (ValueError, TypeError):
        lat = 11.5681

    try:
        lon = float(result_data.get("extracted_lon") or 104.8921)
    except (ValueError, TypeError):
        lon = 104.8921

    try:
        hazard_raw = str(result_data.get("injury_hazard_score", "5"))
        hazard_match = re.search(r'\d+', hazard_raw)
        hazard = int(hazard_match.group()) if hazard_match else 5
    except Exception:
        hazard = 5

    cat = str(result_data.get("category", "Incident"))
    
    # Execute Agent 2 (Spatial Router) safely
    try:
        routed_khan, recommended_agency = geo_route_incident(lat, lon, hazard, cat)
    except Exception as e:
        print(f"Geo routing error: {e}")
        routed_khan, recommended_agency = "Khan Toul Kork", "Traffic Police"

    new_incident = {
        "id": f"INC-2026-{1000 + len(INCIDENTS_DB) + 1}",
        "timestamp": current_time,
        "gps": safe_gps,
        "lat": lat, # Added for GIS Map
        "lon": lon, # Added for GIS Map
        "khan": routed_khan,
        "sangkat": "Auto-Mapped via GIS",
        "agency": recommended_agency,
        "category": cat,
        "severity": f"Score: {hazard}/10",
        "obstruction_percent": result_data.get("obstruction_percent", 0),
        "ocr_text": result_data.get("ocr_text", "None detected"),
        "detected_vehicles": result_data.get("detected_vehicles", "None"),
        "narrative_en": result_data.get("narrative_en", "Manual review needed."),
        "narrative_km": result_data.get("narrative_km", "ត្រូវការត្រួតពិនិត្យ។"),
        "status": "Pending",
        "media_preview": media_preview_url
    }
    INCIDENTS_DB.insert(0, new_incident)
    return new_incident

@app.patch("/api/incidents/{incident_id}/status")
async def update_status(incident_id: str, payload: StatusUpdate, auth: str = Depends(verify_auth)):
    current_time = time.strftime("%I:%M %p").lstrip("0")
    for inc in INCIDENTS_DB:
        if inc["id"] == incident_id:
            inc["status"] = payload.status.value
            
            # Telemetry: Record exact timestamps based on status change
            if payload.status.value == "Go to Scene" and "dispatch_time" not in inc:
                inc["dispatch_time"] = current_time
            elif payload.status.value == "Complete" and "resolve_time" not in inc:
                inc["resolve_time"] = current_time
                
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")