# AI Traffic Dispatcher 🚦

An autonomous multi-agent civic response platform built for Cambodian urban traffic and accident management. Submitted for the **Google Cloud AI Builder Cup**.

## Architecture
* **Frontend:** HTML5, Tailwind CSS, Leaflet GIS (OpenStreetMap)
* **Backend:** Python, FastAPI, Google Cloud Run
* **AI Cognitive Engine:** Google Gemini 3.8-Flash Multimodal API
* **Database:** Firebase

## Core Features
1. **Snap & Go Citizen Intake:** Multimodal upload with Auto-GPS and Khmer/English voice notes.
2. **AI Forensics:** Gemini 3.8-Flash automatically translates audio, assesses hazard urgency, and calculates road obstruction percentages.
3. **Sangkat-Level Geo-Routing:** Deterministic mapping of GPS coordinates directly to Phnom Penh's 14 Khans and 105 Sangkats.
4. **Authority Command Center:** Real-time GIS map interface with telemetry tracking and automated workflow status.
