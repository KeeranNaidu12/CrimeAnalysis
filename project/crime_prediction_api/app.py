"""
Crime Prediction API
====================
FastAPI backend for crime prediction models.
Provides REST endpoints for frontend integration.

Run with: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
import joblib
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import your existing orchestrator functions
try:
    from orchestrator import (
        load_all_models, predict_for_date_range, 
        CRIME_DISPLAY_NAMES, MODELS_DIR, MODEL_FILES,
        create_neighbourhood_mappings, extract_neighbourhood_id,
        clean_neighbourhood_name, find_neighbourhood_by_input
    )
except ImportError:
    from crime_prediction_api.orchestrator import (
        load_all_models, predict_for_date_range, 
        CRIME_DISPLAY_NAMES, MODELS_DIR, MODEL_FILES,
        create_neighbourhood_mappings, extract_neighbourhood_id,
        clean_neighbourhood_name, find_neighbourhood_by_input
    )

# Initialize FastAPI
app = FastAPI(
    title="Crime Prediction API",
    description="API for predicting crime incidents across neighbourhoods",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache
models_cache = None

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class CrimeType(str, Enum):
    COLLISION = "collision"
    ASSAULT = "assault"
    AUTO_THEFT = "auto_theft"
    BREAK_AND_ENTER = "break_and_enter"

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    crime_type: CrimeType
    start_date: date
    end_date: date
    neighbourhood_id: Optional[int] = Field(None, description="Neighbourhood ID (e.g., 20)")
    neighbourhood_name: Optional[str] = Field(None, description="Neighbourhood name (e.g., 'Alderwood')")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        days_diff = (v - values['start_date']).days + 1
        if days_diff < 3:
            raise ValueError('Date range must be at least 3 days')
        return v

class PredictionResult(BaseModel):
    """Individual prediction result"""
    window_start: str
    window_end: str
    neighbourhood: str
    neighbourhood_id: Optional[int]
    neighbourhood_clean: str
    probability: float
    predicted: bool
    risk_level: str

class PredictionSummary(BaseModel):
    """Aggregated prediction summary"""
    total_windows: int
    total_predictions: int
    positive_predictions: int
    positive_percentage: float
    high_risk_alerts: int
    medium_risk_alerts: int
    low_risk_alerts: int

class NeighbourhoodInfo(BaseModel):
    """Neighbourhood information"""
    id: int
    name: str
    display_name: str

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    crime_type: str
    crime_display_name: str
    date_range: Dict[str, str]
    neighbourhood_filter: Optional[str]
    summary: PredictionSummary
    results: List[PredictionResult]
    high_risk_alerts: List[PredictionResult]
    neighbourhood_ranking: List[Dict[str, Any]]
    window_summary: List[Dict[str, Any]]

class ModelInfo(BaseModel):
    """Information about loaded models"""
    crime_type: str
    display_name: str
    window_days: int
    neighbourhood_count: int
    threshold: float

# ============================================================================
# Helper Functions
# ============================================================================

def get_neighbourhood_id_from_name(neighbourhood_name: str, model_bundle) -> Optional[int]:
    """Extract neighbourhood ID from name using model's label encoder"""
    le = model_bundle['label_encoder']
    all_neighbourhoods = le.classes_
    
    for nh in all_neighbourhoods:
        clean_name = clean_neighbourhood_name(nh)
        if clean_name.lower() == neighbourhood_name.lower():
            return extract_neighbourhood_id(nh)
    return None

def format_prediction_result(row, model_bundle) -> Dict:
    """Format a single prediction row for API response"""
    neighbourhood = row['neighbourhood']
    nh_id = extract_neighbourhood_id(neighbourhood)
    clean_name = clean_neighbourhood_name(neighbourhood)
    
    return {
        "window_start": row['window_start'].strftime('%Y-%m-%d'),
        "window_end": row['window_end'].strftime('%Y-%m-%d'),
        "neighbourhood": neighbourhood,
        "neighbourhood_id": nh_id,
        "neighbourhood_clean": clean_name,
        "probability": round(float(row['probability']), 4),
        "predicted": bool(row['predicted']),
        "risk_level": row['risk_level']
    }

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global models_cache
    print("Loading crime prediction models...")
    try:
        models_cache = load_all_models()
        print(f"✅ Loaded {len(models_cache)} models")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        models_cache = {}

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Crime Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/api/docs",
            "models": "/api/models",
            "neighbourhoods": "/api/neighbourhoods/{crime_type}",
            "predict": "/api/predict",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models_cache) if models_cache else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List all available prediction models"""
    if not models_cache:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    models_info = []
    for crime_type, bundle in models_cache.items():
        le = bundle['label_encoder']
        models_info.append(ModelInfo(
            crime_type=crime_type,
            display_name=CRIME_DISPLAY_NAMES.get(crime_type, crime_type),
            window_days=bundle['window_days'],
            neighbourhood_count=len(le.classes_),
            threshold=round(bundle['threshold'], 3)
        ))
    
    return models_info

@app.get("/api/neighbourhoods/{crime_type}", response_model=List[NeighbourhoodInfo])
async def get_neighbourhoods(crime_type: CrimeType):
    """Get list of all neighbourhoods for a specific crime type"""
    if not models_cache:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if crime_type.value not in models_cache:
        raise HTTPException(status_code=404, detail=f"Model for {crime_type.value} not found")
    
    bundle = models_cache[crime_type.value]
    le = bundle['label_encoder']
    all_neighbourhoods = le.classes_
    
    neighbourhoods = []
    for nh in all_neighbourhoods:
        nh_id = extract_neighbourhood_id(nh)
        clean_name = clean_neighbourhood_name(nh)
        if nh_id:
            neighbourhoods.append(NeighbourhoodInfo(
                id=nh_id,
                name=clean_name,
                display_name=f"{clean_name} ({nh_id})"
            ))
    
    # Sort by name
    neighbourhoods.sort(key=lambda x: x.name)
    return neighbourhoods

@app.get("/api/neighbourhoods/{crime_type}/search")
async def search_neighbourhoods(
    crime_type: CrimeType,
    query: str = Query(..., min_length=1, description="Search term")
):
    """Search neighbourhoods by name or ID"""
    if not models_cache:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if crime_type.value not in models_cache:
        raise HTTPException(status_code=404, detail=f"Model for {crime_type.value} not found")
    
    bundle = models_cache[crime_type.value]
    le = bundle['label_encoder']
    all_neighbourhoods = le.classes_
    
    # Create mappings
    id_to_name, name_to_id, _ = create_neighbourhood_mappings(all_neighbourhoods)
    
    # Search
    result = find_neighbourhood_by_input(query, id_to_name, name_to_id, all_neighbourhoods)
    
    if result is None:
        return {"results": []}
    elif isinstance(result, list):
        # Multiple matches
        matches = []
        for nh in result[:20]:  # Limit to 20 results
            nh_id = extract_neighbourhood_id(nh)
            clean_name = clean_neighbourhood_name(nh)
            matches.append({
                "id": nh_id,
                "name": clean_name,
                "display_name": f"{clean_name} ({nh_id})" if nh_id else clean_name,
                "raw": nh
            })
        return {"results": matches}
    else:
        # Single match
        nh_id = extract_neighbourhood_id(result)
        clean_name = clean_neighbourhood_name(result)
        return {"results": [{
            "id": nh_id,
            "name": clean_name,
            "display_name": f"{clean_name} ({nh_id})" if nh_id else clean_name,
            "raw": result
        }]}

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make crime predictions for a date range
    """
    if not models_cache:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    crime_type = request.crime_type.value
    if crime_type not in models_cache:
        raise HTTPException(status_code=404, detail=f"Model for {crime_type} not found")
    
    model_bundle = models_cache[crime_type]
    
    # Determine neighbourhood filter
    neighbourhood_filter = None
    if request.neighbourhood_id:
        # Find neighbourhood by ID
        le = model_bundle['label_encoder']
        all_neighbourhoods = le.classes_
        for nh in all_neighbourhoods:
            nh_id = extract_neighbourhood_id(nh)
            if nh_id == request.neighbourhood_id:
                neighbourhood_filter = nh
                break
        if not neighbourhood_filter:
            raise HTTPException(status_code=404, detail=f"Neighbourhood ID {request.neighbourhood_id} not found")
    
    elif request.neighbourhood_name:
        # Find neighbourhood by name
        le = model_bundle['label_encoder']
        all_neighbourhoods = le.classes_
        for nh in all_neighbourhoods:
            clean_name = clean_neighbourhood_name(nh)
            if clean_name.lower() == request.neighbourhood_name.lower():
                neighbourhood_filter = nh
                break
        if not neighbourhood_filter:
            raise HTTPException(status_code=404, detail=f"Neighbourhood '{request.neighbourhood_name}' not found")
    
    # Make predictions
    try:
        results_df = predict_for_date_range(
            model_bundle,
            request.start_date,
            request.end_date,
            neighbourhood_filter
        )
        
        # Format results
        formatted_results = []
        high_risk_alerts = []
        
        for _, row in results_df.iterrows():
            formatted = format_prediction_result(row, model_bundle)
            formatted_results.append(formatted)
            if formatted['risk_level'] == 'HIGH':
                high_risk_alerts.append(formatted)
        
        # Calculate summary statistics
        total_predictions = len(formatted_results)
        positive_predictions = sum(1 for r in formatted_results if r['predicted'])
        
        risk_counts = {
            'HIGH': sum(1 for r in formatted_results if r['risk_level'] == 'HIGH'),
            'MEDIUM': sum(1 for r in formatted_results if r['risk_level'] == 'MEDIUM'),
            'LOW': sum(1 for r in formatted_results if r['risk_level'] == 'LOW')
        }
        
        # Neighbourhood ranking (if no filter)
        neighbourhood_ranking = []
        if not neighbourhood_filter:
            neighbourhood_agg = {}
            for r in formatted_results:
                key = r['neighbourhood_clean']
                if key not in neighbourhood_agg:
                    neighbourhood_agg[key] = {'prob_sum': 0, 'count': 0, 'predicted': 0}
                neighbourhood_agg[key]['prob_sum'] += r['probability']
                neighbourhood_agg[key]['count'] += 1
                if r['predicted']:
                    neighbourhood_agg[key]['predicted'] += 1
            
            for name, data in neighbourhood_agg.items():
                neighbourhood_ranking.append({
                    'neighbourhood': name,
                    'avg_probability': round(data['prob_sum'] / data['count'], 4),
                    'predicted_count': data['predicted'],
                    'total_windows': data['count']
                })
            neighbourhood_ranking.sort(key=lambda x: x['avg_probability'], reverse=True)
            neighbourhood_ranking = neighbourhood_ranking[:20]  # Top 20
        
        # Window summary
        window_summary = []
        window_data = {}
        for r in formatted_results:
            key = f"{r['window_start']} to {r['window_end']}"
            if key not in window_data:
                window_data[key] = {'probs': [], 'predicted': 0}
            window_data[key]['probs'].append(r['probability'])
            if r['predicted']:
                window_data[key]['predicted'] += 1
        
        for window, data in window_data.items():
            window_summary.append({
                'window': window,
                'avg_risk': round(sum(data['probs']) / len(data['probs']), 4),
                'max_risk': round(max(data['probs']), 4),
                'predicted_count': data['predicted']
            })
        window_summary.sort(key=lambda x: x['window'])
        
        return PredictionResponse(
            crime_type=crime_type,
            crime_display_name=CRIME_DISPLAY_NAMES.get(crime_type, crime_type),
            date_range={
                "start": request.start_date.isoformat(),
                "end": request.end_date.isoformat()
            },
            neighbourhood_filter=neighbourhood_filter,
            summary=PredictionSummary(
                total_windows=len(window_summary),
                total_predictions=total_predictions,
                positive_predictions=positive_predictions,
                positive_percentage=round(positive_predictions / total_predictions * 100, 2),
                high_risk_alerts=risk_counts['HIGH'],
                medium_risk_alerts=risk_counts['MEDIUM'],
                low_risk_alerts=risk_counts['LOW']
            ),
            results=formatted_results,
            high_risk_alerts=high_risk_alerts[:20],  # Limit to 20 alerts
            neighbourhood_ranking=neighbourhood_ranking,
            window_summary=window_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/statistics/{crime_type}")
async def get_statistics(
    crime_type: CrimeType,
    neighbourhood_id: Optional[int] = None,
    days: int = Query(30, ge=7, le=365, description="Number of days to look back")
):
    """Get statistical summary for a crime type"""
    if not models_cache:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    crime_type_value = crime_type.value
    if crime_type_value not in models_cache:
        raise HTTPException(status_code=404, detail=f"Model for {crime_type_value} not found")
    
    # Calculate end date (today) and start date (days ago)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    model_bundle = models_cache[crime_type_value]
    
    # Find neighbourhood if specified
    neighbourhood_filter = None
    if neighbourhood_id:
        le = model_bundle['label_encoder']
        all_neighbourhoods = le.classes_
        for nh in all_neighbourhoods:
            nh_id = extract_neighbourhood_id(nh)
            if nh_id == neighbourhood_id:
                neighbourhood_filter = nh
                break
    
    # Make predictions
    try:
        results_df = predict_for_date_range(
            model_bundle,
            start_date,
            end_date,
            neighbourhood_filter
        )
        
        formatted_results = [format_prediction_result(row, model_bundle) for _, row in results_df.iterrows()]
        
        # Calculate daily averages
        daily_stats = {}
        for r in formatted_results:
            date = r['window_start']
            if date not in daily_stats:
                daily_stats[date] = {'probs': [], 'high_risk': 0}
            daily_stats[date]['probs'].append(r['probability'])
            if r['risk_level'] == 'HIGH':
                daily_stats[date]['high_risk'] += 1
        
        daily_averages = [
            {
                'date': date,
                'avg_risk': round(sum(data['probs']) / len(data['probs']), 4),
                'high_risk_count': data['high_risk']
            }
            for date, data in daily_stats.items()
        ]
        daily_averages.sort(key=lambda x: x['date'])
        
        return {
            "crime_type": crime_type_value,
            "crime_display_name": CRIME_DISPLAY_NAMES.get(crime_type_value, crime_type_value),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "neighbourhood": neighbourhood_filter,
            "daily_statistics": daily_averages,
            "overall_avg_risk": round(sum(r['probability'] for r in formatted_results) / len(formatted_results), 4) if formatted_results else 0,
            "total_high_risk_windows": sum(1 for r in formatted_results if r['risk_level'] == 'HIGH')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics calculation failed: {str(e)}")

# ============================================================================
# Run with: uvicorn app:app --reload --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)