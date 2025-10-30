from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import os
import pandas as pd
import random
import secrets

from database import init_db, get_db, migrate_db

# Read condition from environment variable
CONDITION = os.getenv("CONDITION", "with_ai")
SHOW_AI = (CONDITION == "with_ai")

print(f"Running in {CONDITION} mode (show_ai={SHOW_AI})")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up...")
    init_db()
    migrate_db()
    print(f"Database initialized in {CONDITION} mode")
    yield
    print("Shutting down...")

app = FastAPI(title="Stock Analyzer", lifespan=lifespan)

# Middleware
SECRET_KEY = os.getenv("SESSION_SECRET_KEY", secrets.token_urlsafe(32))
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load stocks data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stock_market_data", "stocks")

# Load main CSV once at startup
STOCKS_CSV_PATH = os.path.join(DATA_DIR, "study_trials_with_ai.csv")
stocks = pd.read_csv(STOCKS_CSV_PATH)

print(f"Loaded {len(stocks)} stocks from CSV")


def get_stock_data():
    """Get random stock data"""
    try:
        # Pick random stock
        index = random.randrange(0, len(stocks))
        stock = stocks.iloc[index]
        ticker = stock["Ticker"]

        # Load historical data
        stock_path = os.path.join(STOCK_DATA_DIR, f"{ticker}.csv")
        
        history = pd.read_csv(stock_path)
        hist = history.head(10)
        
        # Build response based on condition
        stock_data = {
            "ticker": ticker.upper(),
            "previous_open": stock["Previous_Open"],
            "current_price": stock["True_Next_Open"],
            "open": hist['Open'].tolist(),
            "high": hist['High'].tolist(),
            "low": hist['Low'].tolist(),
            "close": hist['Close'].tolist(),
            "volume": hist['Volume'].tolist(),
        }
        
        # Add AI data only if in with_ai mode
        if SHOW_AI:
            stock_data["ai_suggestion"] = stock["AI_Advice"]
            stock_data["ai_prediction"] = stock["Predicted_Next_Open"]
        else:
            stock_data["ai_suggestion"] = "N/A"
            stock_data["ai_prediction"] = 0
        
        return stock_data
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.get("/")
async def home(request: Request):
    """Homepage - creates session and redirects"""
    if "session_id" not in request.session:
        request.session["session_id"] = secrets.token_urlsafe(16)
        request.session["condition"] = CONDITION
        request.session["show_ai"] = SHOW_AI
        print(f"New session created: condition={CONDITION}")
    
    return RedirectResponse(url="/analyze/1", status_code=303)


@app.get("/analyze/{num}")
async def analyze(request: Request, num: int):
    """Show stock for analysis"""
    session_id = request.session.get("session_id")
    
    if not session_id:
        return RedirectResponse(url="/", status_code=303)
    
    # Ensure condition is set
    if "condition" not in request.session:
        request.session["condition"] = CONDITION
        request.session["show_ai"] = SHOW_AI
    
    show_ai = request.session.get("show_ai", SHOW_AI)
    
    # Check how many decisions made
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM stock_decisions WHERE session_id = %s",
                (session_id,)
            )
            count = cursor.fetchone()["count"]
            cursor.close()
    except Exception as e:
        print(f"Database error: {e}")
        count = 0
    
    # If already done 10, redirect to complete
    if count >= 10:
        return RedirectResponse(url="/complete", status_code=303)
    
    # Get stock data
    stock_data = get_stock_data()
    
    if not stock_data:
        return HTMLResponse("Error loading stock data", status_code=500)
    
    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "stock_number": num,
        "stock_data": stock_data,
        "show_ai": show_ai  # Pass to template
    })


@app.post("/decision")
async def save_decision(
    request: Request,
    stock_number: int = Form(...),
    ticker: str = Form(...),
    previous_open: str = Form(...),
    current_price: float = Form(...),
    ai_suggestion: str = Form("N/A"),
    ai_prediction: float = Form(0.0),
    user_decision: str = Form(...),
    user_confidence: int = Form(...)
):
    """Save user decision and redirect to next stock"""
    session_id = request.session.get("session_id")
    condition = request.session.get("condition", CONDITION)
    
    if not session_id:
        return RedirectResponse(url="/", status_code=303)
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Check if completion code already exists for this session
            cursor.execute("""
                SELECT completion_code FROM stock_decisions 
                WHERE session_id = %s AND completion_code IS NOT NULL 
                LIMIT 1
            """, (session_id,))
            
            existing = cursor.fetchone()
            
            if existing:
                completion_code = existing['completion_code']
            else:
                # Generate new 5-digit completion code
                completion_code = str(random.randrange(10000, 100000))
            
            # Insert decision
            cursor.execute("""
                INSERT INTO stock_decisions 
                (session_id, condition, ticker, previous_open, current_price, 
                 ai_suggestion, ai_prediction, user_decision, user_confidence, completion_code)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (session_id, condition, ticker, previous_open, current_price, 
                  ai_suggestion, ai_prediction, user_decision, user_confidence, completion_code))
            
            decision_id = cursor.fetchone()['id']
            cursor.close()
            
            print(f"Saved decision {decision_id}: {ticker} - {user_decision} (condition: {condition})")
            
        # Redirect to next stock
        next_num = stock_number + 1
        return RedirectResponse(url=f"/analyze/{next_num}", status_code=303)
        
    except Exception as e:
        print(f"Error saving decision: {e}")
        import traceback
        traceback.print_exc()
        return HTMLResponse(f"Error saving decision: {e}", status_code=500)


@app.get("/complete")
async def complete(request: Request):
    """Thank you page after 10 decisions"""
    session_id = request.session.get("session_id")
    
    if not session_id:
        return RedirectResponse(url="/", status_code=303)
    
    # Get completion code from database
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT completion_code FROM stock_decisions 
                WHERE session_id = %s AND completion_code IS NOT NULL 
                LIMIT 1
            """, (session_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                completion_code = result['completion_code']
            else:
                # Shouldn't happen, but fallback
                completion_code = str(random.randrange(10000, 100000))
    except Exception as e:
        print(f"Error getting completion code: {e}")
        completion_code = str(random.randrange(10000, 100000))
    
    return templates.TemplateResponse("complete.html", {
        "request": request,
        "completion_code": completion_code
    })


# Optional: Debug endpoint
@app.get("/debug/config")
async def debug_config():
    """Show current configuration"""
    return {
        "condition": CONDITION,
        "show_ai": SHOW_AI,
        "database_url_set": bool(os.getenv("DATABASE_URL")),
        "session_secret_set": bool(os.getenv("SESSION_SECRET_KEY")),
        "stocks_loaded": len(stocks)
    }