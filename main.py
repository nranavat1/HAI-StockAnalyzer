from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse  # ← Add RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager  # ← Add this
import os
import pandas as pd
import random
import secrets
from starlette.middleware.sessions import SessionMiddleware  # ← Add this


from database import init_db, get_db

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    init_db()
    print("Database initialized")
    yield
    print("Shutting down...")

app = FastAPI(title="Stock Analyzer", lifespan=lifespan)

# Middleware
SECRET_KEY = os.getenv("SESSION_SECRET_KEY", secrets.token_urlsafe(32))
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load stocks
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stock_market_data", "stocks")

# Load main CSV once at startup
STOCKS_CSV_PATH = os.path.join(DATA_DIR, "study_trials_with_ai.csv")
stocks = pd.read_csv(STOCKS_CSV_PATH)


def get_stock_data():
    """Get random stock data"""
    try:
        index = random.randrange(0, len(stocks))
        stock = stocks.iloc[index]
        ticker = stock["Ticker"]

        stock_path = os.path.join(STOCK_DATA_DIR, f"{ticker}.csv")
        print("stock_path: ", stock_path)
        history = pd.read_csv(stock_path)
        hist = history.head(10)
        
        return {
            "ticker": ticker.upper(),
            "previous_open": stock["Previous_Open"],
            "current_price": stock["True_Next_Open"],
            "open": hist['Open'].tolist(),  # Convert to list for template
            "high": hist['High'].tolist(),
            "low": hist['Low'].tolist(),
            "close": hist['Close'].tolist(),
            "volume": hist['Volume'].tolist(),
            "ai_suggestion": stock["AI_Advice"],
            "ai_prediction": stock["Predicted_Next_Open"]
        }
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None


@app.get("/")
async def home(request: Request):
    """Homepage - creates session and redirects"""
    if "session_id" not in request.session:
        request.session["session_id"] = secrets.token_urlsafe(16)
    
    return RedirectResponse(url="/analyze/1", status_code=303)


@app.get("/analyze/{num}")
async def analyze(request: Request, num: int):
    """Show stock for analysis"""
    # Get session ID
    session_id = request.session.get("session_id")
    
    if not session_id:
        # No session - redirect to home to create one
        return RedirectResponse(url="/", status_code=303)
    
    # Check how many decisions made
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM stock_decisions WHERE session_id = %s",
            (session_id,)
        )
        count = cursor.fetchone()["count"]
        cursor.close()
    
    # If already done 10, redirect to complete
    if count >= 10:
        return RedirectResponse(url="/complete", status_code=303)
    
    # Get stock data
    stock_data = get_stock_data()
    
    if not stock_data:
        return HTMLResponse("Error loading stock data", status_code=500)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stock_number": num,
        "stock_data": stock_data,
    })


@app.post("/decision")
async def save_decision(
    request: Request,
    stock_number: int = Form(...),
    ticker: str = Form(...),
    previous_open: str=Form(...),
    current_price: float = Form(...),
    ai_suggestion: str = Form(...),
    ai_prediction: float = Form(...),
    user_decision: str = Form(...),
    user_confidence: int=Form(...),
):
    """Save user decision and redirect to next stock"""
    # Get session ID
    session_id = request.session.get("session_id")
    
    if not session_id:
        return RedirectResponse(url="/", status_code=303)
    
    # Save to database
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO stock_decisions 
                (session_id, ticker,previous_open,  current_price, ai_suggestion, ai_prediction, user_decision, user_confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (session_id, ticker, previous_open, current_price, ai_suggestion, ai_prediction, user_decision, user_confidence))
            
            decision_id = cursor.fetchone()['id']
            cursor.close()
            
        # Redirect to next stock
        next_num = stock_number + 1
        return RedirectResponse(url=f"/analyze/{next_num}", status_code=303)
        
    except Exception as e:
        print(f"Error saving decision: {e}")
        return HTMLResponse(f"Error saving decision: {e}", status_code=500)


@app.get("/complete")
async def complete(request: Request):
    """Thank you page after 10 decisions"""
    return templates.TemplateResponse("complete.html", {
        "request": request
    })