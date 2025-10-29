from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class StockDecision(Base):
    __tablename__ = "stock_decisions"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    ai_suggestion = Column(String)  # "BUY" or "HOLD"
    ai_prediction = Column(Float)
    user_decision = Column(String)  # "BUY" or "HOLD"
    timestamp = Column(DateTime, default=datetime.now())
    
    # Stock data fields
    current_price = Column(Float) #current price
    previous_price = Column(Float)