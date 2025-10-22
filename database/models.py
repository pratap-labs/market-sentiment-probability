from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, BigInteger, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class FuturesData(Base):
    __tablename__ = 'futures_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    instrument_type = Column(String(50))
    symbol = Column(String(50))
    expiry_date = Column(Date)
    strike_price = Column(Float)
    option_type = Column(String(10))
    open_interest = Column(BigInteger)
    change_in_oi = Column(BigInteger)
    volume = Column(BigInteger)
    underlying_value = Column(Float)
    timestamp = Column(DateTime)
    
    def __repr__(self):
        return f"<FuturesData(date={self.date}, symbol={self.symbol}, oi={self.open_interest})>"

class OptionsData(Base):
    __tablename__ = 'options_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    symbol = Column(String(50))
    expiry_date = Column(Date)
    strike_price = Column(Float)
    option_type = Column(String(10))
    call_oi = Column(BigInteger)
    put_oi = Column(BigInteger)
    call_volume = Column(BigInteger)
    put_volume = Column(BigInteger)
    call_change_in_oi = Column(BigInteger)
    put_change_in_oi = Column(BigInteger)
    pcr = Column(Float)
    underlying_value = Column(Float)
    timestamp = Column(DateTime)
    
    def __repr__(self):
        return f"<OptionsData(date={self.date}, strike={self.strike_price}, pcr={self.pcr})>"

def get_engine():
    """Create database engine using environment variables"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return create_engine(database_url)

def get_session():
    """Create database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def create_tables():
    """Create all tables in the database"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database tables created successfully")

def drop_tables():
    """Drop all tables from the database"""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("Database tables dropped successfully")
