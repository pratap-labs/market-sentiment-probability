# Market Sentiment Probability Prediction System

## Overview

This application is a machine learning-based market sentiment prediction system for analyzing NSE (National Stock Exchange of India) derivatives data. It fetches futures and options data, processes it to extract meaningful features, and uses logistic regression to predict bullish/bearish market sentiment probabilities. The system includes a Streamlit web interface for visualization and real-time probability gauges.

The application focuses on analyzing Open Interest (OI) changes, Put-Call Ratios (PCR), and Foreign Institutional Investment (FII) proxy data to generate market sentiment predictions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit web framework with Plotly for interactive visualizations

**Design Pattern**: Single-page application with cached data loading

The frontend uses Streamlit's caching mechanisms (`@st.cache_data` and `@st.cache_resource`) to optimize performance by preventing redundant data loading and model training operations. The UI provides:

- Wide layout with expandable sidebar for user controls
- Two-tab interface: Sentiment Prediction and OI Analysis
- Real-time probability gauge visualizations using Plotly
- Interactive charts for Open Interest, PCR, and FII trend analysis
- Expiry-based futures analysis with multi-panel visualization

**Rationale**: Streamlit was chosen for rapid development of data science applications without requiring extensive frontend expertise. Plotly provides sophisticated interactive visualizations that enhance user engagement with the prediction data.

### Backend Architecture

**Technology**: Python with SQLAlchemy ORM for database interactions

**Design Pattern**: Modular service-oriented architecture with separated concerns

The backend is organized into distinct utility modules:

1. **Data Processing Layer** (`utils/db_data_processor.py`, `utils/data_processor.py`)
   - Handles data extraction from PostgreSQL database
   - Performs feature engineering and data transformations
   - Calculates FII proxies from Open Interest changes
   - Computes rolling averages, volatility metrics, and momentum indicators

2. **Model Training Layer** (`utils/model_trainer.py`)
   - Implements logistic regression for binary sentiment classification
   - Handles feature preparation and scaling using StandardScaler
   - Manages train-test splitting and model evaluation
   - Provides classification metrics and model persistence

3. **Data Acquisition Layer** (`utils/nse_fetcher.py`)
   - Fetches live data from NSE India public APIs
   - Implements session management and cookie handling for API access
   - Calculates expiry dates for NIFTY futures (last Thursday logic)
   - Includes fallback to synthetic data generation when API is unavailable

**Rationale**: This modular approach allows independent testing and maintenance of each component. The separation between data processing, model training, and data fetching enables flexibility in swapping implementations (e.g., different ML models or data sources) without affecting other layers.

### Data Storage

**Technology**: PostgreSQL with SQLAlchemy ORM

**Schema Design**: Two primary tables for time-series derivatives data

1. **FuturesData Table**
   - Stores futures contract information
   - Tracks Open Interest (OI) and changes over time
   - Records volume and underlying index values
   - Indexed by date and symbol for efficient querying

2. **OptionsData Table**
   - Stores options chain data
   - Separate columns for Call and Put metrics (OI, volume, changes)
   - Includes calculated Put-Call Ratio (PCR)
   - Supports strike-level analysis

**Database Configuration**: Connection parameters loaded from environment variables for security and deployment flexibility

**Rationale**: PostgreSQL provides robust support for time-series data with excellent query performance for date-range operations. SQLAlchemy ORM abstracts database operations, making the codebase database-agnostic and easier to maintain. The separate tables for futures and options reflect the different data structures while allowing efficient joins when needed.

**Alternative Considered**: NoSQL databases like MongoDB were considered but rejected because the data has a well-defined schema and requires complex analytical queries that SQL databases handle more efficiently.

**Pros**:
- Strong consistency guarantees for financial data
- Excellent aggregation capabilities for time-series analysis
- Mature ecosystem with good Python support

**Cons**:
- Requires more setup compared to file-based storage
- May be over-engineered for small datasets

### Machine Learning Pipeline

**Algorithm**: Logistic Regression with StandardScaler normalization

**Features Engineered**:
- FII net position changes (delta calculations)
- Put-Call Ratio trends and changes
- 3-day rolling averages for smoothing
- Volatility metrics using standard deviation
- Momentum indicators

**Target Variable**: Binary classification (Bullish=1, Bearish=0) derived from future price movements

**Training Process**:
1. Load historical data from PostgreSQL
2. Engineer features with rolling windows and statistical calculations
3. Handle missing values through median imputation
4. Normalize features using StandardScaler
5. Train logistic regression model
6. Cache trained model to avoid retraining on each request

**Rationale**: Logistic regression provides interpretable probabilities (0-100%) that are ideal for displaying sentiment confidence. It's computationally efficient for real-time predictions and performs well with engineered financial features. The model's coefficients can be analyzed to understand which factors most influence sentiment.

**Alternatives Considered**: More complex models like Random Forests or Neural Networks were considered but would reduce interpretability and increase computational overhead without significant accuracy gains for this binary classification task.

**Pros**:
- Fast training and inference
- Probabilistic outputs naturally suited for gauges
- Interpretable feature importance

**Cons**:
- Limited ability to capture non-linear patterns
- Assumes feature independence

### Data Fetching Strategy

**Source**: NSE India public APIs for derivatives market data

**Implementation**:
- Session-based requests with cookie management
- Custom headers mimicking browser behavior to avoid blocking
- Rate limiting with sleep intervals between requests
- Automatic calculation of NIFTY expiry dates using last-Thursday logic
- Graceful fallback to synthetic data generation when API is unavailable

**Rationale**: NSE APIs provide official, accurate market data but require careful session management. The fallback mechanism ensures the application remains functional during API downtime or rate limiting, enabling development and testing without constant API dependency.

**Error Handling**: The system includes comprehensive error messages guiding users to run data fetch scripts and handles database connection failures gracefully.

## External Dependencies

### Third-Party APIs

**NSE India Public API**
- Purpose: Fetch live futures and options market data
- Endpoint: `https://www.nseindia.com/api/option-chain-indices`
- Authentication: Session-based with cookie management
- Rate Limits: Unofficial limits; implementation includes throttling
- Fallback: Synthetic data generation using numpy random functions

### Python Libraries

**Web Framework**
- `streamlit`: Interactive web application framework
- `plotly`: Interactive visualization library for charts and gauges

**Data Processing**
- `pandas`: Data manipulation and time-series analysis
- `numpy`: Numerical computations and array operations

**Database**
- `SQLAlchemy`: ORM for PostgreSQL interactions
- Database engine: PostgreSQL (connection via environment variables)

**Machine Learning**
- `scikit-learn`: Logistic regression, scaling, and model evaluation metrics

**HTTP & Utilities**
- `requests`: HTTP library for API calls
- `python-dateutil`: Date calculations for expiry logic

### Database

**PostgreSQL Database**
- Purpose: Persistent storage for historical market data
- Tables: `futures_data`, `options_data`
- Connection: Via SQLAlchemy engine using environment variables
- Expected Variables: Database credentials stored in environment

### Environment Configuration

The application expects environment variables for database connectivity:
- Database host, port, username, password, and database name
- Configured through SQLAlchemy's `create_engine()` function

### File System Dependencies

**Data Directory Structure**:
- `data/`: Optional directory for CSV-based data (legacy support in `data_processor.py`)
- Database-first approach with file fallback for development

**Scripts**:
- `scripts/fetch_nse_data.py`: Standalone script to populate database with NSE data
- Must be executed before running the main application