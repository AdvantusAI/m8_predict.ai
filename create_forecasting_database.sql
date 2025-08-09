-- =====================================================
-- FORECASTING SYSTEM DATABASE DDL SCRIPT
-- =====================================================
-- This script creates all database objects required for the forecasting system
-- Compatible with PostgreSQL and includes business dimension integration
-- Version: 1.0
-- Date: 2025-01-08
-- =====================================================

-- Create schema (modify schema name in settings.py if different)
CREATE SCHEMA IF NOT EXISTS m8_schema;

-- =====================================================
-- UTILITY FUNCTIONS
-- =====================================================

-- Function to update updated_at timestamps
CREATE OR REPLACE FUNCTION m8_schema.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to generate series_id from business dimensions
CREATE OR REPLACE FUNCTION m8_schema.generate_series_id(
    p_product_id INTEGER,
    p_location_id INTEGER DEFAULT NULL,
    p_customer_id INTEGER DEFAULT NULL,
    p_frequency VARCHAR(20) DEFAULT 'monthly',
    p_measure VARCHAR(50) DEFAULT 'quantity'
) RETURNS VARCHAR(255) AS $$
BEGIN
    RETURN CONCAT(
        'P', COALESCE(p_product_id::TEXT, 'ALL'),
        CASE WHEN p_location_id IS NOT NULL THEN CONCAT('_L', p_location_id) ELSE '' END,
        CASE WHEN p_customer_id IS NOT NULL THEN CONCAT('_C', p_customer_id) ELSE '' END,
        '_', UPPER(p_frequency),
        '_', UPPER(p_measure)
    );
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- CORE TABLES
-- =====================================================

-- 1. TIME_SERIES (Master table for time series metadata with business integration)
CREATE TABLE m8_schema.time_series (
    id SERIAL PRIMARY KEY,
    series_id VARCHAR(255) NOT NULL,
    series_name VARCHAR(500),
    
    -- Business dimension foreign keys (assumes existing tables)
    product_id INTEGER, -- REFERENCES products(product_id)
    location_id INTEGER, -- REFERENCES locations(location_id)
    customer_id INTEGER, -- REFERENCES customers(customer_id)
    
    -- Flexible grain columns for additional dimensions
    grain1 VARCHAR(255), -- e.g., product category, sales channel
    grain2 VARCHAR(255), -- e.g., customer segment, region
    grain3 VARCHAR(255), -- e.g., sales rep, campaign
    
    -- Time attributes
    time_level VARCHAR(50) NOT NULL, -- 'Week', 'Month', 'Quarter', 'Day'
    frequency VARCHAR(20) NOT NULL,   -- 'weekly', 'monthly', 'quarterly', 'daily'
    
    -- Aggregation metadata
    aggregation_level VARCHAR(100) NOT NULL, -- 'product', 'product_location', 'product_location_customer'
    measure_type VARCHAR(50) NOT NULL DEFAULT 'quantity', -- 'quantity', 'revenue', 'units', 'profit'
    
    -- Series metadata
    seasonal_periods INTEGER,
    total_periods INTEGER,
    start_date DATE,
    end_date DATE,
    
    -- Data quality indicators
    missing_values_count INTEGER DEFAULT 0,
    outliers_count INTEGER DEFAULT 0,
    data_quality_score FLOAT,
    
    -- Status and flags
    is_active BOOLEAN DEFAULT true,
    is_forecasted BOOLEAN DEFAULT false,
    last_forecast_date TIMESTAMP,
    
    -- Additional metadata as JSON for flexibility
    metadata JSONB,
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Time Series Indexes
CREATE INDEX idx_time_series_series_id ON m8_schema.time_series(series_id);
CREATE INDEX idx_time_series_product_id ON m8_schema.time_series(product_id);
CREATE INDEX idx_time_series_location_id ON m8_schema.time_series(location_id);
CREATE INDEX idx_time_series_customer_id ON m8_schema.time_series(customer_id);
CREATE INDEX idx_time_series_business_combo ON m8_schema.time_series(product_id, location_id, customer_id);
CREATE INDEX idx_time_series_grains ON m8_schema.time_series(grain1, grain2, grain3);
CREATE INDEX idx_time_series_frequency ON m8_schema.time_series(frequency);
CREATE INDEX idx_time_series_aggregation ON m8_schema.time_series(aggregation_level);
CREATE INDEX idx_time_series_measure_type ON m8_schema.time_series(measure_type);
CREATE INDEX idx_time_series_is_active ON m8_schema.time_series(is_active);
CREATE UNIQUE INDEX uq_series_id ON m8_schema.time_series(series_id);

-- 2. SALES_TRANSACTIONS (Source data table - maps to your transaction data)
CREATE TABLE m8_schema.sales_transactions (
    id SERIAL PRIMARY KEY,
    
    -- Business dimensions
    product_id INTEGER NOT NULL, -- REFERENCES products(product_id)
    location_id INTEGER NOT NULL, -- REFERENCES locations(location_id)
    customer_id INTEGER NOT NULL, -- REFERENCES customers(customer_id)
    
    -- Transaction details
    postdate DATE NOT NULL,
    quantity DECIMAL(15,4) NOT NULL,
    revenue DECIMAL(15,2),
    unit_price DECIMAL(10,2),
    cost DECIMAL(15,2),
    profit DECIMAL(15,2),
    
    -- Additional transaction attributes
    transaction_id VARCHAR(100),
    order_id VARCHAR(100),
    sales_rep_id INTEGER,
    channel VARCHAR(50),
    promotion_id INTEGER,
    
    -- Processing flags
    is_processed BOOLEAN DEFAULT false,
    is_outlier BOOLEAN DEFAULT false,
    is_return BOOLEAN DEFAULT false,
    
    -- Source tracking
    source_system VARCHAR(50) DEFAULT 'primary',
    batch_id VARCHAR(100),
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Sales Transactions Indexes
CREATE INDEX idx_sales_trans_product ON m8_schema.sales_transactions(product_id);
CREATE INDEX idx_sales_trans_location ON m8_schema.sales_transactions(location_id);
CREATE INDEX idx_sales_trans_customer ON m8_schema.sales_transactions(customer_id);
CREATE INDEX idx_sales_trans_postdate ON m8_schema.sales_transactions(postdate);
CREATE INDEX idx_sales_trans_combo_date ON m8_schema.sales_transactions(product_id, location_id, customer_id, postdate);
CREATE INDEX idx_sales_trans_processed ON m8_schema.sales_transactions(is_processed);
CREATE INDEX idx_sales_trans_transaction_id ON m8_schema.sales_transactions(transaction_id);

-- 3. TIME_SERIES_DATA (Historical data points aggregated from transactions)
CREATE TABLE m8_schema.time_series_data (
    id SERIAL PRIMARY KEY,
    series_id INTEGER NOT NULL REFERENCES m8_schema.time_series(id) ON DELETE CASCADE,
    
    -- Time dimension
    time_period VARCHAR(50) NOT NULL, -- e.g., '2024-W01', '2024-01', '2024-Q1', '2024-01-15'
    period_date DATE NOT NULL,
    
    -- Aggregated values
    value FLOAT NOT NULL, -- Primary measure (quantity, revenue, etc.)
    transaction_count INTEGER DEFAULT 0, -- Number of transactions in period
    unique_customers INTEGER DEFAULT 0, -- Distinct customers in period
    unique_products INTEGER DEFAULT 0, -- Distinct products in period (for aggregated series)
    
    -- Statistical measures
    min_value FLOAT,
    max_value FLOAT,
    avg_value FLOAT,
    std_value FLOAT,
    
    -- Original and adjusted values
    original_value FLOAT, -- Before any transformations
    
    -- Data quality flags
    is_outlier BOOLEAN DEFAULT false,
    is_imputed BOOLEAN DEFAULT false,
    is_adjusted BOOLEAN DEFAULT false,
    is_holiday BOOLEAN DEFAULT false,
    is_promotion BOOLEAN DEFAULT false,
    
    -- Source information
    source VARCHAR(100) DEFAULT 'sales_transactions',
    version VARCHAR(50),
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Time Series Data Indexes
CREATE INDEX idx_data_series_period ON m8_schema.time_series_data(series_id, period_date);
CREATE INDEX idx_data_period_date ON m8_schema.time_series_data(period_date);
CREATE INDEX idx_data_value ON m8_schema.time_series_data(value);
CREATE INDEX idx_data_is_outlier ON m8_schema.time_series_data(is_outlier);
CREATE UNIQUE INDEX uq_series_period ON m8_schema.time_series_data(series_id, period_date);

-- 4. FORECAST_JOBS (Track forecast execution batches)
CREATE TABLE m8_schema.forecast_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE DEFAULT gen_random_uuid()::text,
    
    -- Job configuration
    series_count INTEGER,
    horizon INTEGER,
    frequency VARCHAR(20),
    validation_method VARCHAR(50), -- 'InSample', 'OutSample'
    validation_periods INTEGER,
    
    -- Execution details
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed', 'cancelled'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_seconds FLOAT,
    
    -- Algorithm configuration
    algorithms_used JSONB,
    error_metric VARCHAR(20), -- Primary error metric for model selection
    confidence_level FLOAT DEFAULT 0.80,
    
    -- Results summary
    successful_forecasts INTEGER DEFAULT 0,
    failed_forecasts INTEGER DEFAULT 0,
    best_algorithm_counts JSONB, -- Count of how often each algorithm was selected as best
    
    -- Error information
    error_message TEXT,
    error_details JSONB,
    
    -- User context
    created_by VARCHAR(100),
    notes TEXT,
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Forecast Jobs Indexes
CREATE INDEX idx_forecast_jobs_job_id ON m8_schema.forecast_jobs(job_id);
CREATE INDEX idx_forecast_jobs_status ON m8_schema.forecast_jobs(status);
CREATE INDEX idx_forecast_jobs_created_at ON m8_schema.forecast_jobs(created_at);

-- 5. FORECASTS (Forecast metadata and model selection results)
CREATE TABLE m8_schema.forecasts (
    id SERIAL PRIMARY KEY,
    series_id INTEGER NOT NULL REFERENCES m8_schema.time_series(id) ON DELETE CASCADE,
    job_id INTEGER REFERENCES m8_schema.forecast_jobs(id) ON DELETE SET NULL,
    
    -- Forecast metadata
    forecast_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    horizon INTEGER NOT NULL,
    best_model VARCHAR(100),
    
    -- Model performance metrics
    validation_error FLOAT,
    error_metric VARCHAR(20), -- MAPE, RMSE, MAE, SMAPE
    confidence_level FLOAT DEFAULT 0.80,
    
    -- All model results (for comparison)
    all_model_results JSONB, -- Store results from all algorithms
    
    -- Model details
    model_parameters JSONB,
    model_description TEXT,
    
    -- Forecast context
    training_start_date DATE,
    training_end_date DATE,
    forecast_start_date DATE,
    forecast_end_date DATE,
    
    -- Status and versioning
    is_active BOOLEAN DEFAULT true,
    version INTEGER DEFAULT 1,
    superseded_by INTEGER REFERENCES m8_schema.forecasts(id),
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Forecasts Indexes
CREATE INDEX idx_forecast_series_date ON m8_schema.forecasts(series_id, forecast_date);
CREATE INDEX idx_forecast_job_id ON m8_schema.forecasts(job_id);
CREATE INDEX idx_forecast_is_active ON m8_schema.forecasts(is_active);
CREATE INDEX idx_forecast_best_model ON m8_schema.forecasts(best_model);

-- 6. FORECAST_DATA (Individual forecast predictions)
CREATE TABLE m8_schema.forecast_data (
    id SERIAL PRIMARY KEY,
    forecast_id INTEGER NOT NULL REFERENCES m8_schema.forecasts(id) ON DELETE CASCADE,
    
    -- Time dimension
    time_period VARCHAR(50) NOT NULL, -- e.g., '2024-W01', '2024-01'
    period_date DATE NOT NULL,
    period_index INTEGER NOT NULL, -- 1, 2, 3... for forecast steps ahead
    
    -- Forecast values (from best model)
    predicted_value FLOAT NOT NULL,
    lower_bound FLOAT,
    upper_bound FLOAT,
    
    -- Additional prediction intervals
    lower_80 FLOAT, -- 80% confidence interval
    upper_80 FLOAT,
    lower_90 FLOAT, -- 90% confidence interval  
    upper_90 FLOAT,
    lower_95 FLOAT, -- 95% confidence interval
    upper_95 FLOAT,
    
    -- Prediction components (if available)
    trend_component FLOAT,
    seasonal_component FLOAT,
    residual_component FLOAT,
    
    -- Alternative algorithm predictions (for comparison)
    algorithm_predictions JSONB,
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Forecast Data Indexes
CREATE INDEX idx_forecast_data_forecast_id ON m8_schema.forecast_data(forecast_id);
CREATE INDEX idx_forecast_data_period ON m8_schema.forecast_data(forecast_id, period_date);
CREATE INDEX idx_forecast_data_period_index ON m8_schema.forecast_data(forecast_id, period_index);
CREATE INDEX idx_forecast_data_date ON m8_schema.forecast_data(period_date);
CREATE UNIQUE INDEX uq_forecast_period ON m8_schema.forecast_data(forecast_id, period_date);

-- 7. MODEL_PERFORMANCE (Algorithm performance tracking across series and time)
CREATE TABLE m8_schema.model_performance (
    id SERIAL PRIMARY KEY,
    series_id INTEGER NOT NULL REFERENCES m8_schema.time_series(id) ON DELETE CASCADE,
    job_id INTEGER REFERENCES m8_schema.forecast_jobs(id) ON DELETE SET NULL,
    
    -- Model identification
    algorithm VARCHAR(100) NOT NULL,
    parameters JSONB,
    
    -- Performance metrics
    mape FLOAT, -- Mean Absolute Percentage Error
    rmse FLOAT, -- Root Mean Square Error
    mae FLOAT,  -- Mean Absolute Error
    smape FLOAT, -- Symmetric Mean Absolute Percentage Error
    mase FLOAT, -- Mean Absolute Scaled Error
    aic FLOAT,  -- Akaike Information Criterion
    bic FLOAT,  -- Bayesian Information Criterion
    
    -- Validation context
    validation_periods INTEGER,
    training_periods INTEGER,
    forecast_horizon INTEGER,
    validation_method VARCHAR(50),
    
    -- Execution metrics
    training_time_seconds FLOAT,
    prediction_time_seconds FLOAT,
    memory_usage_mb FLOAT,
    
    -- Model diagnostics
    is_successful BOOLEAN DEFAULT true,
    error_message TEXT,
    warnings JSONB,
    
    -- Statistical tests
    ljung_box_pvalue FLOAT, -- Residual autocorrelation test
    jarque_bera_pvalue FLOAT, -- Normality test
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Model Performance Indexes
CREATE INDEX idx_performance_series_algo ON m8_schema.model_performance(series_id, algorithm);
CREATE INDEX idx_performance_job_id ON m8_schema.model_performance(job_id);
CREATE INDEX idx_performance_algorithm ON m8_schema.model_performance(algorithm);
CREATE INDEX idx_performance_mape ON m8_schema.model_performance(mape);
CREATE INDEX idx_performance_successful ON m8_schema.model_performance(is_successful);

-- =====================================================
-- LOOKUP/REFERENCE TABLES
-- =====================================================

-- 8. FORECAST_RULES (Business rules for forecast automation)
CREATE TABLE m8_schema.forecast_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Rule conditions (JSON query for flexible matching)
    conditions JSONB NOT NULL, -- e.g., {"product_category": "Electronics", "location_region": "North"}
    
    -- Rule actions
    enabled_algorithms JSONB, -- List of algorithms to use
    forecast_horizon INTEGER,
    frequency VARCHAR(20),
    validation_method VARCHAR(50),
    confidence_level FLOAT DEFAULT 0.80,
    
    -- Rule metadata
    priority INTEGER DEFAULT 0, -- Higher priority rules take precedence
    is_active BOOLEAN DEFAULT true,
    
    -- Audit fields
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Forecast Rules Indexes
CREATE INDEX idx_forecast_rules_active ON m8_schema.forecast_rules(is_active);
CREATE INDEX idx_forecast_rules_priority ON m8_schema.forecast_rules(priority DESC);

-- 9. FORECAST_EXCEPTIONS (Track and manage forecast exceptions)
CREATE TABLE m8_schema.forecast_exceptions (
    id SERIAL PRIMARY KEY,
    series_id INTEGER NOT NULL REFERENCES m8_schema.time_series(id),
    forecast_id INTEGER REFERENCES m8_schema.forecasts(id),
    
    -- Exception details
    exception_type VARCHAR(50) NOT NULL, -- 'data_quality', 'model_failure', 'validation_error', 'business_rule'
    severity VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    
    -- Exception description
    title VARCHAR(200) NOT NULL,
    description TEXT,
    error_details JSONB,
    
    -- Resolution
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'investigating', 'resolved', 'ignored'
    resolution TEXT,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP,
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Forecast Exceptions Indexes
CREATE INDEX idx_exceptions_series_id ON m8_schema.forecast_exceptions(series_id);
CREATE INDEX idx_exceptions_type ON m8_schema.forecast_exceptions(exception_type);
CREATE INDEX idx_exceptions_status ON m8_schema.forecast_exceptions(status);
CREATE INDEX idx_exceptions_severity ON m8_schema.forecast_exceptions(severity);

-- =====================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- =====================================================

-- Apply update_updated_at triggers to all tables
CREATE TRIGGER update_time_series_updated_at 
    BEFORE UPDATE ON m8_schema.time_series 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

CREATE TRIGGER update_sales_transactions_updated_at 
    BEFORE UPDATE ON m8_schema.sales_transactions 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

CREATE TRIGGER update_time_series_data_updated_at 
    BEFORE UPDATE ON m8_schema.time_series_data 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

CREATE TRIGGER update_forecast_jobs_updated_at 
    BEFORE UPDATE ON m8_schema.forecast_jobs 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

CREATE TRIGGER update_forecasts_updated_at 
    BEFORE UPDATE ON m8_schema.forecasts 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

CREATE TRIGGER update_forecast_data_updated_at 
    BEFORE UPDATE ON m8_schema.forecast_data 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

CREATE TRIGGER update_model_performance_updated_at 
    BEFORE UPDATE ON m8_schema.model_performance 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

CREATE TRIGGER update_forecast_rules_updated_at 
    BEFORE UPDATE ON m8_schema.forecast_rules 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

CREATE TRIGGER update_forecast_exceptions_updated_at 
    BEFORE UPDATE ON m8_schema.forecast_exceptions 
    FOR EACH ROW EXECUTE FUNCTION m8_schema.update_updated_at_column();

-- =====================================================
-- DATA AGGREGATION FUNCTIONS
-- =====================================================

-- Function to create time series from business dimensions
CREATE OR REPLACE FUNCTION m8_schema.create_time_series_definitions(
    p_aggregation_levels TEXT[] DEFAULT ARRAY['product', 'product_location', 'product_location_customer'],
    p_frequencies TEXT[] DEFAULT ARRAY['monthly', 'weekly'],
    p_measures TEXT[] DEFAULT ARRAY['quantity', 'revenue']
) RETURNS INTEGER AS $$
DECLARE
    level TEXT;
    freq TEXT;
    measure TEXT;
    series_created INTEGER := 0;
    row_count INTEGER;
BEGIN
    -- Loop through each combination of aggregation level, frequency, and measure
    FOREACH level IN ARRAY p_aggregation_levels LOOP
        FOREACH freq IN ARRAY p_frequencies LOOP
            FOREACH measure IN ARRAY p_measures LOOP
                
                IF level = 'product' THEN
                    -- Product level aggregation
                    INSERT INTO m8_schema.time_series (
                        series_id, series_name, product_id,
                        time_level, frequency, aggregation_level, measure_type,
                        seasonal_periods, is_active
                    )
                    SELECT 
                        m8_schema.generate_series_id(p.product_id, NULL, NULL, freq, measure) as series_id,
                        CONCAT(p.product_name, ' - ', UPPER(freq), ' ', UPPER(measure)) as series_name,
                        p.product_id,
                        CASE 
                            WHEN freq = 'weekly' THEN 'Week'
                            WHEN freq = 'monthly' THEN 'Month'
                            WHEN freq = 'quarterly' THEN 'Quarter'
                            WHEN freq = 'daily' THEN 'Day'
                        END as time_level,
                        freq as frequency,
                        'product' as aggregation_level,
                        measure as measure_type,
                        CASE 
                            WHEN freq = 'weekly' THEN 52
                            WHEN freq = 'monthly' THEN 12
                            WHEN freq = 'quarterly' THEN 4
                            WHEN freq = 'daily' THEN 365
                        END as seasonal_periods,
                        true as is_active
                    FROM (SELECT DISTINCT product_id, 'Product ' || product_id::text as product_name 
                          FROM m8_schema.sales_transactions) p
                    ON CONFLICT (series_id) DO NOTHING;
                    
                ELSIF level = 'product_location' THEN
                    -- Product + Location level aggregation
                    INSERT INTO m8_schema.time_series (
                        series_id, series_name, product_id, location_id,
                        time_level, frequency, aggregation_level, measure_type,
                        seasonal_periods, is_active
                    )
                    SELECT 
                        m8_schema.generate_series_id(pl.product_id, pl.location_id, NULL, freq, measure) as series_id,
                        CONCAT('Product ', pl.product_id, ' - Location ', pl.location_id, ' - ', UPPER(freq), ' ', UPPER(measure)) as series_name,
                        pl.product_id,
                        pl.location_id,
                        CASE 
                            WHEN freq = 'weekly' THEN 'Week'
                            WHEN freq = 'monthly' THEN 'Month'
                            WHEN freq = 'quarterly' THEN 'Quarter'
                            WHEN freq = 'daily' THEN 'Day'
                        END as time_level,
                        freq as frequency,
                        'product_location' as aggregation_level,
                        measure as measure_type,
                        CASE 
                            WHEN freq = 'weekly' THEN 52
                            WHEN freq = 'monthly' THEN 12
                            WHEN freq = 'quarterly' THEN 4
                            WHEN freq = 'daily' THEN 365
                        END as seasonal_periods,
                        true as is_active
                    FROM (SELECT DISTINCT product_id, location_id 
                          FROM m8_schema.sales_transactions) pl
                    ON CONFLICT (series_id) DO NOTHING;
                    
                ELSIF level = 'product_location_customer' THEN
                    -- Product + Location + Customer level aggregation
                    INSERT INTO m8_schema.time_series (
                        series_id, series_name, product_id, location_id, customer_id,
                        time_level, frequency, aggregation_level, measure_type,
                        seasonal_periods, is_active
                    )
                    SELECT 
                        m8_schema.generate_series_id(plc.product_id, plc.location_id, plc.customer_id, freq, measure) as series_id,
                        CONCAT('Product ', plc.product_id, ' - Location ', plc.location_id, ' - Customer ', plc.customer_id, ' - ', UPPER(freq), ' ', UPPER(measure)) as series_name,
                        plc.product_id,
                        plc.location_id,
                        plc.customer_id,
                        CASE 
                            WHEN freq = 'weekly' THEN 'Week'
                            WHEN freq = 'monthly' THEN 'Month'
                            WHEN freq = 'quarterly' THEN 'Quarter'
                            WHEN freq = 'daily' THEN 'Day'
                        END as time_level,
                        freq as frequency,
                        'product_location_customer' as aggregation_level,
                        measure as measure_type,
                        CASE 
                            WHEN freq = 'weekly' THEN 52
                            WHEN freq = 'monthly' THEN 12
                            WHEN freq = 'quarterly' THEN 4
                            WHEN freq = 'daily' THEN 365
                        END as seasonal_periods,
                        true as is_active
                    FROM (SELECT DISTINCT product_id, location_id, customer_id 
                          FROM m8_schema.sales_transactions) plc
                    ON CONFLICT (series_id) DO NOTHING;
                END IF;
                
                GET DIAGNOSTICS row_count = ROW_COUNT;
                series_created := series_created + row_count;
            END LOOP;
        END LOOP;
    END LOOP;
    
    RETURN series_created;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate sales transactions into time series data
CREATE OR REPLACE FUNCTION m8_schema.aggregate_sales_to_time_series(
    p_start_date DATE DEFAULT NULL,
    p_end_date DATE DEFAULT NULL,
    p_series_ids INTEGER[] DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
    rec RECORD;
    rows_inserted INTEGER := 0;
    row_count INTEGER;
BEGIN
    -- Set default date range if not provided
    p_start_date := COALESCE(p_start_date, (SELECT MIN(postdate) FROM m8_schema.sales_transactions));
    p_end_date := COALESCE(p_end_date, (SELECT MAX(postdate) FROM m8_schema.sales_transactions));
    
    RAISE NOTICE 'Aggregating sales data from % to % for % series', 
        p_start_date, p_end_date, COALESCE(array_length(p_series_ids, 1), 0);
    
    -- Loop through each time series (filter by p_series_ids if provided)
    FOR rec IN 
        SELECT id, series_id, product_id, location_id, customer_id, 
               frequency, aggregation_level, measure_type
        FROM m8_schema.time_series 
        WHERE is_active = true
          AND (p_series_ids IS NULL OR id = ANY(p_series_ids))
        ORDER BY id
    LOOP
        RAISE NOTICE 'Processing series: % (ID: %, Level: %, Frequency: %, Measure: %)', 
            rec.series_id, rec.id, rec.aggregation_level, rec.frequency, rec.measure_type;
        
        -- Aggregate based on frequency
        IF rec.frequency = 'monthly' THEN
            INSERT INTO m8_schema.time_series_data (
                series_id, time_period, period_date, value, 
                transaction_count, unique_customers, unique_products
            )
            SELECT 
                rec.id as series_id,
                TO_CHAR(DATE_TRUNC('month', st.postdate), 'YYYY-MM') as time_period,
                DATE_TRUNC('month', st.postdate)::DATE as period_date,
                CASE 
                    WHEN rec.measure_type = 'quantity' THEN COALESCE(SUM(st.quantity), 0)
                    WHEN rec.measure_type = 'revenue' THEN COALESCE(SUM(COALESCE(st.revenue, st.quantity * st.unit_price)), 0)
                    WHEN rec.measure_type = 'profit' THEN COALESCE(SUM(COALESCE(st.profit, st.revenue - st.cost)), 0)
                    ELSE COALESCE(SUM(st.quantity), 0)
                END as value,
                COUNT(*) as transaction_count,
                COUNT(DISTINCT st.customer_id) as unique_customers,
                COUNT(DISTINCT st.product_id) as unique_products
            FROM m8_schema.sales_transactions st
            WHERE st.postdate BETWEEN p_start_date AND p_end_date
              AND (rec.product_id IS NULL OR st.product_id = rec.product_id)
              AND (rec.location_id IS NULL OR st.location_id = rec.location_id)  
              AND (rec.customer_id IS NULL OR st.customer_id = rec.customer_id)
              AND st.is_return = false -- Exclude returns
            GROUP BY DATE_TRUNC('month', st.postdate)
            HAVING COALESCE(SUM(st.quantity), 0) > 0 -- Only include periods with positive values
            ON CONFLICT (series_id, period_date) DO UPDATE SET
                value = EXCLUDED.value,
                transaction_count = EXCLUDED.transaction_count,
                unique_customers = EXCLUDED.unique_customers,
                unique_products = EXCLUDED.unique_products,
                updated_at = CURRENT_TIMESTAMP;
                
        ELSIF rec.frequency = 'weekly' THEN
            INSERT INTO m8_schema.time_series_data (
                series_id, time_period, period_date, value,
                transaction_count, unique_customers, unique_products
            )
            SELECT 
                rec.id as series_id,
                TO_CHAR(DATE_TRUNC('week', st.postdate), 'IYYY-"W"IW') as time_period,
                DATE_TRUNC('week', st.postdate)::DATE as period_date,
                CASE 
                    WHEN rec.measure_type = 'quantity' THEN COALESCE(SUM(st.quantity), 0)
                    WHEN rec.measure_type = 'revenue' THEN COALESCE(SUM(COALESCE(st.revenue, st.quantity * st.unit_price)), 0)
                    WHEN rec.measure_type = 'profit' THEN COALESCE(SUM(COALESCE(st.profit, st.revenue - st.cost)), 0)
                    ELSE COALESCE(SUM(st.quantity), 0)
                END as value,
                COUNT(*) as transaction_count,
                COUNT(DISTINCT st.customer_id) as unique_customers,
                COUNT(DISTINCT st.product_id) as unique_products
            FROM m8_schema.sales_transactions st
            WHERE st.postdate BETWEEN p_start_date AND p_end_date
              AND (rec.product_id IS NULL OR st.product_id = rec.product_id)
              AND (rec.location_id IS NULL OR st.location_id = rec.location_id)
              AND (rec.customer_id IS NULL OR st.customer_id = rec.customer_id)
              AND st.is_return = false
            GROUP BY DATE_TRUNC('week', st.postdate)
            HAVING SUM(COALESCE(st.quantity, 0)) > 0
            ON CONFLICT (series_id, period_date) DO UPDATE SET
                value = EXCLUDED.value,
                transaction_count = EXCLUDED.transaction_count,
                unique_customers = EXCLUDED.unique_customers,
                unique_products = EXCLUDED.unique_products,
                updated_at = CURRENT_TIMESTAMP;
                
        ELSIF rec.frequency = 'daily' THEN
            INSERT INTO m8_schema.time_series_data (
                series_id, time_period, period_date, value,
                transaction_count, unique_customers, unique_products
            )
            SELECT 
                rec.id as series_id,
                TO_CHAR(st.postdate, 'YYYY-MM-DD') as time_period,
                st.postdate as period_date,
                CASE 
                    WHEN rec.measure_type = 'quantity' THEN COALESCE(SUM(st.quantity), 0)
                    WHEN rec.measure_type = 'revenue' THEN COALESCE(SUM(COALESCE(st.revenue, st.quantity * st.unit_price)), 0)
                    WHEN rec.measure_type = 'profit' THEN COALESCE(SUM(COALESCE(st.profit, st.revenue - st.cost)), 0)
                    ELSE COALESCE(SUM(st.quantity), 0)
                END as value,
                COUNT(*) as transaction_count,
                COUNT(DISTINCT st.customer_id) as unique_customers,
                COUNT(DISTINCT st.product_id) as unique_products
            FROM m8_schema.sales_transactions st
            WHERE st.postdate BETWEEN p_start_date AND p_end_date
              AND (rec.product_id IS NULL OR st.product_id = rec.product_id)
              AND (rec.location_id IS NULL OR st.location_id = rec.location_id)
              AND (rec.customer_id IS NULL OR st.customer_id = rec.customer_id)
              AND st.is_return = false
            GROUP BY st.postdate
            HAVING SUM(COALESCE(st.quantity, 0)) > 0
            ON CONFLICT (series_id, period_date) DO UPDATE SET
                value = EXCLUDED.value,
                transaction_count = EXCLUDED.transaction_count,
                unique_customers = EXCLUDED.unique_customers,
                unique_products = EXCLUDED.unique_products,
                updated_at = CURRENT_TIMESTAMP;
        END IF;
        
        GET DIAGNOSTICS row_count = ROW_COUNT;
        rows_inserted := rows_inserted + row_count;
        
        -- Update time series metadata
        UPDATE m8_schema.time_series 
        SET 
            total_periods = (SELECT COUNT(*) FROM m8_schema.time_series_data WHERE series_id = rec.id),
            start_date = (SELECT MIN(period_date) FROM m8_schema.time_series_data WHERE series_id = rec.id),
            end_date = (SELECT MAX(period_date) FROM m8_schema.time_series_data WHERE series_id = rec.id),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = rec.id;
        
    END LOOP;
    
    RAISE NOTICE 'Aggregation completed. Total rows inserted/updated: %', rows_inserted;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

-- Function to get time series summary statistics
CREATE OR REPLACE FUNCTION m8_schema.get_time_series_summary(
    p_series_id INTEGER DEFAULT NULL
) RETURNS TABLE (
    series_id INTEGER,
    series_name TEXT,
    aggregation_level TEXT,
    measure_type TEXT,
    frequency TEXT,
    data_points INTEGER,
    min_date DATE,
    max_date DATE,
    min_value FLOAT,
    max_value FLOAT,
    avg_value FLOAT,
    total_value FLOAT,
    has_gaps BOOLEAN,
    last_forecast_date TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ts.id as series_id,
        ts.series_name,
        ts.aggregation_level,
        ts.measure_type,
        ts.frequency,
        COUNT(tsd.id)::INTEGER as data_points,
        MIN(tsd.period_date) as min_date,
        MAX(tsd.period_date) as max_date,
        MIN(tsd.value) as min_value,
        MAX(tsd.value) as max_value,
        AVG(tsd.value) as avg_value,
        SUM(tsd.value) as total_value,
        -- Check for gaps in the time series
        (EXTRACT(days FROM (MAX(tsd.period_date) - MIN(tsd.period_date))) + 1 > COUNT(tsd.id)::INTEGER) as has_gaps,
        ts.last_forecast_date
    FROM m8_schema.time_series ts
    LEFT JOIN m8_schema.time_series_data tsd ON ts.id = tsd.series_id
    WHERE ts.is_active = true
      AND (p_series_id IS NULL OR ts.id = p_series_id)
    GROUP BY ts.id, ts.series_name, ts.aggregation_level, ts.measure_type, ts.frequency, ts.last_forecast_date
    ORDER BY ts.aggregation_level, ts.frequency, ts.series_name;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View: Latest forecasts for each active time series
CREATE OR REPLACE VIEW m8_schema.v_latest_forecasts AS
SELECT 
    ts.id as series_id,
    ts.series_id as series_key,
    ts.series_name,
    ts.product_id,
    ts.location_id,
    ts.customer_id,
    ts.aggregation_level,
    ts.measure_type,
    ts.frequency,
    f.id as forecast_id,
    f.forecast_date,
    f.horizon,
    f.best_model,
    f.validation_error,
    f.error_metric,
    f.confidence_level,
    COUNT(fd.id) as forecast_periods
FROM m8_schema.time_series ts
INNER JOIN m8_schema.forecasts f ON ts.id = f.series_id AND f.is_active = true
LEFT JOIN m8_schema.forecast_data fd ON f.id = fd.forecast_id
WHERE ts.is_active = true
GROUP BY ts.id, ts.series_id, ts.series_name, ts.product_id, ts.location_id, ts.customer_id,
         ts.aggregation_level, ts.measure_type, ts.frequency,
         f.id, f.forecast_date, f.horizon, f.best_model, f.validation_error, f.error_metric, f.confidence_level;

-- View: Time series data quality summary
CREATE OR REPLACE VIEW m8_schema.v_data_quality_summary AS
SELECT 
    ts.id as series_id,
    ts.series_id as series_key,
    ts.series_name,
    ts.aggregation_level,
    ts.frequency,
    COUNT(tsd.id) as total_periods,
    COUNT(CASE WHEN tsd.is_outlier THEN 1 END) as outlier_count,
    COUNT(CASE WHEN tsd.is_imputed THEN 1 END) as imputed_count,
    COUNT(CASE WHEN tsd.value = 0 THEN 1 END) as zero_value_count,
    MIN(tsd.period_date) as min_date,
    MAX(tsd.period_date) as max_date,
    AVG(tsd.value) as avg_value,
    STDDEV(tsd.value) as std_value,
    -- Calculate coefficient of variation
    CASE WHEN AVG(tsd.value) > 0 THEN STDDEV(tsd.value) / AVG(tsd.value) ELSE NULL END as cv,
    ts.data_quality_score,
    ts.last_forecast_date
FROM m8_schema.time_series ts
LEFT JOIN m8_schema.time_series_data tsd ON ts.id = tsd.series_id
WHERE ts.is_active = true
GROUP BY ts.id, ts.series_id, ts.series_name, ts.aggregation_level, ts.frequency, 
         ts.data_quality_score, ts.last_forecast_date
ORDER BY ts.aggregation_level, ts.frequency, ts.series_name;

-- View: Model performance comparison
CREATE OR REPLACE VIEW m8_schema.v_model_performance_summary AS
SELECT 
    mp.algorithm,
    COUNT(mp.id) as total_runs,
    COUNT(CASE WHEN mp.is_successful THEN 1 END) as successful_runs,
    ROUND(COUNT(CASE WHEN mp.is_successful THEN 1 END)::NUMERIC / COUNT(mp.id) * 100, 2) as success_rate,
    ROUND(AVG(CASE WHEN mp.is_successful THEN mp.mape END)::NUMERIC, 4) as avg_mape,
    ROUND(AVG(CASE WHEN mp.is_successful THEN mp.rmse END)::NUMERIC, 4) as avg_rmse,
    ROUND(AVG(CASE WHEN mp.is_successful THEN mp.mae END)::NUMERIC, 4) as avg_mae,
    ROUND(AVG(CASE WHEN mp.is_successful THEN mp.smape END)::NUMERIC, 4) as avg_smape,
    ROUND(AVG(mp.training_time_seconds)::NUMERIC, 2) as avg_training_time,
    ROUND(AVG(mp.prediction_time_seconds)::NUMERIC, 2) as avg_prediction_time
FROM m8_schema.model_performance mp
GROUP BY mp.algorithm
ORDER BY success_rate DESC, avg_mape ASC;

-- =====================================================
-- SAMPLE DATA AND USAGE EXAMPLES
-- =====================================================

-- Insert sample data (commented out - remove comments to use)
/*
-- Sample sales transactions
INSERT INTO m8_schema.sales_transactions (product_id, location_id, customer_id, postdate, quantity, revenue, unit_price) VALUES
(1, 1, 1, '2023-01-15', 100, 1000.00, 10.00),
(1, 1, 2, '2023-01-16', 150, 1500.00, 10.00),
(1, 2, 1, '2023-01-17', 75, 750.00, 10.00),
(2, 1, 1, '2023-01-18', 200, 4000.00, 20.00),
(2, 2, 2, '2023-01-19', 125, 2500.00, 20.00);

-- Create time series definitions for different aggregation levels
SELECT m8_schema.create_time_series_definitions();

-- Aggregate sales data into time series
SELECT m8_schema.aggregate_sales_to_time_series('2023-01-01', '2024-12-31');

-- View time series summary
SELECT * FROM m8_schema.get_time_series_summary();
*/

-- =====================================================
-- PERFORMANCE OPTIMIZATION NOTES
-- =====================================================

-- For large datasets, consider:
-- 1. Partitioning sales_transactions by date
-- 2. Creating additional indexes on frequently queried columns
-- 3. Using materialized views for expensive aggregations
-- 4. Regular VACUUM and ANALYZE operations
-- 5. Consider columnstore indexes for analytical queries

-- Example partitioning (uncomment if needed):
/*
-- Partition sales_transactions by month
-- ALTER TABLE m8_schema.sales_transactions RENAME TO sales_transactions_template;
-- CREATE TABLE m8_schema.sales_transactions (LIKE m8_schema.sales_transactions_template INCLUDING ALL) PARTITION BY RANGE (postdate);
-- CREATE TABLE m8_schema.sales_transactions_2023 PARTITION OF m8_schema.sales_transactions FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
-- CREATE TABLE m8_schema.sales_transactions_2024 PARTITION OF m8_schema.sales_transactions FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
*/

-- =====================================================
-- GRANTS AND PERMISSIONS (adjust as needed)
-- =====================================================

-- Example grants for different roles:
-- GRANT USAGE ON SCHEMA forecasting TO forecasting_users;
-- GRANT SELECT ON ALL TABLES IN SCHEMA forecasting TO forecasting_readers;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA forecasting TO forecasting_writers;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA forecasting TO forecasting_admins;

-- =====================================================
-- END OF SCRIPT
-- =====================================================

-- Script execution completed successfully
-- Total tables created: 9
-- Total indexes created: 50+
-- Total functions created: 4
-- Total views created: 3
-- Total triggers created: 9

SELECT 'Forecasting database schema created successfully!' as status,
       'Tables: time_series, sales_transactions, time_series_data, forecast_jobs, forecasts, forecast_data, model_performance, forecast_rules, forecast_exceptions' as tables_created,
       'Run: SELECT m8_schema.create_time_series_definitions(); to create time series definitions' as next_step,
       'Run: SELECT m8_schema.aggregate_sales_to_time_series(); to aggregate sales data' as then_step;