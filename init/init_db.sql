-- ===========================
-- AIRPORTS TABLE
-- ===========================

CREATE TABLE airports (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- ===========================
-- ARRIVAL FLIGHTS TABLE
-- ===========================

CREATE TABLE arrivals (
    id SERIAL PRIMARY KEY,
    scheduled_time TIMESTAMP NOT NULL,
    airline VARCHAR(255) NOT NULL,
    flight_number VARCHAR(50) NOT NULL,
    origin VARCHAR(255) NOT NULL,
    status VARCHAR(255) NOT NULL,
    counter VARCHAR(50),
    actual_time TIMESTAMP,
    register VARCHAR(100),
    aircraft VARCHAR(100),
    airport_id INT NOT NULL,
    
    FOREIGN KEY (airport_id) REFERENCES airports(id),
    CONSTRAINT unique_arrival UNIQUE (flight_number, scheduled_time, airport_id)
);

-- ===========================
-- DEPARTURE FLIGHTS TABLE
-- ===========================

CREATE TABLE departures (
    id SERIAL PRIMARY KEY,
    scheduled_time TIMESTAMP NOT NULL,
    airline VARCHAR(255) NOT NULL,
    flight_number VARCHAR(50) NOT NULL,
    destination VARCHAR(255) NOT NULL,
    status VARCHAR(255) NOT NULL,
    counter VARCHAR(50),
    actual_time TIMESTAMP,
    register VARCHAR(100),
    aircraft VARCHAR(100),
    airport_id INT NOT NULL,
    
    FOREIGN KEY (airport_id) REFERENCES airports(id),
    CONSTRAINT unique_departure UNIQUE (flight_number, scheduled_time, airport_id)
);

-- ===========================
-- INDEXES FOR PERFORMANCE
-- ===========================

-- ARRIVALS
CREATE INDEX idx_arrivals_flight_number ON arrivals(flight_number);
CREATE INDEX idx_arrivals_scheduled_time ON arrivals(scheduled_time);
CREATE INDEX idx_arrivals_airline ON arrivals(airline);
CREATE INDEX idx_arrivals_airport_id ON arrivals(airport_id);
CREATE INDEX idx_arrivals_status ON arrivals(status);

-- DEPARTURES
CREATE INDEX idx_departures_flight_number ON departures(flight_number);
CREATE INDEX idx_departures_scheduled_time ON departures(scheduled_time);
CREATE INDEX idx_departures_airline ON departures(airline);
CREATE INDEX idx_departures_airport_id ON departures(airport_id);
CREATE INDEX idx_departures_status ON departures(status);
