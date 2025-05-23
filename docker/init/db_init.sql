-- Airports table
CREATE TABLE airports (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- Airlines table
CREATE TABLE airlines (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- Statuses table
CREATE TABLE statuses (
    id SERIAL PRIMARY KEY,
    status VARCHAR(255) UNIQUE NOT NULL
);

-- Flights table
CREATE TABLE flights (
    id SERIAL PRIMARY KEY,
    scheduled_time TIMESTAMP NOT NULL,
    airline_id INT NOT NULL,
    flight_number VARCHAR(50) NOT NULL,
    destination VARCHAR(255) NOT NULL,
    status_id INT NOT NULL,
    counter VARCHAR(50),
    actual_time TIMESTAMP,
    register VARCHAR(100),
    aircraft VARCHAR(100),
    airport_id INT NOT NULL,
    flight_type VARCHAR(50) NOT NULL,

    CONSTRAINT fk_airline FOREIGN KEY (airline_id) REFERENCES airlines(id),
    CONSTRAINT fk_status FOREIGN KEY (status_id) REFERENCES statuses(id),
    CONSTRAINT fk_airport FOREIGN KEY (airport_id) REFERENCES airports(id),

    CONSTRAINT unique_flight UNIQUE (airline_id, flight_number, scheduled_time)
);

-- Indexes for frequent lookups

-- Index on flight_number for quick searching flights by number
CREATE INDEX idx_flights_flight_number ON flights(flight_number);

-- Index on scheduled_time to speed up queries by date/time
CREATE INDEX idx_flights_scheduled_time ON flights(scheduled_time);

-- Index on airport_id for filtering flights by airport
CREATE INDEX idx_flights_airport_id ON flights(airport_id);

-- Index on airline_id for filtering by airline
CREATE INDEX idx_flights_airline_id ON flights(airline_id);

-- Index on status_id to speed up status filtering
CREATE INDEX idx_flights_status_id ON flights(status_id);