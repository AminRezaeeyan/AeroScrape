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

-- Flights table (only outgoing, no flight_type, using is_international)
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
    is_international BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT fk_airline FOREIGN KEY (airline_id) REFERENCES airlines(id),
    CONSTRAINT fk_status FOREIGN KEY (status_id) REFERENCES statuses(id),
    CONSTRAINT fk_airport FOREIGN KEY (airport_id) REFERENCES airports(id),

    CONSTRAINT unique_flight UNIQUE (airline_id, flight_number, scheduled_time)
);

-- Indexes for frequent lookups

CREATE INDEX idx_flights_flight_number ON flights(flight_number);
CREATE INDEX idx_flights_scheduled_time ON flights(scheduled_time);
CREATE INDEX idx_flights_airport_id ON flights(airport_id);
CREATE INDEX idx_flights_airline_id ON flights(airline_id);
CREATE INDEX idx_flights_status_id ON flights(status_id);
CREATE INDEX idx_flights_is_international ON flights(is_international);
