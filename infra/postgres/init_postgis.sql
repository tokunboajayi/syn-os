-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
-- Locations table for GPS tracking
CREATE TABLE IF NOT EXISTS locations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    altitude DOUBLE PRECISION,
    accuracy DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    heading DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(50) DEFAULT 'gps',
    -- gps, wifi, manual
    -- PostGIS geography point for efficient spatial queries
    geom GEOGRAPHY(Point, 4326) GENERATED ALWAYS AS (
        ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
    ) STORED
);
-- Index for spatial queries
CREATE INDEX IF NOT EXISTS idx_locations_geom ON locations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_locations_timestamp ON locations (timestamp DESC);
-- Floor Plans for Indoor GIS
CREATE TABLE IF NOT EXISTS floor_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    image_url TEXT NOT NULL,
    -- Path to image or URL
    -- Geographical bounds of the floor plan image
    top_left_lat DOUBLE PRECISION NOT NULL,
    top_left_lon DOUBLE PRECISION NOT NULL,
    bottom_right_lat DOUBLE PRECISION NOT NULL,
    bottom_right_lon DOUBLE PRECISION NOT NULL,
    level INT DEFAULT 0,
    -- Floor level (0=ground, 1=1st floor, -1=basement)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Polygon covering the floor plan area
    area GEOGRAPHY(Polygon, 4326) GENERATED ALWAYS AS (
        ST_MakePolygon(
            ST_MakeLine(
                ARRAY [
            ST_MakePoint(top_left_lon, top_left_lat),
            ST_MakePoint(bottom_right_lon, top_left_lat),
            ST_MakePoint(bottom_right_lon, bottom_right_lat),
            ST_MakePoint(top_left_lon, bottom_right_lat),
            ST_MakePoint(top_left_lon, top_left_lat) -- Close the loop
        ]
            )
        )
    ) STORED
);
CREATE INDEX IF NOT EXISTS idx_floor_plans_area ON floor_plans USING GIST (area);