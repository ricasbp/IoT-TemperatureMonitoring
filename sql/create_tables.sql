
-- TODO - create different database "temperature_db" and USER to be owner of all !!!
-- TODO - then, map this inside the docker to recreate db on init, whenever database-files do not exist yet

CREATE TABLE public.historical_data (
	"date" timestamptz NOT NULL,
	temperature numeric NOT NULL,
	"label" varchar NOT NULL
);
CREATE TABLE public.online_data (
	"date" timestamptz NOT NULL,
	temperature numeric NOT NULL,
	"label" varchar NOT NULL
);
SELECT create_hypertable('online_data', 'date');
SELECT create_hypertable('historical_data', 'date');
