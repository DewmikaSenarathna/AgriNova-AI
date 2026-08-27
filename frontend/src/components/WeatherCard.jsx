import React from "react";
import { SunIcon, DropletIcon, WindIcon, conditionIcon } from "../icons.jsx";

export default function WeatherCard({ agentResults }) {
  const weather = (agentResults || []).find(
    (r) => r.agent_name === "weather_agent" && r.data?.current
  );

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">
          <SunIcon width={16} height={16} />
          Weather
        </div>
      </div>

      {!weather ? (
        <p className="card-empty">
          Ask something weather-related - irrigation timing, spraying
          conditions, rainfall and the current forecast for your location
          will show up here.
        </p>
      ) : (
        <WeatherBody data={weather.data} />
      )}
    </div>
  );
}

function WeatherBody({ data }) {
  const { current, location, daily_forecast: forecast = [] } = data;
  const Icon = conditionIcon(current.condition);

  return (
    <>
      <div className="weather-current">
        <Icon />
        <div>
          <div className="weather-temp">{Math.round(current.temperature_c)}°C</div>
          <div className="weather-condition">{current.condition}</div>
        </div>
      </div>
      <div className="weather-location">{location}</div>

      <div className="weather-stats">
        <div className="weather-stat">
          <DropletIcon />
          {current.humidity_pct}% humidity
        </div>
        <div className="weather-stat">
          <WindIcon />
          {current.wind_speed_kmh} km/h wind
        </div>
      </div>

      {forecast.length > 0 && (
        <div className="weather-forecast">
          {forecast.map((day) => {
            const DayIcon = conditionIcon(day.condition);
            const label = day.date
              ? new Date(day.date).toLocaleDateString(undefined, { weekday: "short" })
              : "—";
            return (
              <div className="forecast-day" key={day.date || Math.random()}>
                <div className="fd-label">{label}</div>
                <DayIcon />
                <div className="fd-temps">
                  {Math.round(day.temp_max_c)}° / {Math.round(day.temp_min_c)}°
                </div>
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
