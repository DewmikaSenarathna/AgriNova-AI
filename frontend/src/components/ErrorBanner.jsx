import React from "react";

export default function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="banner-error" role="alert">
      <div>
        <strong>Couldn't get an answer</strong>
        {message}
      </div>
    </div>
  );
}
