import React from "react";

export default function Footer() {
  return (
    <footer className="app-footer">
      <div className="app-footer-brand">
        <img src="/Dons_logo.png" alt="Don logo" />
        <div>
          <div className="app-footer-title">AgriNova AI</div>
          <div className="app-footer-text">Don's product</div>
        </div>
      </div>
      <div className="app-footer-note">
        Intelligent farming assistance for better field decisions.
      </div>
    </footer>
  );
}
