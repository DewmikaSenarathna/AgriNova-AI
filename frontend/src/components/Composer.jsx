import React, { useEffect, useRef, useState } from "react";
import { MicIcon, MicOffIcon, LeafClipIcon, SendIcon, CloseIcon } from "../icons.jsx";

const QUICK_PROMPTS = [
  "My tomato leaves are turning yellow, what's wrong?",
  "Should I irrigate today?",
  "What's today's market price for my crop?",
  "Any government subsidies for fertilizer?",
];

const SpeechRecognitionCtor =
  typeof window !== "undefined"
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;

export default function Composer({ onSubmit, busy, disabled }) {
  const [question, setQuestion] = useState("");
  const [image, setImage] = useState(null); // { name, dataUrl }
  const [listening, setListening] = useState(false);
  const [voiceError, setVoiceError] = useState("");
  const recognitionRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!SpeechRecognitionCtor) return undefined;
    const recognition = new SpeechRecognitionCtor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join(" ");
      setQuestion(transcript);
    };
    recognition.onerror = (event) => {
      setVoiceError(
        event.error === "not-allowed"
          ? "Microphone access was denied."
          : "Voice input stopped unexpectedly."
      );
      setListening(false);
    };
    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    return () => recognition.stop();
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        160
      )}px`;
    }
  }, [question]);

  const toggleVoice = () => {
    if (!SpeechRecognitionCtor) {
      setVoiceError("Voice input isn't supported in this browser — try Chrome or Edge.");
      return;
    }
    setVoiceError("");
    if (listening) {
      recognitionRef.current.stop();
      setListening(false);
    } else {
      try {
        recognitionRef.current.start();
        setListening(true);
      } catch {
        // start() throws if a recognition session is already in flight —
        // safe to ignore, onend/onerror will settle `listening` state.
      }
    }
  };

  const handleFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setVoiceError("Please attach an image file.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImage({ name: file.name, dataUrl: reader.result });
    reader.readAsDataURL(file);
  };

  const submit = () => {
    const trimmed = question.trim();
    if (!trimmed || busy || disabled) return;
    onSubmit(trimmed, image?.dataUrl || null);
    setQuestion("");
    setImage(null);
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer">
      {image && (
        <div className="composer-attachment">
          <img src={image.dataUrl} alt="Attached leaf" />
          <span>{image.name}</span>
          <button type="button" onClick={() => setImage(null)} aria-label="Remove attached photo">
            <CloseIcon width={13} height={13} />
          </button>
        </div>
      )}

      <div className="composer-row">
        <textarea
          ref={textareaRef}
          rows={1}
          placeholder="Ask about your crop, disease, irrigation, fertilizer, prices, schemes…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={disabled}
          aria-label="Your question for AgriNova AI"
        />
        <div className="composer-actions">
          <button
            type="button"
            className="composer-icon-btn"
            title="Attach a leaf photo"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
          >
            <LeafClipIcon />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          <button
            type="button"
            className={`composer-icon-btn${listening ? " is-active" : ""}`}
            title={listening ? "Stop voice input" : "Ask by voice"}
            onClick={toggleVoice}
            disabled={disabled}
          >
            {listening ? <MicOffIcon /> : <MicIcon />}
          </button>
          <button
            type="button"
            className="composer-send"
            onClick={submit}
            disabled={disabled || busy || !question.trim()}
            title="Send"
          >
            <SendIcon width={15} height={15} />
          </button>
        </div>
      </div>

      {listening && (
        <span className="recording-indicator">
          <span className="recording-dot" /> Listening — speak your question
        </span>
      )}
      {voiceError && <span className="composer-hint">{voiceError}</span>}
      {!listening && !voiceError && (
        <span className="composer-hint">Enter to send · Shift+Enter for a new line</span>
      )}

      <div className="composer-quick-prompts">
        {QUICK_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="quick-prompt"
            onClick={() => setQuestion(prompt)}
            disabled={disabled}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
