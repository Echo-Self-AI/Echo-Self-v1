"use client";

import { CSSProperties } from "react";

type RecordState = "idle" | "recording" | "processing" | "done" | "error";

interface RecordButtonProps {
  state: RecordState;
  onStart: () => void;
  onStop: () => void;
  disabled?: boolean;
}

export default function RecordButton({
  state,
  onStart,
  onStop,
  disabled,
}: RecordButtonProps) {
  const isRecording = state === "recording";
  const isProcessing = state === "processing";
  const isDone = state === "done";
  const isError = state === "error";
  const isDisabled = disabled || isProcessing;

  const handleClick = () => {
    if (isDisabled) return;
    if (isRecording) onStop();
    else if (state === "idle" || isDone || isError) onStart();
  };

  const label = {
    idle:       "Start Recording",
    recording:  "Stop Recording",
    processing: "Processing…",
    done:       "Record Again",
    error:      "Try Again",
  }[state];

  const icon = {
    idle:       <MicIcon />,
    recording:  <StopIcon />,
    processing: <SpinnerIcon />,
    done:       <CheckIcon />,
    error:      <ErrorIcon />,
  }[state];

  const accentStyle: CSSProperties = {
    background: isRecording
      ? "oklch(65% 0.22 25)"
      : isDone
      ? "oklch(72% 0.18 145)"
      : isError
      ? "oklch(65% 0.22 25)"
      : isProcessing
      ? "transparent"
      : "var(--color-accent)",
    borderColor: isRecording
      ? "oklch(65% 0.22 25)"
      : isDone
      ? "oklch(72% 0.18 145)"
      : isError
      ? "oklch(65% 0.22 25)"
      : "var(--color-accent)",
  };

  return (
    <div style={{ position: "relative", display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
      {/* Pulse ring — only when recording */}
      {isRecording && (
        <>
          <span style={pulseRing(1)} />
          <span style={pulseRing(2)} />
        </>
      )}

      <button
        onClick={handleClick}
        disabled={isDisabled}
        aria-label={label}
        aria-live="polite"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--space-3)",
          padding: "var(--space-4) var(--space-8)",
          fontFamily: "var(--font-display)",
          fontSize: "var(--text-base)",
          fontWeight: 500,
          letterSpacing: "0.01em",
          color: isProcessing ? "var(--color-ink-3)" : "var(--color-paper)",
          background: accentStyle.background,
          border: `1px solid ${accentStyle.borderColor}`,
          borderRadius: "var(--radius-full)",
          cursor: isDisabled ? "not-allowed" : "pointer",
          transition: `all var(--dur-base) var(--ease-out)`,
          position: "relative",
          zIndex: 1,
          minWidth: 200,
          justifyContent: "center",
          opacity: isProcessing ? 0.7 : 1,
          boxShadow: isRecording
            ? `0 0 24px oklch(65% 0.22 25 / 0.4)`
            : isDone
            ? `0 0 24px oklch(72% 0.18 145 / 0.3)`
            : !isProcessing
            ? `0 0 24px var(--color-accent-glow)`
            : "none",
        }}
      >
        <span
          style={{
            display: "flex",
            alignItems: "center",
            animation: isProcessing ? "spinner 0.9s linear infinite" : "none",
          }}
        >
          {icon}
        </span>
        <span>{label}</span>
      </button>
    </div>
  );
}

function pulseRing(index: number): CSSProperties {
  return {
    position: "absolute",
    inset: 0,
    borderRadius: "var(--radius-full)",
    border: "1.5px solid oklch(65% 0.22 25 / 0.5)",
    animation: `pulse-ring 2s ${index === 1 ? "0s" : "0.6s"} ease-out infinite`,
    pointerEvents: "none",
  };
}

function MicIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" x2="12" y1="19" y2="22"/>
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2"/>
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-3)" strokeWidth="2.5" strokeLinecap="round">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
}

function ErrorIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="15" x2="9" y1="9" y2="15"/>
      <line x1="9" x2="15" y1="9" y2="15"/>
    </svg>
  );
}
