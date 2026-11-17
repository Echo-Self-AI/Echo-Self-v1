"use client";

import WaveformVisualizer from "./WaveformVisualizer";

interface StageRecordingProps {
  analyser: AnalyserNode | null;
  timeSec: number;
  isRecording: boolean;
  onStart: () => void;
  onStop: () => void;
}

export default function StageRecording({
  analyser, timeSec, isRecording, onStart, onStop,
}: StageRecordingProps) {
  const mm = String(Math.floor(timeSec / 60)).padStart(2, "0");
  const ss = String(timeSec % 60).padStart(2, "0");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)", alignItems: "center", width: "100%" }}>
      <WaveformVisualizer analyser={analyser} isActive={isRecording} mode={isRecording ? "recording" : "idle"} height={72} />

      <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)" }}>
        {isRecording && (
          <>
            <span style={{ width: 7, height: 7, borderRadius: "9999px", background: "var(--color-error)", display: "inline-block", animation: "pulse-ring 1.4s ease-out infinite" }} />
            <span style={{ color: "var(--color-ink-2)", fontWeight: 500 }}>{mm}:{ss}</span>
            <span>Recording — speak naturally</span>
          </>
        )}
        {!isRecording && <span>Press below to begin voice enrollment</span>}
      </div>

      <button
        onClick={isRecording ? onStop : onStart}
        style={{
          display: "flex", alignItems: "center", gap: "var(--space-3)",
          padding: "var(--space-4) var(--space-10)",
          fontFamily: "var(--font-display)", fontSize: "var(--text-base)", fontWeight: 500,
          color: "var(--color-paper)",
          background: isRecording ? "oklch(65% 0.22 25)" : "var(--color-accent)",
          border: "none", borderRadius: "var(--radius-full)", cursor: "pointer",
          transition: "opacity 0.15s ease, transform 0.1s ease",
          boxShadow: isRecording
            ? "0 0 24px oklch(65% 0.22 25 / 0.4)"
            : "0 0 24px var(--color-accent-glow)",
          position: "relative",
        }}
        onMouseDown={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(0.97)"; }}
        onMouseUp={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(1)"; }}
      >
        {isRecording ? (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
            Stop Recording
          </>
        ) : (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" x2="12" y1="19" y2="22"/>
            </svg>
            Start Recording
          </>
        )}
      </button>

      <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", textAlign: "center", maxWidth: 360 }}>
        Speak for at least 3 seconds. Your audio never leaves this machine.
      </p>
    </div>
  );
}
