"use client";

import { useEffect, useRef, useState } from "react";

interface ResultCardProps {
  audioBase64: string | null;
  sampleRate: number;
  watermarkId: string | null;
  detectionResult: boolean | null;
  stageLtencies: Record<string, number> | null;
  totalLatencyMs: number | null;
  realTimeFactor: number | null;
  isDemoMode: boolean;
  transcript: string | null;
}

export default function ResultCard({
  audioBase64,
  sampleRate,
  watermarkId,
  detectionResult,
  stageLtencies,
  totalLatencyMs,
  realTimeFactor,
  isDemoMode,
  transcript,
}: ResultCardProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Build an object URL from base64 WAV so the browser can play it
  useEffect(() => {
    if (!audioBase64) return;
    const bytes = Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [audioBase64]);

  const handlePlayPause = () => {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      audioRef.current.play();
      setPlaying(true);
    }
  };

  const topStages = stageLtencies
    ? Object.entries(stageLtencies)
        .filter(([, v]) => v > 0)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
    : [];

  const maxLatency = topStages.length ? topStages[0][1] : 1;

  return (
    <div
      className="animate-result-lift"
      style={{
        width: "100%",
        display: "grid",
        gridTemplateColumns: "1fr",
        gap: "var(--space-4)",
      }}
    >
      {/* Demo mode banner */}
      {isDemoMode && (
        <div
          style={{
            padding: "var(--space-3) var(--space-5)",
            background: "oklch(78% 0.18 70 / 0.08)",
            border: "1px solid oklch(78% 0.18 70 / 0.3)",
            borderRadius: "var(--radius-md)",
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-xs)",
            color: "oklch(78% 0.18 70)",
            display: "flex",
            alignItems: "center",
            gap: "var(--space-3)",
          }}
        >
          <span>⚠</span>
          <span>
            <strong>Demo mode</strong> — AI service offline. No real voice synthesis was
            performed. Start the Python service to process actual audio.
          </span>
        </div>
      )}

      {/* Audio playback */}
      {audioUrl && (
        <div
          style={{
            padding: "var(--space-5) var(--space-6)",
            background: "var(--color-paper-2)",
            border: "var(--rule-hairline)",
            borderRadius: "var(--radius-lg)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "var(--space-4)",
            }}
          >
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "var(--text-xs)",
                color: "var(--color-ink-3)",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
              }}
            >
              Synthesized Voice Output
            </span>
            {!isDemoMode && (
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "var(--text-xs)",
                  color: "var(--color-success)",
                  display: "flex",
                  alignItems: "center",
                  gap: "var(--space-2)",
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "var(--radius-full)",
                    background: "var(--color-success)",
                    display: "inline-block",
                  }}
                />
                Real Model Output
              </span>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)" }}>
            <button
              onClick={handlePlayPause}
              aria-label={playing ? "Pause audio" : "Play audio"}
              style={{
                width: 44,
                height: 44,
                borderRadius: "var(--radius-full)",
                background: "var(--color-accent)",
                border: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
                color: "var(--color-paper)",
                transition: `transform var(--dur-fast) var(--ease-out), opacity var(--dur-fast) var(--ease-out)`,
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = "0.85"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = "1"; }}
              onMouseDown={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(0.95)"; }}
              onMouseUp={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(1)"; }}
            >
              {playing ? <PauseIcon /> : <PlayIcon />}
            </button>

            <div style={{ flex: 1 }}>
              {transcript && (
                <p
                  style={{
                    fontFamily: "var(--font-display)",
                    fontSize: "var(--text-sm)",
                    color: "var(--color-ink-2)",
                    marginBottom: "var(--space-2)",
                    fontStyle: "normal",
                  }}
                >
                  &ldquo;{transcript}&rdquo;
                </p>
              )}
              <audio
                ref={audioRef}
                src={audioUrl}
                onEnded={() => setPlaying(false)}
                style={{ width: "100%", height: 32, opacity: 0.7 }}
                controls
              />
            </div>
          </div>
        </div>
      )}

      {/* Watermark provenance */}
      {watermarkId && (
        <div
          style={{
            padding: "var(--space-4) var(--space-5)",
            background: "var(--color-paper-2)",
            border: "var(--rule-hairline)",
            borderRadius: "var(--radius-md)",
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "var(--space-3)",
          }}
        >
          <MetaRow
            label="Watermark ID"
            value={watermarkId.slice(0, 18) + "…"}
            mono
          />
          <MetaRow
            label="Watermark Detected"
            value={detectionResult ? "✓ Verified" : "— Not detected"}
            accent={detectionResult ?? false}
          />
        </div>
      )}

      {/* Stage latencies */}
      {topStages.length > 0 && (
        <div
          style={{
            padding: "var(--space-5) var(--space-6)",
            background: "var(--color-paper-2)",
            border: "var(--rule-hairline)",
            borderRadius: "var(--radius-md)",
          }}
        >
          <p
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "var(--text-xs)",
              color: "var(--color-ink-3)",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              marginBottom: "var(--space-4)",
            }}
          >
            Pipeline Latency
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            {topStages.map(([name, ms]) => (
              <div key={name} style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "var(--text-xs)",
                    color: "var(--color-ink-3)",
                    width: 72,
                    flexShrink: 0,
                  }}
                >
                  {name}
                </span>
                <div
                  style={{
                    flex: 1,
                    height: 4,
                    background: "var(--color-paper-4)",
                    borderRadius: 2,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${(ms / maxLatency) * 100}%`,
                      background: "var(--color-accent)",
                      borderRadius: 2,
                      transition: "width 0.6s var(--ease-out)",
                    }}
                  />
                </div>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "var(--text-xs)",
                    color: "var(--color-ink-2)",
                    width: 56,
                    textAlign: "right",
                    flexShrink: 0,
                  }}
                >
                  {ms.toFixed(1)} ms
                </span>
              </div>
            ))}
          </div>

          {totalLatencyMs !== null && (
            <div
              style={{
                marginTop: "var(--space-4)",
                paddingTop: "var(--space-4)",
                borderTop: "var(--rule-hairline)",
                display: "flex",
                justifyContent: "space-between",
                fontFamily: "var(--font-mono)",
                fontSize: "var(--text-xs)",
              }}
            >
              <span style={{ color: "var(--color-ink-3)" }}>Total</span>
              <span style={{ color: "var(--color-ink-2)", fontWeight: 500 }}>
                {totalLatencyMs.toFixed(1)} ms
                {realTimeFactor !== null && (
                  <span style={{ color: "var(--color-ink-4)", marginLeft: "var(--space-3)" }}>
                    ({(1 / Math.max(realTimeFactor, 0.0001)).toFixed(0)}× real-time)
                  </span>
                )}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MetaRow({
  label,
  value,
  mono,
  accent,
}: {
  label: string;
  value: string;
  mono?: boolean;
  accent?: boolean;
}) {
  return (
    <div>
      <p
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "var(--text-xs)",
          color: "var(--color-ink-4)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: 2,
        }}
      >
        {label}
      </p>
      <p
        style={{
          fontFamily: mono ? "var(--font-mono)" : "var(--font-display)",
          fontSize: "var(--text-sm)",
          color: accent ? "var(--color-success)" : "var(--color-ink-2)",
          fontStyle: "normal",
        }}
      >
        {value}
      </p>
    </div>
  );
}

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5,3 19,12 5,21"/>
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="4" width="4" height="16"/>
      <rect x="14" y="4" width="4" height="16"/>
    </svg>
  );
}
