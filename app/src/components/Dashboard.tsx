"use client";

import { useEffect, useRef, useState } from "react";
import type { VoiceResult, AvatarResult } from "@/hooks/useEchoState";

interface DashboardProps {
  voiceResult: VoiceResult | null;
  avatarResult: AvatarResult | null;
  originalBlobUrl: string | null;
  onReset: () => void;
}

export default function Dashboard({ voiceResult, avatarResult, originalBlobUrl, onReset }: DashboardProps) {
  const [synthUrl, setSynthUrl] = useState<string | null>(null);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [gifUrl, setGifUrl] = useState<string | null>(null);
  const [synthError, setSynthError] = useState<string | null>(null);

  const synthAudioRef = useRef<HTMLAudioElement>(null);

  // Decode synthesized audio
  useEffect(() => {
    if (!voiceResult?.audioBase64) return;
    try {
      const bytes = Uint8Array.from(atob(voiceResult.audioBase64), (c) => c.charCodeAt(0));

      // Validate WAV header
      const header = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
      if (header !== "RIFF" || bytes.length < 44) {
        setSynthError("Audio file is malformed.");
        return;
      }

      // Check actual duration from WAV header
      const dataView = new DataView(bytes.buffer);
      const numSamples = dataView.getUint32(40, true) / 2;
      const sampleRate = dataView.getUint32(24, true);
      const durationSec = sampleRate > 0 ? numSamples / sampleRate : 0;

      if (durationSec < 0.05) {
        setSynthError(`Generated audio is too short (${durationSec.toFixed(3)}s).`);
        return;
      }

      const blob = new Blob([bytes], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      setSynthUrl(url);
    } catch (e) {
      setSynthError("Failed to decode audio: " + String(e));
    }

    return () => { if (synthUrl) URL.revokeObjectURL(synthUrl); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voiceResult?.audioBase64]);

  // Decode avatar image
  useEffect(() => {
    if (!avatarResult?.imageBase64) return;
    try {
      const bytes = Uint8Array.from(atob(avatarResult.imageBase64), (c) => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: "image/png" });
      setAvatarUrl(URL.createObjectURL(blob));
    } catch {}
    return () => { if (avatarUrl) URL.revokeObjectURL(avatarUrl); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [avatarResult?.imageBase64]);

  // Decode GIF
  useEffect(() => {
    if (!avatarResult?.gifBase64) return;
    try {
      const bytes = Uint8Array.from(atob(avatarResult.gifBase64), (c) => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: "image/gif" });
      setGifUrl(URL.createObjectURL(blob));
    } catch {}
  }, [avatarResult?.gifBase64]);

  const topStages = voiceResult?.stageLtencies
    ? Object.entries(voiceResult.stageLtencies).filter(([, v]) => v > 0).sort(([, a], [, b]) => b - a).slice(0, 5)
    : [];
  const maxMs = topStages[0]?.[1] ?? 1;

  return (
    <div className="animate-result-lift" style={{ width: "100%", display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "var(--space-2)" }}>
        <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-success)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "var(--space-2)" }}>
          ✓ Pipeline Complete
        </p>
        <h2 style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-2xl)", fontWeight: 600, fontStyle: "normal", color: "var(--color-ink)", margin: 0 }}>
          Your Digital Twin
        </h2>
      </div>

      {/* Demo banner */}
      {voiceResult?.isDemo && (
        <div style={{ padding: "var(--space-3) var(--space-5)", background: "oklch(78% 0.18 70 / 0.08)", border: "1px solid oklch(78% 0.18 70 / 0.3)", borderRadius: "var(--radius-md)", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "oklch(78% 0.18 70)", display: "flex", gap: "var(--space-3)" }}>
          <span>⚠</span>
          <span><strong>Demo mode</strong> — AI service was offline. No real synthesis was performed. Start the Python service (see README) to run the actual model.</span>
        </div>
      )}

      {/* Voice panels */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 280px), 1fr))", gap: "var(--space-4)" }}>

        {/* Original recording */}
        {originalBlobUrl && (
          <AudioPanel label="Original Recording" blobUrl={originalBlobUrl} isReal />
        )}

        {/* Synthesized output */}
        <div style={{ padding: "var(--space-5)", background: "var(--color-paper-2)", border: `1px solid ${voiceResult?.isDemo ? "var(--color-border)" : "oklch(68% 0.22 255 / 0.3)"}`, borderRadius: "var(--radius-lg)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-4)" }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
              Synthesized Voice
            </span>
            {!voiceResult?.isDemo ? (
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-success)", display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                <span style={{ width: 5, height: 5, borderRadius: "9999px", background: "var(--color-success)", display: "inline-block" }} />
                Real Model Output
              </span>
            ) : (
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "oklch(78% 0.18 70)" }}>Demo</span>
            )}
          </div>

          {synthError ? (
            <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-error)" }}>⚠ {synthError}</p>
          ) : synthUrl ? (
            <>
              <audio ref={synthAudioRef} src={synthUrl} controls style={{ width: "100%", height: 36 }} />
              {voiceResult && (
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", marginTop: "var(--space-3)" }}>
                  {voiceResult.durationSec.toFixed(2)}s · {voiceResult.sampleRate} Hz
                  {voiceResult.watermarkId && <> · WM: {voiceResult.watermarkId.slice(0, 8)}…</>}
                </p>
              )}
            </>
          ) : (
            <div style={{ height: 36, borderRadius: "var(--radius-sm)" }} className="shimmer" />
          )}
        </div>
      </div>

      {/* Avatar panels */}
      {(avatarResult || avatarResult === null) && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 200px), 1fr))", gap: "var(--space-4)" }}>
          {avatarUrl && (
            <div style={{ padding: "var(--space-4)", background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)", textAlign: "center" }}>
              <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "var(--space-3)" }}>
                Generated Avatar
              </p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={avatarUrl} alt="Generated avatar" style={{ width: "100%", maxWidth: 180, height: "auto", borderRadius: "var(--radius-md)", display: "block", margin: "0 auto", imageRendering: "pixelated" }} />
              <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", marginTop: "var(--space-3)" }}>
                {avatarResult?.width}×{avatarResult?.height}px · untrained model
              </p>
            </div>
          )}

          {gifUrl && (
            <div style={{ padding: "var(--space-4)", background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)", textAlign: "center" }}>
              <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "var(--space-3)" }}>
                Avatar Animation
              </p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={gifUrl} alt="Avatar animation" style={{ width: "100%", maxWidth: 180, height: "auto", borderRadius: "var(--radius-md)", display: "block", margin: "0 auto", imageRendering: "pixelated" }} />
            </div>
          )}
        </div>
      )}

      {/* Stage latency chart */}
      {topStages.length > 0 && (
        <div style={{ padding: "var(--space-5)", background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-md)" }}>
          <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "var(--space-4)" }}>Pipeline Latency</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            {topStages.map(([name, ms]) => (
              <div key={name} style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", width: 72, flexShrink: 0 }}>{name}</span>
                <div style={{ flex: 1, height: 4, background: "var(--color-paper-4)", borderRadius: 2, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${(ms / maxMs) * 100}%`, background: "var(--color-accent)", borderRadius: 2, transition: "width 0.6s var(--ease-out)" }} />
                </div>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-2)", width: 56, textAlign: "right", flexShrink: 0 }}>{ms.toFixed(1)} ms</span>
              </div>
            ))}
          </div>
          {voiceResult?.totalLatencyMs != null && (
            <div style={{ marginTop: "var(--space-4)", paddingTop: "var(--space-4)", borderTop: "var(--rule-hairline)", display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)" }}>
              <span style={{ color: "var(--color-ink-4)" }}>Total</span>
              <span style={{ color: "var(--color-ink-2)" }}>
                {voiceResult.totalLatencyMs.toFixed(1)} ms
                {voiceResult.realTimeFactor > 0 && (
                  <span style={{ color: "var(--color-ink-4)", marginLeft: "var(--space-3)" }}>
                    ({Math.round(1 / voiceResult.realTimeFactor)}× real-time)
                  </span>
                )}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Model info */}
      <div style={{ padding: "var(--space-4) var(--space-5)", background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-md)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "var(--space-4)" }}>
        <InfoRow label="Acoustic Model" value="AcousticModel (3-layer ConvBlock)" />
        <InfoRow label="Vocoder" value="Griffin-Lim (80 mel, 32 iters)" />
        <InfoRow label="Speaker Encoder" value="d-vector 192-dim (untrained)" />
        <InfoRow label="Avatar Model" value="Conv decoder 64px (untrained)" />
        <InfoRow label="Watermark" value="Spread-spectrum SS-WM (α=0.01)" />
        <InfoRow label="Voice Clone" value="Not trained — generic synthesis" warn />
      </div>

      <button
        onClick={onReset}
        style={{ alignSelf: "center", padding: "var(--space-3) var(--space-8)", fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 400, color: "var(--color-ink-3)", background: "transparent", border: "1px solid var(--color-border)", borderRadius: "var(--radius-full)", cursor: "pointer" }}
      >
        Start Over
      </button>
    </div>
  );
}

function AudioPanel({ label, blobUrl, isReal }: { label: string; blobUrl: string; isReal: boolean }) {
  return (
    <div style={{ padding: "var(--space-5)", background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-4)" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em" }}>{label}</span>
        {isReal && <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>your mic</span>}
      </div>
      <audio src={blobUrl} controls style={{ width: "100%", height: 36 }} />
    </div>
  );
}

function InfoRow({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div>
      <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 2 }}>{label}</p>
      <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: warn ? "oklch(78% 0.18 70)" : "var(--color-ink-3)", fontStyle: "normal" }}>{value}</p>
    </div>
  );
}
