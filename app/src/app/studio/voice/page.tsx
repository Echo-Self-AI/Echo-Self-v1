"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useEchoState } from "@/hooks/useEchoState";

export default function VoiceStudioPage() {
  const echo = useEchoState();
  const { stage, error, analyser, recordingTimeSec, recordingBlobUrl, voiceResult } = echo;

  // Synthesis panel state
  const [synthText, setSynthText] = useState("Hello, I am your EchoSelf digital twin. This is a demonstration of the voice synthesis pipeline.");
  const [synthStatus, setSynthStatus] = useState<"idle" | "working" | "done" | "error">("idle");
  const [synthAudioUrl, setSynthAudioUrl] = useState<string | null>(null);
  const [synthError, setSynthError] = useState<string | null>(null);
  const [synthMeta, setSynthMeta] = useState<{ duration: number; bytes: number; demo: boolean } | null>(null);
  const synthRef = useRef<HTMLAudioElement>(null);
  const urlRef = useRef<string | null>(null);

  useEffect(() => () => { if (urlRef.current) URL.revokeObjectURL(urlRef.current); }, []);

  const synthesize = useCallback(async () => {
    setSynthStatus("working");
    setSynthError(null);
    if (urlRef.current) { URL.revokeObjectURL(urlRef.current); urlRef.current = null; setSynthAudioUrl(null); }
    try {
      await fetch("/api/echoself", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_type: "consent_register", user_id: "demo-user", modality: "voice", speaker_id: "demo-speaker", scope: "all" }),
      });
      const res = await fetch("/api/echoself", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_type: "voice_synthesis", user_id: "demo-user", text_input: synthText }),
      });
      const data = await res.json();
      const out = data.output ?? data;
      const ab = out.audio_base64 ?? "";
      if (!ab) throw new Error(data.error ?? "No audio returned");
      const wav = Uint8Array.from(atob(ab), (c) => c.charCodeAt(0));
      const dv = new DataView(wav.buffer);
      const ns = dv.getUint32(40, true) / 2;
      const sr = dv.getUint32(24, true);
      const url = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
      urlRef.current = url;
      setSynthAudioUrl(url);
      setSynthMeta({ duration: ns / sr, bytes: wav.length, demo: data.demo ?? false });
      setSynthStatus("done");
    } catch (err) {
      setSynthError(err instanceof Error ? err.message : String(err));
      setSynthStatus("error");
    }
  }, [synthText]);

  const isRecording = stage === "recording";
  const mm = String(Math.floor(recordingTimeSec / 60)).padStart(2, "0");
  const ss = String(recordingTimeSec % 60).padStart(2, "0");

  return (
    <main style={{ minHeight: "calc(100dvh - 52px)", padding: "var(--space-8) var(--space-6)", maxWidth: 900, margin: "0 auto" }}>
      <h1 style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-3xl)", fontWeight: 600, fontStyle: "normal", color: "var(--color-ink)", margin: 0, letterSpacing: "-0.02em", marginBottom: "var(--space-2)" }}>
        Voice Studio
      </h1>
      <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", color: "var(--color-ink-3)", marginBottom: "var(--space-8)", fontStyle: "normal" }}>
        Record your voice for enrollment and test voice synthesis.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-6)" }}>

        {/* Enrollment panel */}
        <Panel title="Voice Enrollment">
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
            <CanvasWaveform analyser={analyser} active={isRecording} />

            {isRecording && (
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)" }}>
                <span style={{ width: 7, height: 7, borderRadius: "9999px", background: "var(--color-error)", display: "inline-block", animation: "pulse-ring 1.4s ease infinite" }} />
                <span style={{ color: "var(--color-ink-2)", fontWeight: 500 }}>{mm}:{ss}</span>
                <span>Speak naturally for 5+ seconds</span>
              </div>
            )}

            {stage === "welcome" && (
              <button onClick={echo.startRecording} style={recBtn("start")}>
                <MicIcon /> Start Recording
              </button>
            )}
            {stage === "recording" && (
              <button onClick={echo.stopRecording} style={recBtn("stop")}>
                <StopIcon /> Stop Recording
              </button>
            )}
            {stage === "preview" && recordingBlobUrl && (
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
                <audio src={recordingBlobUrl} controls style={{ width: "100%", height: 36 }} />
                <div style={{ display: "flex", gap: "var(--space-3)" }}>
                  <button onClick={echo.discardRecording} style={recBtn("secondary")}>Re-record</button>
                  <button onClick={echo.submitRecording} style={recBtn("start")}>Enroll Voice →</button>
                </div>
              </div>
            )}
            {stage === "processing" && (
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
                <div style={{ width: 20, height: 20, border: "2px solid var(--color-paper-4)", borderTop: "2px solid var(--color-accent)", borderRadius: "9999px", animation: "spinner 0.9s linear infinite" }} />
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)" }}>Processing enrollment…</span>
              </div>
            )}
            {stage === "dashboard" && voiceResult && (
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-success)" }}>✓ Enrolled · {voiceResult.durationSec.toFixed(2)}s synthesized</span>
                {voiceResult.audioBase64 && (
                  <audio src={`data:audio/wav;base64,${voiceResult.audioBase64}`} controls style={{ width: "100%", height: 36 }} />
                )}
                <button onClick={echo.reset} style={recBtn("secondary")}>Re-enroll</button>
              </div>
            )}
            {stage === "error" && (
              <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-error)" }}>⚠ {error}</p>
            )}

            <div style={{ padding: "var(--space-3) var(--space-4)", background: "oklch(78% 0.18 70 / 0.07)", border: "1px solid oklch(78% 0.18 70 / 0.2)", borderRadius: "var(--radius-md)", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "oklch(78% 0.18 70)" }}>
              Voice clone status: speaker encoder uses random weights. The enrolled embedding is extracted but cannot yet produce a personalised voice. Synthesis uses a generic untrained acoustic model.
            </div>
          </div>
        </Panel>

        {/* Synthesis panel */}
        <Panel title="Voice Synthesis">
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
            <textarea
              value={synthText}
              onChange={(e) => setSynthText(e.target.value)}
              rows={4}
              style={{
                width: "100%", resize: "vertical", minHeight: 90,
                fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", color: "var(--color-ink-2)",
                background: "var(--color-paper-3)", border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)", padding: "var(--space-4)", outline: "none",
                lineHeight: 1.6, fontStyle: "normal",
              }}
              onFocus={(e) => { e.currentTarget.style.borderColor = "var(--color-accent-ring)"; }}
              onBlur={(e) => { e.currentTarget.style.borderColor = "var(--color-border)"; }}
            />

            <button
              onClick={synthesize}
              disabled={synthStatus === "working" || !synthText.trim()}
              style={{
                padding: "var(--space-3) var(--space-6)",
                fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 500,
                color: "var(--color-paper)",
                background: synthStatus === "working" || !synthText.trim() ? "var(--color-paper-3)" : "var(--color-accent)",
                border: "none", borderRadius: "var(--radius-full)", cursor: synthStatus === "working" ? "wait" : "pointer",
                transition: "opacity 0.15s ease",
              }}
            >
              {synthStatus === "working" ? "Synthesizing…" : "Synthesize"}
            </button>

            {synthError && <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-error)" }}>⚠ {synthError}</p>}

            {synthAudioUrl && synthMeta && (
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
                <audio ref={synthRef} src={synthAudioUrl} controls style={{ width: "100%", height: 36 }} />
                <div style={{ display: "flex", gap: "var(--space-3)", flexWrap: "wrap" }}>
                  <Badge label={`${synthMeta.duration.toFixed(2)}s`} />
                  <Badge label={`${synthMeta.bytes} bytes`} />
                  <Badge label={synthMeta.demo ? "Demo mode" : "Real output"} warn={synthMeta.demo} />
                </div>
                {synthMeta.demo && (
                  <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "oklch(78% 0.18 70)" }}>
                    Service offline — demo silent audio returned.
                  </p>
                )}
              </div>
            )}

            {/* Model info */}
            <div style={{ padding: "var(--space-4)", background: "var(--color-paper-3)", borderRadius: "var(--radius-md)", display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <InfoRow label="Acoustic model" value="AcousticModel 3-layer ConvBlock" />
              <InfoRow label="Checkpoint" value="checkpoints/acoustic_real/best_model.pt" />
              <InfoRow label="Vocoder" value="Griffin-Lim (n_mels=80, n_iters=8)" />
              <InfoRow label="Sample rate" value="16000 Hz mono" />
              <InfoRow label="Speech quality" value="Audible but not intelligible (untrained)" warn />
            </div>
          </div>
        </Panel>
      </div>
    </main>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)", overflow: "hidden" }}>
      <div style={{ padding: "var(--space-3) var(--space-5)", borderBottom: "var(--rule-hairline)" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em" }}>{title}</span>
      </div>
      <div style={{ padding: "var(--space-5)" }}>{children}</div>
    </div>
  );
}

function InfoRow({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between" }}>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: warn ? "oklch(78% 0.18 70)" : "var(--color-ink-3)" }}>{value}</span>
    </div>
  );
}

function Badge({ label, warn }: { label: string; warn?: boolean }) {
  return <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", padding: "2px var(--space-3)", background: "var(--color-paper-3)", borderRadius: "var(--radius-sm)", color: warn ? "oklch(78% 0.18 70)" : "var(--color-ink-3)" }}>{label}</span>;
}

function recBtn(variant: "start" | "stop" | "secondary"): React.CSSProperties {
  const base: React.CSSProperties = { width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: "var(--space-3)", padding: "var(--space-3) var(--space-6)", fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 500, border: "none", borderRadius: "var(--radius-full)", cursor: "pointer", transition: "opacity 0.15s ease" };
  if (variant === "start") return { ...base, background: "var(--color-accent)", color: "var(--color-paper)", boxShadow: "0 0 20px var(--color-accent-glow)" };
  if (variant === "stop") return { ...base, background: "oklch(65% 0.22 25)", color: "var(--color-paper)", boxShadow: "0 0 20px oklch(65% 0.22 25 / 0.3)" };
  return { ...base, background: "transparent", border: "1px solid var(--color-border)", color: "var(--color-ink-3)", flex: 1 };
}

function MicIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>;
}
function StopIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>;
}

function CanvasWaveform({ analyser, active }: { analyser: AnalyserNode | null; active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    const buf = new Uint8Array(analyser?.fftSize ?? 256);
    function draw() {
      if (!canvas || !ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== rect.width * dpr) { canvas.width = rect.width * dpr; canvas.height = rect.height * dpr; ctx.scale(dpr, dpr); }
      const W = rect.width, H = rect.height;
      ctx.clearRect(0, 0, W, H);
      if (!analyser || !active) { ctx.lineWidth = 1; ctx.strokeStyle = "oklch(22% 0.012 245)"; ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke(); }
      else {
        analyser.getByteTimeDomainData(buf);
        ctx.lineWidth = 1.5; ctx.strokeStyle = "oklch(65% 0.22 25)"; ctx.shadowColor = "oklch(65% 0.22 25 / 0.3)"; ctx.shadowBlur = 8;
        ctx.beginPath();
        const slice = W / buf.length;
        let x = 0;
        for (let i = 0; i < buf.length; i++) { const v = buf[i]/128; const y = v*H/2; i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y); x+=slice; }
        ctx.lineTo(W, H/2); ctx.stroke();
      }
      rafRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [analyser, active]);
  return (
    <div style={{ width: "100%", height: 56, background: "var(--color-paper-3)", borderRadius: "var(--radius-md)", overflow: "hidden", border: `1px solid ${active ? "oklch(65% 0.22 25 / 0.4)" : "var(--color-border)"}` }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
