"use client";

import FloatingNav from "@/components/FloatingNav";
import StageRecording from "@/components/StageRecording";
import StagePreview from "@/components/StagePreview";
import StageProcessing from "@/components/StageProcessing";
import StageAvatarUpload from "@/components/StageAvatarUpload";
import Dashboard from "@/components/Dashboard";
import { useEchoState } from "@/hooks/useEchoState";

const STAGE_LABELS: Record<string, string> = {
  welcome:            "01 — Start",
  recording:          "02 — Record",
  preview:            "03 — Preview",
  processing:         "04 — Processing",
  avatar_upload:      "05 — Avatar",
  avatar_processing:  "06 — Generating",
  dashboard:          "07 — Your Twin",
  error:              "Error",
};

export default function Home() {
  const echo = useEchoState();
  const { stage, error } = echo;

  // Progress steps (shown at top during active flow)
  const progressSteps = ["Record", "Preview", "Process", "Avatar", "Dashboard"];
  const stepIndex = {
    welcome: -1, recording: 0, preview: 1, processing: 2,
    avatar_upload: 3, avatar_processing: 3, dashboard: 4, error: -1,
  }[stage] ?? -1;

  return (
    <>
      <FloatingNav />

      <main
        style={{
          minHeight: "100dvh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: stage === "welcome" ? "center" : "flex-start",
          padding: `clamp(5rem, 12vh, 8rem) var(--space-6) var(--space-16)`,
          position: "relative",
        }}
      >
        {/* Ambient glow */}
        <div aria-hidden style={{ position: "fixed", top: "30%", left: "50%", transform: "translate(-50%, -50%)", width: "min(600px, 90vw)", height: "min(600px, 90vw)", borderRadius: "9999px", background: "radial-gradient(ellipse, oklch(68% 0.22 255 / 0.05) 0%, transparent 70%)", pointerEvents: "none", zIndex: 0 }} />

        <div style={{ width: "100%", maxWidth: 600, position: "relative", zIndex: 1 }}>

          {/* ── Welcome ──────────────────────────────────────────── */}
          {stage === "welcome" && (
            <div className="animate-reveal" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "var(--space-8)", textAlign: "center" }}>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--color-accent)" }}>
                Voice Enrollment Demo
              </span>

              <h1 style={{ fontFamily: "var(--font-display)", fontSize: "clamp(2.5rem, 7vw, 4.5rem)", fontWeight: 300, fontStyle: "normal", lineHeight: 1.08, letterSpacing: "-0.025em", color: "var(--color-ink)", margin: 0, overflowWrap: "anywhere" }}>
                Your voice.{" "}
                <span style={{ fontWeight: 600, background: "linear-gradient(135deg, var(--color-accent), oklch(72% 0.18 280))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
                  Echoed digitally.
                </span>
              </h1>

              <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", fontStyle: "normal", color: "var(--color-ink-3)", lineHeight: 1.65, maxWidth: 440, margin: 0 }}>
                Speak once. EchoSelf processes your voice through a 7-stage on-device neural pipeline — VAD, ASR, synthesis, watermarking, speaker verification — entirely locally.
              </p>

              <button
                onClick={echo.startRecording}
                style={{
                  display: "flex", alignItems: "center", gap: "var(--space-3)",
                  padding: "var(--space-4) var(--space-10)",
                  fontFamily: "var(--font-display)", fontSize: "var(--text-base)", fontWeight: 500,
                  color: "var(--color-paper)", background: "var(--color-accent)",
                  border: "none", borderRadius: "var(--radius-full)", cursor: "pointer",
                  boxShadow: "0 0 32px var(--color-accent-glow)",
                  transition: "opacity 0.15s ease, transform 0.1s ease",
                }}
                onMouseDown={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(0.97)"; }}
                onMouseUp={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(1)"; }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
                Begin Voice Enrollment
              </button>

              <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
                No account needed · Nothing leaves your machine · Free, open source
              </p>
            </div>
          )}

          {/* ── Active flow stages ────────────────────────────────── */}
          {stage !== "welcome" && stage !== "error" && (
            <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "var(--space-8)" }}>

              {/* Stage header + progress */}
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-accent)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                    {STAGE_LABELS[stage] ?? "EchoSelf"}
                  </span>
                  <button
                    onClick={echo.reset}
                    style={{ background: "none", border: "none", color: "var(--color-ink-4)", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", cursor: "pointer", textDecoration: "underline" }}
                  >
                    Start over
                  </button>
                </div>

                {/* Progress bar */}
                {stepIndex >= 0 && (
                  <div style={{ display: "flex", gap: "var(--space-2)" }}>
                    {progressSteps.map((label, i) => (
                      <div key={label} style={{ flex: 1, display: "flex", flexDirection: "column", gap: "var(--space-2)", alignItems: "center" }}>
                        <div style={{ width: "100%", height: 2, background: i <= stepIndex ? "var(--color-accent)" : "var(--color-paper-4)", borderRadius: 1, transition: "background 0.4s ease" }} />
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: i <= stepIndex ? "var(--color-accent)" : "var(--color-ink-4)", whiteSpace: "nowrap" }}>
                          {label}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Stage content */}
              {(stage === "recording") && (
                <StageRecording
                  analyser={echo.analyser}
                  timeSec={echo.recordingTimeSec}
                  isRecording={stage === "recording"}
                  onStart={echo.startRecording}
                  onStop={echo.stopRecording}
                />
              )}

              {stage === "preview" && echo.recordingBlobUrl && (
                <StagePreview
                  blobUrl={echo.recordingBlobUrl}
                  timeSec={echo.recordingTimeSec}
                  onSubmit={echo.submitRecording}
                  onDiscard={echo.discardRecording}
                />
              )}

              {stage === "processing" && <StageProcessing />}

              {stage === "avatar_upload" && (
                <StageAvatarUpload onSubmit={echo.submitAvatar} onSkip={echo.skipAvatar} />
              )}

              {stage === "avatar_processing" && (
                <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)", alignItems: "center" }}>
                  <div style={{ width: 44, height: 44, border: "2px solid var(--color-paper-4)", borderTop: "2px solid var(--color-accent)", borderRadius: "9999px", animation: "spinner 0.9s linear infinite" }} />
                  <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", color: "var(--color-ink-2)" }}>Generating avatar…</p>
                </div>
              )}

              {stage === "dashboard" && echo.voiceResult && (
                <Dashboard
                  voiceResult={echo.voiceResult}
                  avatarResult={echo.avatarResult}
                  originalBlobUrl={echo.recordingBlobUrl}
                  onReset={echo.reset}
                />
              )}
            </div>
          )}

          {/* ── Error state ───────────────────────────────────────── */}
          {stage === "error" && (
            <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)", alignItems: "center", textAlign: "center" }}>
              <div style={{ width: 48, height: 48, borderRadius: "9999px", background: "oklch(65% 0.22 25 / 0.1)", border: "1px solid oklch(65% 0.22 25 / 0.3)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-error)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" x2="12" y1="8" y2="12"/>
                  <line x1="12" x2="12.01" y1="16" y2="16"/>
                </svg>
              </div>
              <div>
                <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-lg)", fontWeight: 500, fontStyle: "normal", color: "var(--color-ink)", marginBottom: "var(--space-3)" }}>Something went wrong</p>
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-sm)", color: "var(--color-ink-3)", maxWidth: 440 }}>{error}</p>
              </div>
              <button
                onClick={echo.reset}
                style={{ padding: "var(--space-3) var(--space-8)", fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 400, color: "var(--color-ink-3)", background: "transparent", border: "1px solid var(--color-border)", borderRadius: "var(--radius-full)", cursor: "pointer" }}
              >
                Try Again
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer style={{ position: "fixed", bottom: 0, left: 0, right: 0, padding: "var(--space-4) var(--space-6)", borderTop: "var(--rule-hairline)", background: "oklch(7% 0.012 245 / 0.95)", backdropFilter: "blur(12px)", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "var(--space-3)", zIndex: 10 }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
            EchoSelf · Local-first voice AI · ₹0 cloud cost
          </span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
            PyTorch · Next.js · RTX 3050
          </span>
        </footer>
      </main>
    </>
  );
}
