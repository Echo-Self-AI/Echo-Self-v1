"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────

type GenStatus =
  | "idle"
  | "loading_avatar"
  | "synthesizing_voice"
  | "generating_video"
  | "done"
  | "error";

interface VideoSettings {
  fps: 12 | 25;
  width: 512 | 768;
  expression: "neutral" | "confident" | "happy" | "serious";
  speech: boolean;
  script: string;
}

interface GenerationResult {
  videoBase64: string;
  audioBase64: string;
  durationSec: number;
  fps: number;
  width: number;
  height: number;
  numFrames: number;
  method: string;
  avatarSize: [number, number];
  isDemo: boolean;
}

// ── Helpers ────────────────────────────────────────────────────────────────

async function apiCall(job_type: string, extra: Record<string, unknown>) {
  const res = await fetch("/api/echoself", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_type, user_id: "demo-user", ...extra }),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

function b64ToObjectUrl(b64: string, mime: string): string {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

// ── Component ──────────────────────────────────────────────────────────────

export default function VideoStudioPage() {
  const [prompt, setPrompt] = useState(
    "Create a confident introduction. I look directly into the camera and introduce EchoSelf as a privacy-first, local AI digital twin platform. Professional lighting, dark background."
  );
  const [settings, setSettings] = useState<VideoSettings>({
    fps: 25,
    width: 512,
    expression: "confident",
    speech: true,
    script: "Hello. I'm your EchoSelf digital twin. Everything you see and hear was generated locally on your device — no cloud, no subscriptions, no data leaving your machine.",
  });

  const [status, setStatus] = useState<GenStatus>("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [avatarInfo, setAvatarInfo] = useState<{ found: boolean; size?: [number, number]; thumb?: string } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const videoUrlRef = useRef<string | null>(null);
  const audioUrlRef = useRef<string | null>(null);

  // Check for existing avatar on mount
  useEffect(() => {
    apiCall("portrait_load", { avatar_id: "primary" })
      .then((r) => {
        const out = r.output ?? r;
        setAvatarInfo({
          found: out.found ?? false,
          size: out.output_size ?? undefined,
          thumb: out.thumbnail_base64 ?? undefined,
        });
      })
      .catch(() => setAvatarInfo({ found: false }));
  }, []);

  const generate = useCallback(async () => {
    setStatus("idle");
    setError(null);
    setResult(null);

    // Cleanup previous URLs
    if (videoUrlRef.current) { URL.revokeObjectURL(videoUrlRef.current); videoUrlRef.current = null; }
    if (audioUrlRef.current) { URL.revokeObjectURL(audioUrlRef.current); audioUrlRef.current = null; }

    try {
      // 1. Consent
      await apiCall("consent_register", { modality: "all", speaker_id: "demo-speaker", avatar_id: "primary", scope: "all" });

      // 2. Load avatar
      setStatus("loading_avatar");
      setStatusMsg("Loading your portrait…");
      const avatarRes = await apiCall("portrait_load", { avatar_id: "primary" });
      const avatarOut = avatarRes.output ?? avatarRes;
      if (!avatarOut.found) {
        throw new Error("No portrait found. Upload your portrait in the Avatar studio first.");
      }
      const portraitB64: string = avatarOut.image_base64;
      const avatarSize: [number, number] = avatarOut.output_size ?? [512, 512];

      // 3. Synthesize voice (if speech enabled)
      let audioB64: string | null = null;
      if (settings.speech) {
        setStatus("synthesizing_voice");
        setStatusMsg("Synthesizing speech…");
        const script = settings.script.trim() || prompt.slice(0, 200);
        const voiceRes = await apiCall("voice_synthesis", { text_input: script });
        const voiceOut = voiceRes.output ?? voiceRes;
        audioB64 = voiceOut.audio_base64 ?? null;
        if (!audioB64) throw new Error("Voice synthesis returned no audio.");
      }

      // 4. Generate talking-head video
      setStatus("generating_video");
      setStatusMsg("Generating talking-head video…");
      const vidPayload: Record<string, unknown> = {
        avatar_id: "primary",
        portrait_base64: portraitB64,
        fps: settings.fps,
        width: settings.width,
        height: settings.width, // square for now
      };
      if (audioB64) vidPayload.audio_base64 = audioB64;
      else vidPayload.text = settings.script || prompt.slice(0, 200);

      const vidRes = await apiCall("talking_head_video", vidPayload);
      const vidOut = vidRes.output ?? vidRes;

      if (vidRes.status !== "success" || !vidOut.video_base64) {
        throw new Error(vidRes.error ?? "Video generation returned no output.");
      }

      setResult({
        videoBase64: vidOut.video_base64,
        audioBase64: vidOut.audio_base64 ?? audioB64 ?? "",
        durationSec: vidOut.duration_sec ?? 0,
        fps: vidOut.fps ?? settings.fps,
        width: vidOut.width ?? settings.width,
        height: vidOut.height ?? settings.width,
        numFrames: vidOut.num_frames ?? 0,
        method: vidOut.generation_method ?? "geometric-portrait-animation",
        avatarSize,
        isDemo: vidRes.demo ?? false,
      });
      setStatus("done");
      setStatusMsg("Generation complete");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [prompt, settings]);

  // Wire video element when result arrives
  useEffect(() => {
    if (!result?.videoBase64 || !videoRef.current) return;
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    const url = b64ToObjectUrl(result.videoBase64, "video/mp4");
    videoUrlRef.current = url;
    videoRef.current.src = url;
    videoRef.current.load();
  }, [result]);

  function download() {
    if (!result?.videoBase64) return;
    const url = b64ToObjectUrl(result.videoBase64, "video/mp4");
    const a = document.createElement("a");
    a.href = url;
    a.download = `echoself-twin-${Date.now()}.mp4`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  const isGenerating = ["loading_avatar", "synthesizing_voice", "generating_video"].includes(status);

  return (
    <main style={{ minHeight: "calc(100dvh - 52px)", padding: "var(--space-8) var(--space-6)", maxWidth: 1100, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ marginBottom: "var(--space-8)" }}>
        <h1 style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-3xl)", fontWeight: 600, fontStyle: "normal", color: "var(--color-ink)", margin: 0, letterSpacing: "-0.02em" }}>
          Video Studio
        </h1>
        <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", color: "var(--color-ink-3)", marginTop: "var(--space-2)", fontStyle: "normal" }}>
          Generate a talking-head video with your enrolled portrait and synthesized voice.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr min(360px, 100%)", gap: "var(--space-6)", alignItems: "start" }}>

        {/* ── Left: prompt + preview ─────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>

          {/* Prompt editor */}
          <Panel title="Scene Description">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the video scene, mood, and what you want to say…"
              rows={4}
              style={{
                width: "100%", resize: "vertical", minHeight: 100,
                fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", color: "var(--color-ink-2)",
                background: "var(--color-paper-3)", border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)", padding: "var(--space-4)",
                outline: "none", lineHeight: 1.6, fontStyle: "normal",
              }}
              onFocus={(e) => { e.currentTarget.style.borderColor = "var(--color-accent-ring)"; }}
              onBlur={(e) => { e.currentTarget.style.borderColor = "var(--color-border)"; }}
            />
            <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", marginTop: "var(--space-2)" }}>
              Note: the description controls video metadata. Actual visual content is determined by your portrait + geometric animation.
            </p>
          </Panel>

          {/* Speech script */}
          {settings.speech && (
            <Panel title="Speech Script">
              <textarea
                value={settings.script}
                onChange={(e) => setSettings((s) => ({ ...s, script: e.target.value }))}
                placeholder="What should your digital twin say?"
                rows={3}
                style={{
                  width: "100%", resize: "vertical", minHeight: 72,
                  fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", color: "var(--color-ink-2)",
                  background: "var(--color-paper-3)", border: "1px solid var(--color-border)",
                  borderRadius: "var(--radius-md)", padding: "var(--space-4)",
                  outline: "none", lineHeight: 1.6, fontStyle: "normal",
                }}
                onFocus={(e) => { e.currentTarget.style.borderColor = "var(--color-accent-ring)"; }}
                onBlur={(e) => { e.currentTarget.style.borderColor = "var(--color-border)"; }}
              />
            </Panel>
          )}

          {/* Avatar status */}
          <Panel title="Selected Portrait">
            {avatarInfo === null ? (
              <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>Checking…</p>
            ) : avatarInfo.found ? (
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)" }}>
                {avatarInfo.thumb && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={`data:image/png;base64,${avatarInfo.thumb}`} alt="Avatar thumbnail" style={{ width: 56, height: 56, borderRadius: "var(--radius-md)", objectFit: "cover" }} />
                )}
                <div>
                  <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontStyle: "normal", color: "var(--color-ink)" }}>Portrait loaded</p>
                  {avatarInfo.size && (
                    <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
                      {avatarInfo.size[0]}×{avatarInfo.size[1]}px · pillow-portrait-processing
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)" }}>
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "oklch(78% 0.18 70)" }}>
                  ⚠ No portrait found. <a href="/studio/avatar" style={{ color: "var(--color-accent)", textDecoration: "underline" }}>Upload in Avatar studio →</a>
                </p>
              </div>
            )}
          </Panel>

          {/* Video result */}
          {(status === "done" || status === "error" || isGenerating) && (
            <Panel title={status === "error" ? "Error" : "Output"}>
              {isGenerating && (
                <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)", padding: "var(--space-4) 0" }}>
                  <div style={{ width: 24, height: 24, border: "2px solid var(--color-paper-4)", borderTop: "2px solid var(--color-accent)", borderRadius: "9999px", animation: "spinner 0.9s linear infinite", flexShrink: 0 }} />
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)" }}>{statusMsg}</span>
                </div>
              )}
              {status === "error" && (
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-error)" }}>⚠ {error}</p>
              )}
              {status === "done" && result && (
                <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
                  <video
                    ref={videoRef}
                    controls
                    playsInline
                    style={{ width: "100%", borderRadius: "var(--radius-md)", background: "var(--color-paper-3)", maxHeight: 480 }}
                  />
                  <div style={{ display: "flex", gap: "var(--space-3)", flexWrap: "wrap" }}>
                    <MetaBadge label="Duration" value={`${result.durationSec.toFixed(1)}s`} />
                    <MetaBadge label="Frames" value={`${result.numFrames} @ ${result.fps}fps`} />
                    <MetaBadge label="Resolution" value={`${result.width}×${result.height}`} />
                    <MetaBadge label="Method" value={result.method} warn />
                  </div>
                  <div style={{ padding: "var(--space-3) var(--space-4)", background: "oklch(78% 0.18 70 / 0.08)", border: "1px solid oklch(78% 0.18 70 / 0.25)", borderRadius: "var(--radius-md)", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "oklch(78% 0.18 70)" }}>
                    Geometric portrait animation — mouth movement is driven by audio amplitude, not neural lip-sync. This is not a Wav2Lip or SadTalker output.
                  </div>
                  <button onClick={download} style={btnStyle("accent")}>
                    Download MP4
                  </button>
                </div>
              )}
            </Panel>
          )}
        </div>

        {/* ── Right: settings + generate ─────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)", position: "sticky", top: "calc(52px + var(--space-6))" }}>
          <Panel title="Settings">
            <SettingsRow label="FPS">
              <SegControl
                options={[{ value: 12, label: "12" }, { value: 25, label: "25" }]}
                value={settings.fps}
                onChange={(v) => setSettings((s) => ({ ...s, fps: v as 12 | 25 }))}
              />
            </SettingsRow>
            <SettingsRow label="Resolution">
              <SegControl
                options={[{ value: 512, label: "512" }, { value: 768, label: "768" }]}
                value={settings.width}
                onChange={(v) => setSettings((s) => ({ ...s, width: v as 512 | 768 }))}
              />
            </SettingsRow>
            <SettingsRow label="Speech">
              <SegControl
                options={[{ value: 1, label: "On" }, { value: 0, label: "Off" }]}
                value={settings.speech ? 1 : 0}
                onChange={(v) => setSettings((s) => ({ ...s, speech: v === 1 }))}
              />
            </SettingsRow>
          </Panel>

          <Panel title="Capability Status">
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <CapRow label="Portrait animation" status="available" />
              <CapRow label="Audio-driven mouth" status="available" />
              <CapRow label="Neural lip sync" status="unavailable" note="Wav2Lip not installed" />
              <CapRow label="Voice cloning" status="unavailable" note="Untrained acoustic model" />
              <CapRow label="Face generation" status="unavailable" note="Untrained generator" />
            </div>
          </Panel>

          <button
            onClick={generate}
            disabled={isGenerating || !avatarInfo?.found}
            style={btnStyle(isGenerating || !avatarInfo?.found ? "disabled" : "primary")}
          >
            {isGenerating ? statusMsg : "Generate Video →"}
          </button>
        </div>
      </div>
    </main>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

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

function SettingsRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "var(--space-4)" }}>
      <span style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontStyle: "normal", color: "var(--color-ink-3)" }}>{label}</span>
      {children}
    </div>
  );
}

function SegControl({ options, value, onChange }: {
  options: { value: number; label: string }[];
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ display: "flex", background: "var(--color-paper-3)", borderRadius: "var(--radius-sm)", padding: 2 }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          style={{
            padding: "var(--space-1) var(--space-3)",
            fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)",
            color: value === opt.value ? "var(--color-ink)" : "var(--color-ink-4)",
            background: value === opt.value ? "var(--color-paper-5)" : "transparent",
            border: "none", borderRadius: "var(--radius-sm)", cursor: "pointer",
            transition: "background 0.15s ease, color 0.15s ease",
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function CapRow({ label, status, note }: { label: string; status: "available" | "unavailable"; note?: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
      <span style={{ width: 6, height: 6, borderRadius: "9999px", background: status === "available" ? "var(--color-success)" : "var(--color-ink-4)", flexShrink: 0, display: "inline-block" }} />
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: status === "available" ? "var(--color-ink-3)" : "var(--color-ink-4)", flex: 1 }}>{label}</span>
      {note && <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>{note}</span>}
    </div>
  );
}

function MetaBadge({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", padding: "var(--space-1) var(--space-3)", background: "var(--color-paper-3)", borderRadius: "var(--radius-sm)", color: warn ? "oklch(78% 0.18 70)" : "var(--color-ink-3)" }}>
      {label}: {value}
    </span>
  );
}

function btnStyle(variant: "primary" | "accent" | "disabled") {
  const base: React.CSSProperties = {
    width: "100%", padding: "var(--space-4)",
    fontFamily: "var(--font-display)", fontSize: "var(--text-base)", fontWeight: 500,
    border: "none", borderRadius: "var(--radius-md)", cursor: "pointer",
    transition: "opacity 0.15s ease",
  };
  if (variant === "disabled") return { ...base, background: "var(--color-paper-3)", color: "var(--color-ink-4)", cursor: "not-allowed" };
  if (variant === "accent") return { ...base, background: "transparent", border: "1px solid var(--color-accent)", color: "var(--color-accent)" };
  return { ...base, background: "var(--color-accent)", color: "var(--color-paper)", boxShadow: "0 0 24px var(--color-accent-glow)" };
}
