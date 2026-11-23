"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type UploadState = "idle" | "processing" | "done" | "error";

async function apiCall(job_type: string, extra: Record<string, unknown>) {
  const res = await fetch("/api/echoself", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_type, user_id: "demo-user", ...extra }),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

async function fileToBase64(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = "";
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK)
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  return btoa(bin);
}

export default function AvatarStudioPage() {
  const [state, setState] = useState<UploadState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<{ imageBase64: string; size: [number,number]; sha256: string; method: string } | null>(null);
  const [existingAvatar, setExistingAvatar] = useState<{ imageBase64: string; size: [number,number]; createdAt: string } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load existing avatar
  useEffect(() => {
    apiCall("portrait_load", { avatar_id: "primary" })
      .then((r) => {
        const out = r.output ?? r;
        if (out.found) {
          setExistingAvatar({ imageBase64: out.image_base64, size: out.output_size ?? [512,512], createdAt: out.created_at ?? "" });
        }
      })
      .catch(() => {});
  }, []);

  function handleFile(f: File) {
    if (!f.type.startsWith("image/")) { setError("Please upload an image file (JPG, PNG, WEBP)."); return; }
    setFile(f);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
    setState("idle");
    setResult(null);
  }

  const processPortrait = useCallback(async () => {
    if (!file) return;
    setState("processing");
    setError(null);
    try {
      await apiCall("consent_register", { modality: "visual", avatar_id: "primary", scope: "all" });
      const b64 = await fileToBase64(file);
      const res = await apiCall("portrait_process", { image_base64: b64, avatar_id: "primary" });
      const out = res.output ?? res;
      if (res.status !== "success") throw new Error(res.error ?? "Processing failed");
      setResult({ imageBase64: out.image_base64, size: out.output_size ?? [512,512], sha256: out.sha256 ?? "", method: out.processing_method ?? "pillow" });
      setExistingAvatar({ imageBase64: out.image_base64, size: out.output_size ?? [512,512], createdAt: out.created_at ?? "" });
      setState("done");
    } catch (err) {
      setState("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [file]);

  return (
    <main style={{ minHeight: "calc(100dvh - 52px)", padding: "var(--space-8) var(--space-6)", maxWidth: 900, margin: "0 auto" }}>
      <h1 style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-3xl)", fontWeight: 600, fontStyle: "normal", color: "var(--color-ink)", margin: 0, letterSpacing: "-0.02em", marginBottom: "var(--space-2)" }}>
        Avatar Studio
      </h1>
      <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", color: "var(--color-ink-3)", marginBottom: "var(--space-8)", fontStyle: "normal" }}>
        Upload your portrait. EchoSelf processes it locally into a standardised 512×512 avatar for use across all studio features.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-6)", alignItems: "start" }}>

        {/* Upload panel */}
        <div style={{ background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)", overflow: "hidden" }}>
          <div style={{ padding: "var(--space-3) var(--space-5)", borderBottom: "var(--rule-hairline)" }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em" }}>Upload Portrait</span>
          </div>
          <div style={{ padding: "var(--space-5)", display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
              onClick={() => inputRef.current?.click()}
              style={{
                minHeight: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                border: `1px dashed ${previewUrl ? "var(--color-accent)" : "var(--color-border-light)"}`,
                borderRadius: "var(--radius-md)", cursor: "pointer", overflow: "hidden", position: "relative",
                background: previewUrl ? "transparent" : "var(--color-paper-3)", gap: "var(--space-4)",
                padding: previewUrl ? 0 : "var(--space-8)",
              }}
            >
              {previewUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={previewUrl} alt="Portrait preview" style={{ width: "100%", height: 220, objectFit: "cover" }} />
              ) : (
                <>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
                  </svg>
                  <span style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", color: "var(--color-ink-3)", fontStyle: "normal" }}>Drop portrait here, or click to browse</span>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>JPG, PNG, WEBP — min 64px — stays on your device</span>
                </>
              )}
              <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
            </div>

            {error && <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-error)" }}>⚠ {error}</p>}

            {file && state !== "processing" && (
              <button onClick={processPortrait} style={{ padding: "var(--space-3) var(--space-6)", fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 500, color: "var(--color-paper)", background: "var(--color-accent)", border: "none", borderRadius: "var(--radius-full)", cursor: "pointer", boxShadow: "0 0 20px var(--color-accent-glow)" }}>
                Process Portrait →
              </button>
            )}
            {state === "processing" && (
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
                <div style={{ width: 20, height: 20, border: "2px solid var(--color-paper-4)", borderTop: "2px solid var(--color-accent)", borderRadius: "9999px", animation: "spinner 0.9s linear infinite" }} />
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)" }}>Processing locally…</span>
              </div>
            )}
          </div>
        </div>

        {/* Result / existing avatar */}
        <div style={{ background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)", overflow: "hidden" }}>
          <div style={{ padding: "var(--space-3) var(--space-5)", borderBottom: "var(--rule-hairline)", display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em" }}>Processed Avatar</span>
            {existingAvatar && <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-success)" }}>✓ Enrolled</span>}
          </div>
          <div style={{ padding: "var(--space-5)" }}>
            {(result?.imageBase64 || existingAvatar?.imageBase64) ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/png;base64,${result?.imageBase64 ?? existingAvatar?.imageBase64}`}
                  alt="Processed avatar"
                  style={{ width: "100%", borderRadius: "var(--radius-md)", display: "block" }}
                />
                <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
                  <InfoRow label="Size" value={`${(result?.size ?? existingAvatar?.size ?? [0,0])[0]}×${(result?.size ?? existingAvatar?.size ?? [0,0])[1]}px`} />
                  <InfoRow label="Method" value={result?.method ?? "pillow-portrait-processing"} />
                  {result?.sha256 && <InfoRow label="SHA256" value={result.sha256.slice(0, 16) + "…"} />}
                  <InfoRow label="Storage" value="Local only — never uploaded" />
                </div>
                <div style={{ padding: "var(--space-3) var(--space-4)", background: "oklch(68% 0.22 255 / 0.08)", border: "1px solid oklch(68% 0.22 255 / 0.2)", borderRadius: "var(--radius-md)", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)" }}>
                  This portrait is now used as the anchor frame in Video Studio. The processing is PIL-based image enhancement — not generative AI.
                </div>
              </div>
            ) : (
              <div style={{ minHeight: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "var(--space-4)" }}>
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-4)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                </svg>
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", textAlign: "center" }}>No avatar enrolled yet. Upload a portrait on the left.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* What this does */}
      <div style={{ marginTop: "var(--space-8)", padding: "var(--space-5) var(--space-6)", background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)" }}>
        <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "var(--space-4)" }}>What portrait processing does</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "var(--space-4)" }}>
          {[
            ["Centre-crop to square", "Takes the centred square from your portrait"],
            ["Resize to 512×512", "LANCZOS high-quality resampling"],
            ["Enhance contrast & colour", "Mild PIL ImageEnhance — ×1.15 contrast, ×1.10 saturation"],
            ["Sharpen & auto-levels", "×1.20 sharpness, autocontrast 0.5% cutoff"],
            ["Save original", "Original JPEG preserved at data/avatars/"],
          ].map(([title, desc]) => (
            <div key={title}>
              <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 500, fontStyle: "normal", color: "var(--color-ink-2)", marginBottom: "var(--space-1)" }}>{title}</p>
              <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-xs)", fontStyle: "normal", color: "var(--color-ink-4)", lineHeight: 1.5 }}>{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between" }}>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)" }}>{value}</span>
    </div>
  );
}
