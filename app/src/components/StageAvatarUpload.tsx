"use client";

import { useRef, useState } from "react";

interface StageAvatarUploadProps {
  onSubmit: (file: File) => void;
  onSkip: () => void;
}

export default function StageAvatarUpload({ onSubmit, onSkip }: StageAvatarUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  function handleFile(f: File) {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)", width: "100%", alignItems: "center" }}>
      <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", color: "var(--color-ink-2)", textAlign: "center" }}>
        Upload a portrait to generate your digital avatar.
      </p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          width: "100%", minHeight: 160,
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "var(--space-4)",
          border: `1px dashed ${preview ? "var(--color-accent)" : "var(--color-border-light)"}`,
          borderRadius: "var(--radius-lg)",
          background: preview ? "transparent" : "var(--color-paper-2)",
          cursor: "pointer",
          transition: "border-color 0.2s ease",
          overflow: "hidden",
          position: "relative",
          padding: preview ? 0 : "var(--space-8)",
        }}
      >
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt="Portrait preview"
            style={{ width: "100%", height: 220, objectFit: "cover", display: "block" }}
          />
        ) : (
          <>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
              <circle cx="9" cy="9" r="2"/>
              <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
            </svg>
            <span style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", color: "var(--color-ink-3)" }}>
              Drop a portrait here, or click to browse
            </span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
              JPG, PNG, WEBP — your image stays local
            </span>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />
      </div>

      {preview && (
        <button
          onClick={() => { setPreview(null); setFile(null); }}
          style={{ background: "none", border: "none", color: "var(--color-ink-4)", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", cursor: "pointer", textDecoration: "underline" }}
        >
          Choose different image
        </button>
      )}

      <div style={{ display: "flex", gap: "var(--space-4)", flexWrap: "wrap", justifyContent: "center" }}>
        <button
          onClick={onSkip}
          style={{
            padding: "var(--space-3) var(--space-7)",
            fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 400,
            color: "var(--color-ink-4)", background: "transparent",
            border: "1px solid var(--color-border)", borderRadius: "var(--radius-full)", cursor: "pointer",
          }}
        >
          Skip Avatar
        </button>

        {file && (
          <button
            onClick={() => file && onSubmit(file)}
            style={{
              padding: "var(--space-3) var(--space-7)",
              fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 500,
              color: "var(--color-paper)", background: "var(--color-accent)",
              border: "none", borderRadius: "var(--radius-full)", cursor: "pointer",
              boxShadow: "0 0 20px var(--color-accent-glow)",
            }}
          >
            Generate Avatar →
          </button>
        )}
      </div>

      <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", textAlign: "center", maxWidth: 400 }}>
        Note: the avatar generator uses an untrained model — the output will be abstract, not a photorealistic likeness. This demonstrates the generation pipeline.
      </p>
    </div>
  );
}
