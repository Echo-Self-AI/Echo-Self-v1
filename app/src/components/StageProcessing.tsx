"use client";

const STEPS = [
  "Registering voice consent…",
  "Extracting speaker embedding from your recording…",
  "Synthesizing speech through acoustic model…",
  "Embedding provenance watermark…",
  "Verifying watermark integrity…",
  "Writing audit log entry…",
];

export default function StageProcessing() {
  return (
    <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: "var(--space-5)", alignItems: "center" }}>
      <div
        style={{
          width: 44, height: 44,
          border: "2px solid var(--color-paper-4)",
          borderTop: "2px solid var(--color-accent)",
          borderRadius: "9999px",
          animation: "spinner 0.9s linear infinite",
        }}
      />

      <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", color: "var(--color-ink-2)", textAlign: "center" }}>
        Processing your voice…
      </p>

      <div style={{ width: "100%", padding: "var(--space-5)", background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-md)", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
        {STEPS.map((step, i) => (
          <div
            key={step}
            style={{
              display: "flex", alignItems: "center", gap: "var(--space-3)",
              opacity: 0,
              animation: `fade-in 0.35s ease ${i * 0.5}s both`,
            }}
          >
            <span style={{ width: 5, height: 5, borderRadius: "9999px", background: "var(--color-accent)", display: "inline-block", flexShrink: 0, animation: `pulse-ring 1.8s ${i * 0.4}s ease-out infinite` }} />
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)" }}>{step}</span>
          </div>
        ))}
      </div>

      <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
        All processing is local — nothing leaves your machine.
      </p>
    </div>
  );
}
