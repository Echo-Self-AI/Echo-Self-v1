"use client";

interface StagePreviewProps {
  blobUrl: string;
  timeSec: number;
  onSubmit: () => void;
  onDiscard: () => void;
}

export default function StagePreview({ blobUrl, timeSec, onSubmit, onDiscard }: StagePreviewProps) {
  const mm = String(Math.floor(timeSec / 60)).padStart(2, "0");
  const ss = String(timeSec % 60).padStart(2, "0");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)", width: "100%", alignItems: "center" }}>
      <div style={{ width: "100%", padding: "var(--space-5) var(--space-6)", background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "var(--space-4)" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            Your Recording
          </span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
            {mm}:{ss}
          </span>
        </div>
        <audio
          src={blobUrl}
          controls
          style={{ width: "100%", height: 36 }}
        />
      </div>

      <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", color: "var(--color-ink-3)", textAlign: "center" }}>
        Listen to your recording. Happy with it?
      </p>

      <div style={{ display: "flex", gap: "var(--space-4)", flexWrap: "wrap", justifyContent: "center" }}>
        <button
          onClick={onDiscard}
          style={{
            padding: "var(--space-3) var(--space-7)",
            fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 400,
            color: "var(--color-ink-3)", background: "transparent",
            border: "1px solid var(--color-border)", borderRadius: "var(--radius-full)",
            cursor: "pointer", transition: "border-color 0.15s ease, color 0.15s ease",
          }}
          onMouseEnter={(e) => { const b = e.currentTarget as HTMLButtonElement; b.style.borderColor = "var(--color-border-light)"; b.style.color = "var(--color-ink-2)"; }}
          onMouseLeave={(e) => { const b = e.currentTarget as HTMLButtonElement; b.style.borderColor = "var(--color-border)"; b.style.color = "var(--color-ink-3)"; }}
        >
          Record Again
        </button>

        <button
          onClick={onSubmit}
          style={{
            padding: "var(--space-3) var(--space-7)",
            fontFamily: "var(--font-display)", fontSize: "var(--text-sm)", fontWeight: 500,
            color: "var(--color-paper)", background: "var(--color-accent)",
            border: "none", borderRadius: "var(--radius-full)", cursor: "pointer",
            transition: "opacity 0.15s ease",
            boxShadow: "0 0 20px var(--color-accent-glow)",
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = "0.85"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = "1"; }}
        >
          Process My Voice →
        </button>
      </div>
    </div>
  );
}
