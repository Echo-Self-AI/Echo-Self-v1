"use client";

import { useEffect, useState } from "react";

interface AuditRecord {
  event_id: string;
  action: string;
  user_id: string | null;
  timestamp: string;
  model_id: string;
  watermark_type: string;
  watermark_id: string | null;
  output_hash: string | null;
  detection_result: boolean | null;
  verification_status: boolean | null;
}

export default function LibraryPage() {
  const [records, setRecords] = useState<AuditRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [avatarInfo, setAvatarInfo] = useState<{ found: boolean; size?: [number, number]; createdAt?: string; thumb?: string } | null>(null);
  const [serviceOnline, setServiceOnline] = useState<boolean | null>(null);

  useEffect(() => {
    // Check service
    fetch("/api/echoself", { signal: AbortSignal.timeout(3000) })
      .then((r) => r.json())
      .then((d) => setServiceOnline(d.online ?? false))
      .catch(() => setServiceOnline(false));

    // Load avatar
    fetch("/api/echoself", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_type: "portrait_load", user_id: "demo-user", avatar_id: "primary" }),
    })
      .then((r) => r.json())
      .then((d) => {
        const out = d.output ?? d;
        setAvatarInfo({ found: out.found ?? false, size: out.output_size, createdAt: out.created_at, thumb: out.thumbnail_base64 });
      })
      .catch(() => setAvatarInfo({ found: false }));

    // Load audit log — read from API
    fetch("/api/echoself/audit", { signal: AbortSignal.timeout(3000) })
      .then((r) => r.ok ? r.json() : null)
      .then((d) => {
        if (d?.records) setRecords(d.records.slice().reverse().slice(0, 50));
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <main style={{ minHeight: "calc(100dvh - 52px)", padding: "var(--space-8) var(--space-6)", maxWidth: 1100, margin: "0 auto" }}>
      <h1 style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-3xl)", fontWeight: 600, fontStyle: "normal", color: "var(--color-ink)", margin: 0, letterSpacing: "-0.02em", marginBottom: "var(--space-2)" }}>
        Library
      </h1>
      <p style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", color: "var(--color-ink-3)", marginBottom: "var(--space-8)", fontStyle: "normal" }}>
        Your enrolled assets and generation history.
      </p>

      {/* Asset cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))", gap: "var(--space-5)", marginBottom: "var(--space-10)" }}>

        {/* Avatar card */}
        <AssetCard
          title="Portrait Avatar"
          icon="👤"
          status={avatarInfo === null ? "loading" : avatarInfo.found ? "enrolled" : "missing"}
          statusLabel={avatarInfo?.found ? "Enrolled" : "Not uploaded"}
          detail={avatarInfo?.found ? `${avatarInfo.size?.[0]}×${avatarInfo.size?.[1]}px · pillow-portrait-processing` : undefined}
          createdAt={avatarInfo?.createdAt}
          actionLabel={avatarInfo?.found ? "Manage" : "Upload"}
          actionHref="/studio/avatar"
          thumb={avatarInfo?.thumb ? `data:image/png;base64,${avatarInfo.thumb}` : undefined}
        />

        {/* Voice card */}
        <AssetCard
          title="Voice Profile"
          icon="🎙"
          status="missing"
          statusLabel="Not trained"
          detail="Acoustic model: untrained. Speaker embeddings: extracted but generic."
          actionLabel="Go to Voice Studio"
          actionHref="/studio/voice"
        />

        {/* AI service card */}
        <AssetCard
          title="AI Service"
          icon="⚙"
          status={serviceOnline === null ? "loading" : serviceOnline ? "enrolled" : "missing"}
          statusLabel={serviceOnline === null ? "Checking…" : serviceOnline ? "Running on :8000" : "Offline"}
          detail={serviceOnline ? "portrait_process · voice_synthesis · talking_head_video · audit_log" : "Start with: cd ai && .venv/bin/python -m echoself_ai.service.server --port 8000"}
          actionLabel="Health check"
          actionHref="/api/echoself"
        />
      </div>

      {/* Audit log */}
      <div style={{ background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)", overflow: "hidden" }}>
        <div style={{ padding: "var(--space-4) var(--space-6)", borderBottom: "var(--rule-hairline)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-3)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            Generation Audit Log
          </span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
            {records.length} records
          </span>
        </div>

        {loading ? (
          <div style={{ padding: "var(--space-8)", textAlign: "center", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>Loading…</div>
        ) : records.length === 0 ? (
          <div style={{ padding: "var(--space-8)", textAlign: "center" }}>
            <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)" }}>
              No audit records yet. Generate something in the Video Studio to see events here.
            </p>
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)" }}>
              <thead>
                <tr style={{ borderBottom: "var(--rule-hairline)" }}>
                  {["Action", "Model", "Watermark", "Hash", "Timestamp"].map((h) => (
                    <th key={h} style={{ padding: "var(--space-3) var(--space-4)", textAlign: "left", color: "var(--color-ink-4)", fontWeight: 400, letterSpacing: "0.08em", textTransform: "uppercase", whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {records.map((r) => (
                  <tr key={r.event_id} style={{ borderBottom: "1px solid var(--color-border)" }}>
                    <td style={{ padding: "var(--space-3) var(--space-4)", color: "var(--color-ink-2)", whiteSpace: "nowrap" }}>{r.action}</td>
                    <td style={{ padding: "var(--space-3) var(--space-4)", color: "var(--color-ink-3)" }}>{r.model_id}</td>
                    <td style={{ padding: "var(--space-3) var(--space-4)", color: "var(--color-ink-4)" }}>{r.watermark_id ? r.watermark_id.slice(0, 12) + "…" : "—"}</td>
                    <td style={{ padding: "var(--space-3) var(--space-4)", color: "var(--color-ink-4)" }}>{r.output_hash ? r.output_hash.slice(0, 12) + "…" : "—"}</td>
                    <td style={{ padding: "var(--space-3) var(--space-4)", color: "var(--color-ink-4)", whiteSpace: "nowrap" }}>{new Date(r.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}

function AssetCard({ title, icon, status, statusLabel, detail, actionLabel, actionHref, thumb, createdAt }: {
  title: string; icon: string; status: "enrolled" | "missing" | "loading";
  statusLabel: string; detail?: string; actionLabel: string; actionHref: string;
  thumb?: string; createdAt?: string;
}) {
  const statusColor = status === "enrolled" ? "var(--color-success)" : status === "loading" ? "var(--color-ink-4)" : "oklch(78% 0.18 70)";
  return (
    <div style={{ background: "var(--color-paper-2)", border: "var(--rule-hairline)", borderRadius: "var(--radius-lg)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "var(--space-5)", flex: 1 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "var(--space-4)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
            {thumb ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={thumb} alt={title} style={{ width: 40, height: 40, borderRadius: "var(--radius-sm)", objectFit: "cover", imageRendering: "auto" }} />
            ) : (
              <span style={{ fontSize: "1.5rem" }}>{icon}</span>
            )}
            <span style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-base)", fontWeight: 500, fontStyle: "normal", color: "var(--color-ink)" }}>{title}</span>
          </div>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: statusColor, display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
            <span style={{ width: 5, height: 5, borderRadius: "9999px", background: statusColor, display: "inline-block" }} />
            {statusLabel}
          </span>
        </div>
        {detail && <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", lineHeight: 1.6 }}>{detail}</p>}
        {createdAt && <p style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-ink-4)", marginTop: "var(--space-2)" }}>Created {new Date(createdAt).toLocaleDateString()}</p>}
      </div>
      <div style={{ borderTop: "var(--rule-hairline)", padding: "var(--space-3) var(--space-5)" }}>
        <a href={actionHref} style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--color-accent)", textDecoration: "none" }}>
          {actionLabel} →
        </a>
      </div>
    </div>
  );
}
