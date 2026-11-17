"use client";

import { useEffect, useState } from "react";

type ServiceStatus = "checking" | "online" | "offline";

export default function FloatingNav() {
  const [status, setStatus] = useState<ServiceStatus>("checking");

  useEffect(() => {
    async function check() {
      try {
        const res = await fetch("/api/echoself", { signal: AbortSignal.timeout(4000) });
        const data = await res.json();
        setStatus(data.online ? "online" : "offline");
      } catch {
        setStatus("offline");
      }
    }
    check();
  }, []);

  const statusDot: Record<ServiceStatus, { color: string; label: string }> = {
    checking: { color: "var(--color-ink-4)",   label: "Checking…" },
    online:   { color: "var(--color-success)",  label: "AI Service Online" },
    offline:  { color: "var(--color-warning)",  label: "Demo Mode" },
  };

  const { color, label } = statusDot[status];

  return (
    <header
      style={{
        position: "fixed",
        top: "var(--space-5)",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 100,
        width: "fit-content",
      }}
    >
      <nav
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--space-5)",
          padding: "var(--space-3) var(--space-5)",
          background: "oklch(12% 0.014 245 / 0.85)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          border: "var(--rule-hairline)",
          borderRadius: "var(--radius-full)",
          whiteSpace: "nowrap",
        }}
      >
        {/* Wordmark */}
        <span
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "var(--text-sm)",
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--color-ink)",
          }}
        >
          EchoSelf
        </span>

        {/* Divider */}
        <span style={{ width: 1, height: 14, background: "var(--color-border-light)", display: "block" }} />

        {/* Service status pill */}
        <span
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--space-2)",
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-xs)",
            color: "var(--color-ink-3)",
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "var(--radius-full)",
              background: color,
              display: "inline-block",
              boxShadow: status === "online" ? `0 0 6px ${color}` : "none",
              transition: `background ${process.env.NODE_ENV ? "0.3s" : "0.3s"} ease`,
            }}
          />
          {label}
        </span>
      </nav>
    </header>
  );
}
