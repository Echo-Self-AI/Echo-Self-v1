"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const NAV_LINKS = [
  { href: "/", label: "Twin", icon: IconTwin },
  { href: "/studio/video", label: "Video", icon: IconVideo },
  { href: "/studio/voice", label: "Voice", icon: IconMic },
  { href: "/studio/avatar", label: "Avatar", icon: IconAvatar },
  { href: "/library", label: "Library", icon: IconLibrary },
];

export default function AppNav() {
  const path = usePathname();
  const [serviceOnline, setServiceOnline] = useState<boolean | null>(null);

  useEffect(() => {
    fetch("/api/echoself", { signal: AbortSignal.timeout(3000) })
      .then((r) => r.json())
      .then((d) => setServiceOnline(d.online ?? false))
      .catch(() => setServiceOnline(false));
  }, []);

  return (
    <nav
      style={{
        position: "fixed",
        top: 0, left: 0, right: 0,
        zIndex: 100,
        height: 52,
        display: "flex",
        alignItems: "center",
        padding: "0 var(--space-6)",
        background: "oklch(9% 0.013 245 / 0.92)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderBottom: "var(--rule-hairline)",
        gap: "var(--space-2)",
      }}
    >
      {/* Wordmark */}
      <Link href="/" style={{ textDecoration: "none", marginRight: "var(--space-6)", flexShrink: 0 }}>
        <span style={{
          fontFamily: "var(--font-display)",
          fontSize: "var(--text-sm)",
          fontWeight: 700,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--color-ink)",
        }}>
          EchoSelf
        </span>
      </Link>

      {/* Nav links */}
      <div style={{ display: "flex", alignItems: "center", gap: "var(--space-1)", flex: 1 }}>
        {NAV_LINKS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? path === "/" : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "var(--space-2)",
                padding: "var(--space-2) var(--space-3)",
                borderRadius: "var(--radius-sm)",
                textDecoration: "none",
                fontFamily: "var(--font-display)",
                fontSize: "var(--text-sm)",
                fontWeight: active ? 500 : 400,
                color: active ? "var(--color-ink)" : "var(--color-ink-3)",
                background: active ? "var(--color-paper-3)" : "transparent",
                transition: "background 0.15s ease, color 0.15s ease",
                whiteSpace: "nowrap",
              }}
            >
              <Icon size={14} active={active} />
              <span>{label}</span>
            </Link>
          );
        })}
      </div>

      {/* Service status */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--space-2)",
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-xs)",
        color: "var(--color-ink-4)",
        flexShrink: 0,
      }}>
        <span style={{
          width: 6, height: 6, borderRadius: "9999px", display: "inline-block",
          background: serviceOnline === null
            ? "var(--color-ink-4)"
            : serviceOnline
            ? "var(--color-success)"
            : "oklch(78% 0.18 70)",
          boxShadow: serviceOnline ? "0 0 5px var(--color-success)" : "none",
        }} />
        <span style={{ display: "none" }}>
          {serviceOnline === null ? "…" : serviceOnline ? "AI Online" : "Demo Mode"}
        </span>
        <span className="nav-status-label">
          {serviceOnline === null ? "Checking" : serviceOnline ? "AI Online" : "Demo Mode"}
        </span>
      </div>

      <style>{`
        @media (max-width: 600px) {
          .nav-status-label { display: none; }
        }
      `}</style>
    </nav>
  );
}

function IconTwin({ size, active }: { size: number; active: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={active ? "var(--color-accent)" : "currentColor"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
    </svg>
  );
}
function IconVideo({ size, active }: { size: number; active: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={active ? "var(--color-accent)" : "currentColor"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="6" width="14" height="12" rx="2"/><polygon points="22 8 16 12 22 16 22 8"/>
    </svg>
  );
}
function IconMic({ size, active }: { size: number; active: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={active ? "var(--color-accent)" : "currentColor"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>
    </svg>
  );
}
function IconAvatar({ size, active }: { size: number; active: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={active ? "var(--color-accent)" : "currentColor"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="12" cy="10" r="3"/><path d="M7 20c0-3 2.2-5 5-5s5 2 5 5"/>
    </svg>
  );
}
function IconLibrary({ size, active }: { size: number; active: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={active ? "var(--color-accent)" : "currentColor"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="3" width="6" height="6" rx="1"/><rect x="3" y="15" width="6" height="6" rx="1"/><rect x="15" y="15" width="6" height="6" rx="1"/>
    </svg>
  );
}
