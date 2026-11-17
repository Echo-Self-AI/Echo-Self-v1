"use client";

import { useState } from "react";
import FloatingNav from "@/components/FloatingNav";
import WaveformVisualizer from "@/components/WaveformVisualizer";
import RecordButton from "@/components/RecordButton";
import ResultCard from "@/components/ResultCard";
import { useEchoRecorder } from "@/hooks/useEchoRecorder";

export default function Home() {
  const { state, result, analyser, recordingTime, startRecording, stopRecording } =
    useEchoRecorder();

  const [hoveredFeature, setHoveredFeature] = useState<number | null>(null);

  const isActive = state === "recording";
  const showResult = state === "done" || state === "error";
  const showProcessing = state === "processing";

  const waveformMode =
    state === "recording" ? "recording"
    : state === "done"      ? "playback"
    : "idle";

  return (
    <>
      <FloatingNav />

      <main
        style={{
          minHeight: "100dvh",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* ── Hero stage ──────────────────────────────────────────────── */}
        <section
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "clamp(var(--space-24), 15vh, var(--space-32)) var(--space-6)",
            position: "relative",
            overflow: "hidden",
          }}
        >
          {/* Ambient glow behind the stage */}
          <div
            aria-hidden
            style={{
              position: "absolute",
              top: "30%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              width: "min(700px, 90vw)",
              height: "min(700px, 90vw)",
              borderRadius: "var(--radius-full)",
              background:
                "radial-gradient(ellipse, oklch(68% 0.22 255 / 0.06) 0%, transparent 70%)",
              pointerEvents: "none",
            }}
          />

          {/* Content column */}
          <div
            className="animate-reveal"
            style={{
              width: "100%",
              maxWidth: 640,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "var(--space-8)",
              position: "relative",
              zIndex: 1,
            }}
          >
            {/* Eyebrow */}
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "var(--text-xs)",
                fontWeight: 400,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: "var(--color-accent)",
                animationDelay: "0ms",
              }}
            >
              Voice Enrollment Demo
            </span>

            {/* Display headline */}
            <h1
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "clamp(2.5rem, 7vw, 4.5rem)",
                fontWeight: 300,
                fontStyle: "normal",
                lineHeight: 1.08,
                letterSpacing: "-0.025em",
                textAlign: "center",
                color: "var(--color-ink)",
                margin: 0,
                overflowWrap: "anywhere",
                minWidth: 0,
              }}
            >
              Your voice.
              <br />
              <span
                style={{
                  fontWeight: 600,
                  background:
                    "linear-gradient(135deg, var(--color-accent), oklch(72% 0.18 280))",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundClip: "text",
                }}
              >
                Echoed digitally.
              </span>
            </h1>

            {/* Sub-copy */}
            <p
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "var(--text-base)",
                fontWeight: 400,
                fontStyle: "normal",
                color: "var(--color-ink-3)",
                textAlign: "center",
                lineHeight: 1.65,
                maxWidth: 480,
                margin: 0,
              }}
            >
              Speak once. EchoSelf processes your voice through an on-device
              neural pipeline and synthesizes speech — entirely locally, nothing
              leaves your machine.
            </p>

            {/* ── Enrollment stage ──────────────────────────────────── */}
            <div
              style={{
                width: "100%",
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-5)",
              }}
            >
              {/* Waveform */}
              <WaveformVisualizer
                analyser={analyser}
                isActive={isActive}
                mode={waveformMode}
                height={80}
              />

              {/* Timer + status */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "var(--space-4)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "var(--text-xs)",
                  color: "var(--color-ink-4)",
                  minHeight: 20,
                }}
              >
                {state === "recording" && (
                  <>
                    <span
                      style={{
                        width: 6,
                        height: 6,
                        borderRadius: "var(--radius-full)",
                        background: "var(--color-error)",
                        animation: "pulse-ring 1.4s ease-out infinite",
                        display: "inline-block",
                      }}
                    />
                    <span style={{ color: "var(--color-ink-2)" }}>
                      {formatTime(recordingTime)}
                    </span>
                    <span>Recording — speak naturally</span>
                  </>
                )}
                {showProcessing && (
                  <span style={{ color: "var(--color-accent)" }}>
                    Processing through neural pipeline…
                  </span>
                )}
                {state === "idle" && (
                  <span>Press the button to start voice enrollment</span>
                )}
                {state === "done" && (
                  <span style={{ color: "var(--color-success)" }}>
                    Pipeline complete
                  </span>
                )}
                {state === "error" && (
                  <span style={{ color: "var(--color-error)" }}>
                    {result?.error ?? "An error occurred"}
                  </span>
                )}
              </div>

              {/* CTA button */}
              <div
                style={{ display: "flex", justifyContent: "center" }}
              >
                <RecordButton
                  state={state}
                  onStart={startRecording}
                  onStop={stopRecording}
                />
              </div>
            </div>

            {/* Processing indicator */}
            {showProcessing && (
              <div
                className="animate-fade-in"
                style={{
                  width: "100%",
                  display: "flex",
                  flexDirection: "column",
                  gap: "var(--space-3)",
                }}
              >
                <ProcessingSteps />
              </div>
            )}

            {/* Result card */}
            {showResult && result && (
              <div style={{ width: "100%" }}>
                <ResultCard
                  audioBase64={result.audioBase64}
                  sampleRate={result.sampleRate}
                  watermarkId={result.watermarkId}
                  detectionResult={result.detectionResult}
                  stageLtencies={result.stageLtencies}
                  totalLatencyMs={result.totalLatencyMs}
                  realTimeFactor={result.realTimeFactor}
                  isDemoMode={result.isDemoMode}
                  transcript={result.transcript}
                />
              </div>
            )}
          </div>
        </section>

        {/* ── Pipeline feature strip ───────────────────────────────────── */}
        <section
          style={{
            padding: "var(--space-20) var(--space-6)",
            borderTop: "var(--rule-hairline)",
          }}
        >
          <div
            style={{
              maxWidth: 960,
              margin: "0 auto",
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-12)",
            }}
          >
            <p
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "var(--text-xs)",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "var(--color-ink-4)",
                textAlign: "center",
              }}
            >
              Seven-stage on-device neural pipeline
            </p>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))",
                gap: "var(--space-4)",
              }}
            >
              {PIPELINE_STAGES.map((stage, i) => (
                <PipelineCard
                  key={stage.name}
                  index={i + 1}
                  {...stage}
                  isHovered={hoveredFeature === i}
                  onMouseEnter={() => setHoveredFeature(i)}
                  onMouseLeave={() => setHoveredFeature(null)}
                />
              ))}
            </div>
          </div>
        </section>

        {/* ── Footer ──────────────────────────────────────────────────── */}
        <footer
          style={{
            padding: "var(--space-8) var(--space-6)",
            borderTop: "var(--rule-hairline)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "var(--space-4)",
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "var(--text-xs)",
              color: "var(--color-ink-4)",
            }}
          >
            EchoSelf · Local-first voice AI · ₹0 cloud cost · All processing on-device
          </span>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "var(--text-xs)",
              color: "var(--color-ink-4)",
            }}
          >
            PyTorch · Next.js · Go · RTX 3050
          </span>
        </footer>
      </main>
    </>
  );
}

// ── Processing animation steps ─────────────────────────────────────────────

function ProcessingSteps() {
  const steps = [
    "Validating audio…",
    "Running VAD (voice detection)…",
    "Transcribing speech (ASR)…",
    "Synthesizing voice (NVS)…",
    "Embedding watermark…",
    "Verifying provenance…",
    "Logging audit event…",
  ];
  return (
    <div
      style={{
        padding: "var(--space-5)",
        background: "var(--color-paper-2)",
        border: "var(--rule-hairline)",
        borderRadius: "var(--radius-md)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-2)",
      }}
    >
      {steps.map((step, i) => (
        <div
          key={step}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--space-3)",
            opacity: 0,
            animation: `fade-in 0.3s ease ${i * 0.18}s both`,
          }}
        >
          <span
            style={{
              width: 5,
              height: 5,
              borderRadius: "var(--radius-full)",
              background: "var(--color-accent)",
              animation: `pulse-ring 1.6s ${i * 0.3}s ease-out infinite`,
              display: "inline-block",
              flexShrink: 0,
            }}
          />
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "var(--text-xs)",
              color: "var(--color-ink-3)",
            }}
          >
            {step}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Pipeline stage card ────────────────────────────────────────────────────

interface PipelineCardProps {
  index: number;
  name: string;
  description: string;
  badge: string;
  isHovered: boolean;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

function PipelineCard({
  index,
  name,
  description,
  badge,
  isHovered,
  onMouseEnter,
  onMouseLeave,
}: PipelineCardProps) {
  return (
    <div
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{
        padding: "var(--space-5)",
        background: isHovered ? "var(--color-paper-3)" : "var(--color-paper-2)",
        border: `1px solid ${isHovered ? "var(--color-accent-ring)" : "var(--color-border)"}`,
        borderRadius: "var(--radius-md)",
        transition: `background var(--dur-base) var(--ease-out), border-color var(--dur-base) var(--ease-out)`,
        cursor: "default",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "var(--space-3)",
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-xs)",
            color: isHovered ? "var(--color-accent)" : "var(--color-ink-4)",
            transition: `color var(--dur-base) var(--ease-out)`,
          }}
        >
          {String(index).padStart(2, "0")}
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-xs)",
            padding: "2px var(--space-2)",
            background: "var(--color-paper-4)",
            borderRadius: "var(--radius-sm)",
            color: "var(--color-ink-4)",
          }}
        >
          {badge}
        </span>
      </div>
      <p
        style={{
          fontFamily: "var(--font-display)",
          fontSize: "var(--text-base)",
          fontWeight: 500,
          fontStyle: "normal",
          color: "var(--color-ink)",
          marginBottom: "var(--space-2)",
          lineHeight: 1.3,
        }}
      >
        {name}
      </p>
      <p
        style={{
          fontFamily: "var(--font-display)",
          fontSize: "var(--text-sm)",
          fontWeight: 400,
          fontStyle: "normal",
          color: "var(--color-ink-3)",
          lineHeight: 1.6,
          margin: 0,
        }}
      >
        {description}
      </p>
    </div>
  );
}

// ── Data ───────────────────────────────────────────────────────────────────

const PIPELINE_STAGES = [
  {
    name: "Voice Activity Detection",
    description:
      "Energy-based VAD segments speech from silence. Runs at <0.5 ms, zero false positives on pure silence.",
    badge: "EnergyVAD",
  },
  {
    name: "Speech Recognition",
    description:
      "CTC-based ASR model transcribes your speech to text. Architecture is Whisper-compatible for future upgrade.",
    badge: "ASR",
  },
  {
    name: "Neural Voice Synthesis",
    description:
      "Acoustic model predicts mel-spectrogram frames. Griffin-Lim vocoder converts them to audio. ~17 ms per utterance.",
    badge: "NVS",
  },
  {
    name: "Speaker Verification",
    description:
      "192-dim d-vector embeddings from a convolutional encoder. Cosine similarity matching against enrolled profile.",
    badge: "d-vector",
  },
  {
    name: "Watermark Embedding",
    description:
      "Spread-spectrum watermark embedded at SNR = 35 dB. Imperceptible. 100% detection rate on clean audio.",
    badge: "SS-WM",
  },
  {
    name: "Watermark Detection",
    description:
      "Independent detector verifies the embed succeeded. Normalized cross-correlation, <0.5 ms per check.",
    badge: "Detector",
  },
  {
    name: "Audit Logging",
    description:
      "Append-only tamper-evident log. SHA-256 output hash, watermark ID, and event ID written per synthesis.",
    badge: "JSONL",
  },
];

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, "0");
  const s = (seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}
