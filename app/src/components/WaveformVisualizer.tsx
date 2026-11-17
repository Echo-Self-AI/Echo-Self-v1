"use client";

import { useEffect, useRef } from "react";

interface WaveformVisualizerProps {
  analyser: AnalyserNode | null;
  isActive: boolean;
  /** "recording" pulses red, "playback" pulses blue, "idle" is dim */
  mode: "recording" | "playback" | "idle";
  height?: number;
}

export default function WaveformVisualizer({
  analyser,
  isActive,
  mode,
  height = 80,
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  const accentColor =
    mode === "recording" ? "oklch(65% 0.22 25)"
    : mode === "playback"  ? "oklch(68% 0.22 255)"
    : "oklch(38% 0.010 240)";

  const glowColor =
    mode === "recording" ? "oklch(65% 0.22 25 / 0.25)"
    : mode === "playback"  ? "oklch(68% 0.22 255 / 0.25)"
    : "transparent";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = analyser?.fftSize ?? 256;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      if (!canvas || !ctx) return;

      // Handle device pixel ratio
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
      }

      const W = rect.width;
      const H = rect.height;

      ctx.clearRect(0, 0, W, H);

      if (!analyser || !isActive) {
        // Draw flat idle line
        drawIdleLine(ctx, W, H, mode === "idle" ? "oklch(22% 0.012 245)" : accentColor);
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      analyser.getByteTimeDomainData(dataArray);

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = accentColor;
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 10;

      ctx.beginPath();
      const sliceWidth = W / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * H) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();

      rafRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [analyser, isActive, mode, accentColor, glowColor]);

  return (
    <div
      style={{
        width: "100%",
        height,
        position: "relative",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
        background: "var(--color-paper-2)",
        border: `1px solid ${mode !== "idle" ? accentColor + "40" : "var(--color-border)"}`,
        transition: `border-color ${process.env.NODE_ENV ? "0.3s" : "0.3s"} ease`,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
    </div>
  );
}

function drawIdleLine(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  color: string
) {
  ctx.lineWidth = 1;
  ctx.strokeStyle = color;
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();
}
