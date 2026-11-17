"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type RecordState = "idle" | "recording" | "processing" | "done" | "error";

export interface EchoResult {
  audioBase64: string | null;
  sampleRate: number;
  watermarkId: string | null;
  detectionResult: boolean | null;
  stageLtencies: Record<string, number> | null;
  totalLatencyMs: number | null;
  realTimeFactor: number | null;
  isDemoMode: boolean;
  transcript: string | null;
  error: string | null;
}

export function useEchoRecorder() {
  const [state, setState] = useState<RecordState>("idle");
  const [result, setResult] = useState<EchoResult | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      cleanupAudio();
      if (timerRef.current) clearInterval(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function cleanupAudio() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  }

  // ── Pick the best supported mimeType ──────────────────────────────────────
  function getSupportedMimeType(): string {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
      "audio/mp4",
    ];
    for (const type of candidates) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return ""; // let the browser pick
  }

  // ── Process collected chunks → call API ───────────────────────────────────
  async function processChunks(chunks: Blob[], mimeType: string) {
    if (chunks.length === 0) {
      setState("error");
      setResult({
        audioBase64: null, sampleRate: 16000,
        watermarkId: null, detectionResult: null,
        stageLtencies: null, totalLatencyMs: null,
        realTimeFactor: null, isDemoMode: false,
        transcript: null,
        error: "No audio was captured. Check microphone permissions.",
      });
      return;
    }

    const blob = new Blob(chunks, { type: mimeType || "audio/webm" });

    try {
      const arrayBuf = await blob.arrayBuffer();
      const uint8 = new Uint8Array(arrayBuf);

      // btoa on large buffers via fromCharCode can stack-overflow — use chunked approach
      let binary = "";
      const CHUNK = 8192;
      for (let i = 0; i < uint8.length; i += CHUNK) {
        binary += String.fromCharCode(...uint8.subarray(i, i + CHUNK));
      }
      const audioB64 = btoa(binary);

      const res = await fetch("/api/echoself", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_type: "voice_pipeline",
          audio_base64: audioB64,
          text_input: "Echo self — your voice and identity, echoed digitally.",
          user_id: "demo-user",
        }),
      });

      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      const output = data.output ?? data;

      setResult({
        audioBase64: output.audio_base64 ?? null,
        sampleRate: output.audio_sample_rate ?? 16000,
        watermarkId: output.watermark_id ?? null,
        detectionResult: output.detection_result ?? null,
        stageLtencies: output.stage_latencies ?? null,
        totalLatencyMs: output.total_latency_ms ?? null,
        realTimeFactor: output.real_time_factor ?? null,
        isDemoMode: data.demo ?? false,
        transcript: output.generated_text ?? output.transcript ?? null,
        error: null,
      });
      setState("done");
    } catch (err) {
      console.error("[EchoSelf] Processing error:", err);
      setState("error");
      setResult({
        audioBase64: null, sampleRate: 16000,
        watermarkId: null, detectionResult: null,
        stageLtencies: null, totalLatencyMs: null,
        realTimeFactor: null, isDemoMode: false,
        transcript: null,
        error: `Processing failed: ${err instanceof Error ? err.message : String(err)}`,
      });
    }
  }

  // ── Start ──────────────────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    setResult(null);
    setRecordingTime(0);
    chunksRef.current = [];

    // Check API — works on http://localhost, not on file:// or cross-origin
    if (typeof window === "undefined" || !navigator?.mediaDevices?.getUserMedia) {
      setState("error");
      setResult({
        audioBase64: null, sampleRate: 16000,
        watermarkId: null, detectionResult: null,
        stageLtencies: null, totalLatencyMs: null,
        realTimeFactor: null, isDemoMode: false,
        transcript: null,
        error: "Microphone API not available. Open the page on http://localhost:3000 (not an IP address or file://).",
      });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
      streamRef.current = stream;

      // Web Audio analyser for waveform
      const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const an = audioCtx.createAnalyser();
      an.fftSize = 2048;
      an.smoothingTimeConstant = 0.8;
      source.connect(an);
      setAnalyser(an);

      const mimeType = getSupportedMimeType();
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      // onstop fires AFTER the last ondataavailable — safe to process here
      mr.onstop = () => {
        // Stop stream tracks AFTER onstop (not before) so the last chunk arrives
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
        if (audioCtxRef.current) {
          audioCtxRef.current.close().catch(() => {});
          audioCtxRef.current = null;
        }
        processChunks(chunksRef.current, mimeType);
      };

      mr.onerror = (e) => {
        console.error("[EchoSelf] MediaRecorder error:", e);
        setState("error");
        setResult({
          audioBase64: null, sampleRate: 16000,
          watermarkId: null, detectionResult: null,
          stageLtencies: null, totalLatencyMs: null,
          realTimeFactor: null, isDemoMode: false,
          transcript: null,
          error: "Recording error. Try a different browser.",
        });
      };

      mr.start(250); // collect in 250ms chunks for reliability
      mediaRecorderRef.current = mr;
      setState("recording");

      timerRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch (err) {
      console.error("[EchoSelf] getUserMedia failed:", err);
      const msg =
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Microphone permission denied. Click the lock icon in your browser address bar and allow microphone access."
          : err instanceof DOMException && err.name === "NotFoundError"
          ? "No microphone found. Connect a microphone and try again."
          : `Microphone error: ${err instanceof Error ? err.message : String(err)}`;
      setState("error");
      setResult({
        audioBase64: null, sampleRate: 16000,
        watermarkId: null, detectionResult: null,
        stageLtencies: null, totalLatencyMs: null,
        realTimeFactor: null, isDemoMode: false,
        transcript: null, error: msg,
      });
    }
  }, []);

  // ── Stop ───────────────────────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setAnalyser(null);
    setState("processing");

    // Stop the MediaRecorder — this triggers onstop after the last chunk arrives
    // Do NOT stop the stream here; onstop handler does it after data is collected
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const reset = useCallback(() => {
    setState("idle");
    setResult(null);
    setRecordingTime(0);
    setAnalyser(null);
    chunksRef.current = [];
  }, []);

  return { state, result, analyser, recordingTime, startRecording, stopRecording, reset };
}
