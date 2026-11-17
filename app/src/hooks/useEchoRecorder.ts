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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAudio();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  function stopAudio() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
  }

  const startRecording = useCallback(async () => {
    setResult(null);
    setRecordingTime(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up Web Audio API analyser
      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const an = audioCtx.createAnalyser();
      an.fftSize = 2048;
      source.connect(an);
      setAnalyser(an);

      // MediaRecorder
      const mr = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => handleRecordingStop();

      mr.start(100); // collect in 100ms chunks
      mediaRecorderRef.current = mr;

      setState("recording");

      // Recording timer
      timerRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch (err) {
      console.error("Failed to start recording:", err);
      setState("error");
      setResult({
        audioBase64: null,
        sampleRate: 16000,
        watermarkId: null,
        detectionResult: null,
        stageLtencies: null,
        totalLatencyMs: null,
        realTimeFactor: null,
        isDemoMode: false,
        transcript: null,
        error: "Microphone access denied. Please allow microphone access and try again.",
      });
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setAnalyser(null);
    stopAudio();
    setState("processing");
  }, []);

  async function handleRecordingStop() {
    const blob = new Blob(chunksRef.current, { type: "audio/webm" });

    try {
      // Convert blob to base64 (we send the raw audio as base64 to our API)
      const arrayBuf = await blob.arrayBuffer();
      const uint8 = new Uint8Array(arrayBuf);
      let binary = "";
      uint8.forEach((b) => (binary += String.fromCharCode(b)));
      const audioB64 = btoa(binary);

      // Call the Next.js API route which proxies to Python
      const res = await fetch("/api/echoself", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_type: "voice_pipeline",
          audio_base64: audioB64,
          text_input: "Echo self, your voice and identity, echoed digitally.",
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
      console.error("Processing error:", err);
      setState("error");
      setResult({
        audioBase64: null,
        sampleRate: 16000,
        watermarkId: null,
        detectionResult: null,
        stageLtencies: null,
        totalLatencyMs: null,
        realTimeFactor: null,
        isDemoMode: false,
        transcript: null,
        error: "Processing failed. Check the console for details.",
      });
    }
  }

  const reset = useCallback(() => {
    setState("idle");
    setResult(null);
    setRecordingTime(0);
    setAnalyser(null);
  }, []);

  return {
    state,
    result,
    analyser,
    recordingTime,
    startRecording,
    stopRecording,
    reset,
  };
}
