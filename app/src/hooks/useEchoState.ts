"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type AppStage =
  | "welcome"
  | "recording"
  | "preview"
  | "processing"
  | "avatar_upload"
  | "avatar_processing"
  | "dashboard"
  | "error";

export interface VoiceResult {
  audioBase64: string;
  durationSec: number;
  sampleRate: number;
  watermarkId: string | null;
  detectionResult: boolean | null;
  stageLtencies: Record<string, number>;
  totalLatencyMs: number;
  realTimeFactor: number;
  isDemo: boolean;
}

export interface AvatarResult {
  imageBase64: string;
  gifBase64: string | null;
  width: number;
  height: number;
  isDemo: boolean;
}

export interface EchoState {
  stage: AppStage;
  error: string | null;

  // Recording
  analyser: AnalyserNode | null;
  recordingTimeSec: number;
  recordingBlobUrl: string | null;   // for playback before submission
  recordingBlob: Blob | null;

  // Results
  voiceResult: VoiceResult | null;
  avatarResult: AvatarResult | null;

  // Actions
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  submitRecording: () => Promise<void>;
  discardRecording: () => void;
  submitAvatar: (file: File) => Promise<void>;
  skipAvatar: () => void;
  reset: () => void;
}

function getSupportedMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) ?? "";
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  const uint8 = new Uint8Array(buf);
  let binary = "";
  const CHUNK = 8192;
  for (let i = 0; i < uint8.length; i += CHUNK) {
    binary += String.fromCharCode(...uint8.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

async function fileToBase64(file: File): Promise<string> {
  return blobToBase64(file);
}

async function callApi(job_type: string, extra: Record<string, unknown>): Promise<Record<string, unknown>> {
  const res = await fetch("/api/echoself", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_type, user_id: "demo-user", ...extra }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export function useEchoState(): EchoState {
  const [stage, setStage] = useState<AppStage>("welcome");
  const [error, setError] = useState<string | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [recordingTimeSec, setRecordingTimeSec] = useState(0);
  const [recordingBlobUrl, setRecordingBlobUrl] = useState<string | null>(null);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
  const [voiceResult, setVoiceResult] = useState<VoiceResult | null>(null);
  const [avatarResult, setAvatarResult] = useState<AvatarResult | null>(null);

  const mrRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mimeTypeRef = useRef("");

  useEffect(() => {
    return () => {
      cleanupStream();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function cleanupStream() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
  }

  const startRecording = useCallback(async () => {
    setError(null);
    setRecordingTimeSec(0);
    setRecordingBlobUrl(null);
    setRecordingBlob(null);
    chunksRef.current = [];

    if (!navigator?.mediaDevices?.getUserMedia) {
      setStage("error");
      setError("Microphone API unavailable. Open the page at http://localhost:3000.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;

      const AudioCtx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx();
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const an = ctx.createAnalyser();
      an.fftSize = 2048;
      an.smoothingTimeConstant = 0.75;
      src.connect(an);
      setAnalyser(an);

      const mimeType = getSupportedMimeType();
      mimeTypeRef.current = mimeType;
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      mr.ondataavailable = (e) => {
        if (e.data?.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        // Stop stream AFTER onstop so last chunk arrives
        cleanupStream();

        const chunks = chunksRef.current;
        if (!chunks.length) {
          setStage("error");
          setError("No audio captured. Please check microphone permissions.");
          return;
        }

        const blob = new Blob(chunks, { type: mimeTypeRef.current || "audio/webm" });
        const url = URL.createObjectURL(blob);
        setRecordingBlob(blob);
        setRecordingBlobUrl(url);
        setStage("preview");
      };

      mr.onerror = () => {
        setStage("error");
        setError("Recording error. Try a different browser.");
      };

      mr.start(250);
      mrRef.current = mr;
      setStage("recording");

      timerRef.current = setInterval(() => setRecordingTimeSec((t) => t + 1), 1000);
    } catch (err) {
      const e = err as DOMException;
      const msg =
        e.name === "NotAllowedError" ? "Microphone permission denied. Click the lock icon in the address bar and allow microphone access."
        : e.name === "NotFoundError"  ? "No microphone found. Connect a microphone and reload."
        : `Microphone error: ${e.message}`;
      setStage("error");
      setError(msg);
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setAnalyser(null);
    if (mrRef.current && mrRef.current.state !== "inactive") {
      mrRef.current.stop(); // triggers onstop → preview stage
    }
  }, []);

  const submitRecording = useCallback(async () => {
    if (!recordingBlob) {
      setStage("error");
      setError("No recording to submit.");
      return;
    }
    setStage("processing");
    setError(null);

    try {
      // 1. Register consent (required by the Python service before any synthesis)
      await callApi("consent_register", {
        modality: "all",
        speaker_id: "demo-speaker",
        avatar_id: "demo-avatar",
        scope: "all",
      });

      // 2. Extract speaker embedding from the actual recording
      const audioB64 = await blobToBase64(recordingBlob);
      const embRes = await callApi("voice_embedding", {
        speaker_id: "demo-speaker",
        audio_base64: audioB64,
      });

      // 3. Synthesize voice
      const synthText = "EchoSelf — your voice and identity, echoed digitally. This is your enrolled voice speaking.";
      const synthRes = await callApi("voice_synthesis", {
        text_input: synthText,
        speaker_id: (embRes as { output?: { speaker_id?: string } }).output?.speaker_id ?? "demo-speaker",
      });

      const sr = synthRes as {
        status: string;
        demo?: boolean;
        error?: string;
        output?: {
          audio_base64?: string;
          duration_seconds?: number;
          sample_rate?: number;
          watermark_id?: string;
          detection_result?: boolean;
          stage_latencies?: Record<string, number>;
          total_latency_ms?: number;
          real_time_factor?: number;
        };
      };

      if (sr.status !== "success" || !sr.output?.audio_base64) {
        throw new Error(sr.error ?? "Voice synthesis failed — no audio returned.");
      }

      // Validate the WAV
      const wavBytes = Uint8Array.from(atob(sr.output.audio_base64), (c) => c.charCodeAt(0));
      if (wavBytes.length < 44 || String.fromCharCode(...wavBytes.slice(0, 4)) !== "RIFF") {
        throw new Error("Voice synthesis returned a malformed audio file.");
      }

      setVoiceResult({
        audioBase64: sr.output.audio_base64,
        durationSec: sr.output.duration_seconds ?? 0,
        sampleRate: sr.output.sample_rate ?? 16000,
        watermarkId: sr.output.watermark_id ?? null,
        detectionResult: sr.output.detection_result ?? null,
        stageLtencies: sr.output.stage_latencies ?? {},
        totalLatencyMs: sr.output.total_latency_ms ?? 0,
        realTimeFactor: sr.output.real_time_factor ?? 0,
        isDemo: sr.demo ?? false,
      });

      setStage("avatar_upload");
    } catch (err) {
      setStage("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [recordingBlob]);

  const discardRecording = useCallback(() => {
    if (recordingBlobUrl) URL.revokeObjectURL(recordingBlobUrl);
    setRecordingBlob(null);
    setRecordingBlobUrl(null);
    setRecordingTimeSec(0);
    setStage("recording");
    startRecording();
  }, [recordingBlobUrl, startRecording]);

  const submitAvatar = useCallback(async (file: File) => {
    setStage("avatar_processing");
    setError(null);

    try {
      const imageB64 = await fileToBase64(file);

      // Extract visual embedding
      const embRes = await callApi("avatar_embedding", {
        avatar_id: "demo-avatar",
        image_base64: imageB64,
      });

      const embedding = (embRes as { output?: { embedding?: number[] } }).output?.embedding ?? null;

      // Generate avatar
      const genRes = await callApi("avatar_generate", {
        avatar_id: "demo-avatar",
        embedding,
      });

      const gr = genRes as {
        status: string;
        demo?: boolean;
        error?: string;
        output?: { image_base64?: string; width?: number; height?: number };
      };

      if (gr.status !== "success" || !gr.output?.image_base64) {
        throw new Error(gr.error ?? "Avatar generation failed — no image returned.");
      }

      // Generate short video GIF
      let gifBase64: string | null = null;
      try {
        const vidRes = await callApi("video_generate", {
          prompt: "avatar speaking",
          avatar_id: "demo-avatar",
          num_frames: 6,
          fps: 6,
        }) as { status?: string; output?: { video_gif_base64?: string } };
        if (vidRes.status === "success") {
          gifBase64 = vidRes.output?.video_gif_base64 ?? null;
        }
      } catch {
        // Video optional — don't fail the whole flow
      }

      setAvatarResult({
        imageBase64: gr.output.image_base64,
        gifBase64,
        width: gr.output.width ?? 64,
        height: gr.output.height ?? 64,
        isDemo: gr.demo ?? false,
      });

      setStage("dashboard");
    } catch (err) {
      setStage("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const skipAvatar = useCallback(() => {
    setAvatarResult(null);
    setStage("dashboard");
  }, []);

  const reset = useCallback(() => {
    if (recordingBlobUrl) URL.revokeObjectURL(recordingBlobUrl);
    setStage("welcome");
    setError(null);
    setAnalyser(null);
    setRecordingTimeSec(0);
    setRecordingBlobUrl(null);
    setRecordingBlob(null);
    setVoiceResult(null);
    setAvatarResult(null);
    chunksRef.current = [];
  }, [recordingBlobUrl]);

  return {
    stage, error, analyser, recordingTimeSec,
    recordingBlobUrl, recordingBlob,
    voiceResult, avatarResult,
    startRecording, stopRecording, submitRecording,
    discardRecording, submitAvatar, skipAvatar, reset,
  };
}
