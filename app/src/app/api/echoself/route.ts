import { NextRequest, NextResponse } from "next/server";

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL ?? "http://127.0.0.1:8000";
const TIMEOUT_MS = 60_000;

/** Health-check the Python AI service. */
export async function GET() {
  try {
    const res = await fetch(`${PYTHON_SERVICE_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return NextResponse.json({ online: true, ...data });
  } catch {
    return NextResponse.json({ online: false }, { status: 503 });
  }
}

/**
 * POST /api/echoself
 *
 * Supported job_types:
 *   consent_register  — register consent for demo user (must be called first)
 *   voice_synthesis   — synthesize speech from text, returns audio_base64 WAV
 *   voice_embedding   — extract speaker embedding from audio_base64
 *   avatar_embedding  — extract visual embedding from image_base64
 *   avatar_generate   — generate avatar PNG from embedding, returns image_base64
 *   video_generate    — generate avatar GIF, returns video_gif_base64
 */
export async function POST(req: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { job_type, user_id = "demo-user", ...rest } = body as {
    job_type?: string;
    user_id?: string;
    [key: string]: unknown;
  };

  if (!job_type) {
    return NextResponse.json({ error: "job_type required" }, { status: 400 });
  }

  // Build payload for the Python service based on job type
  const payload = buildPayload(job_type, rest);

  try {
    const serviceRes = await fetch(`${PYTHON_SERVICE_URL}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: `demo-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        job_type,
        user_id,
        payload,
      }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });

    const data = await serviceRes.json();

    // Validate voice output has actual audio
    if (job_type === "voice_synthesis" && data.status === "success") {
      const ab = data.output?.audio_base64 as string | undefined;
      if (!ab || ab.length < 100) {
        return NextResponse.json({
          ...data,
          status: "failed",
          error: "Voice synthesis produced empty audio.",
          demo: false,
        });
      }
      // Decode and check duration > 0
      const wavBytes = Buffer.from(ab, "base64");
      if (wavBytes.length < 44) {
        return NextResponse.json({
          ...data,
          status: "failed",
          error: "Voice synthesis produced a malformed WAV file.",
          demo: false,
        });
      }
      const numSamples = wavBytes.readUInt32LE(40) / 2;
      const sampleRate = wavBytes.readUInt32LE(24);
      const durationSec = sampleRate > 0 ? numSamples / sampleRate : 0;
      if (durationSec < 0.05) {
        return NextResponse.json({
          ...data,
          status: "failed",
          error: `Voice synthesis produced near-silent audio (${durationSec.toFixed(3)}s).`,
          demo: false,
        });
      }
    }

    return NextResponse.json({ ...data, demo: false });
  } catch (err) {
    const isTimeout = err instanceof Error && err.name === "TimeoutError";
    const msg = isTimeout
      ? "AI service timed out. Try a shorter text."
      : "AI service unavailable. Start the Python service and try again.";

    // For voice_synthesis offline, return a clearly labelled demo WAV
    if (job_type === "voice_synthesis") {
      return NextResponse.json({
        status: "success",
        demo: true,
        service_status: "offline",
        message: msg,
        output: {
          audio_base64: makeSilentWavB64(16000, 1.0),
          audio_sample_rate: 16000,
          duration_seconds: 1.0,
          watermark_id: null,
          detection_result: null,
          stage_latencies: {},
          total_latency_ms: 0,
          real_time_factor: 0,
        },
      });
    }

    return NextResponse.json({
      status: "failed",
      demo: true,
      service_status: "offline",
      error: msg,
      output: {},
    });
  }
}

function buildPayload(job_type: string, rest: Record<string, unknown>): Record<string, unknown> {
  switch (job_type) {
    case "consent_register":
      return {
        modality: rest.modality ?? "all",
        speaker_id: rest.speaker_id ?? "demo-speaker",
        avatar_id: rest.avatar_id ?? "demo-avatar",
        scope: rest.scope ?? "all",
      };
    case "voice_synthesis":
      return {
        text: rest.text_input ?? rest.text ?? "EchoSelf — your voice and identity, echoed digitally.",
        speaker_id: rest.speaker_id ?? null,
      };
    case "voice_embedding":
      return {
        speaker_id: rest.speaker_id ?? "demo-speaker",
        audio_base64: rest.audio_base64 ?? null,
      };
    case "portrait_process":
      return {
        image_base64: rest.image_base64 ?? null,
        avatar_id: rest.avatar_id ?? "primary",
      };
    case "portrait_load":
      return {
        avatar_id: rest.avatar_id ?? "primary",
      };
    case "talking_head_video":
      return {
        avatar_id: rest.avatar_id ?? "primary",
        portrait_base64: rest.portrait_base64 ?? null,
        audio_base64: rest.audio_base64 ?? null,
        text: rest.text ?? rest.text_input ?? null,
        speaker_id: rest.speaker_id ?? null,
        fps: rest.fps ?? 25,
        width: rest.width ?? 512,
        height: rest.height ?? 512,
      };
    case "avatar_embedding":
      return {
        avatar_id: rest.avatar_id ?? "demo-avatar",
        image_base64: rest.image_base64 ?? null,
      };
    case "avatar_generate":
      return {
        avatar_id: rest.avatar_id ?? "demo-avatar",
        embedding: rest.embedding ?? null,
      };
    case "video_generate":
      return {
        prompt: rest.prompt ?? "avatar speaking",
        avatar_id: rest.avatar_id ?? null,
        num_frames: rest.num_frames ?? 8,
        fps: rest.fps ?? 8,
      };
    default:
      return (rest.payload as Record<string, unknown>) ?? {};
  }
}

function makeSilentWavB64(sampleRate: number, durationSec: number): string {
  const numSamples = Math.floor(sampleRate * durationSec);
  const dataSize = numSamples * 2;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(dataSize, 40);
  return buf.toString("base64");
}
