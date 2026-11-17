import { NextRequest, NextResponse } from "next/server";

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL ?? "http://127.0.0.1:8000";

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
 * Body (JSON):
 *   { job_type: "voice_synthesis" | "llm_generate" | "voice_pipeline",
 *     audio_base64?: string,   // base64 WAV — for voice_pipeline
 *     text_input?: string,     // for voice_synthesis / voice_pipeline
 *     user_id?: string }
 *
 * Routes to Python service POST /process, or returns a demo result
 * when the service is offline. The response always carries
 * `{ demo: true }` when falling back to demo mode.
 */
export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const { job_type, audio_base64, text_input, user_id } = body as {
    job_type?: string;
    audio_base64?: string;
    text_input?: string;
    user_id?: string;
  };

  if (!job_type) {
    return NextResponse.json({ error: "job_type required" }, { status: 400 });
  }

  // ── Build the payload for the Python service ─────────────────────────────
  let payload: Record<string, unknown> = {};
  if (job_type === "voice_synthesis" || job_type === "voice_pipeline") {
    payload = {
      text: text_input ?? "Hello, this is EchoSelf.",
      speaker_id: null,
    };
  } else if (job_type === "llm_generate") {
    payload = { prompt: text_input ?? "", max_new_tokens: 20 };
  }

  const serviceJob = job_type === "voice_pipeline" ? "voice_synthesis" : job_type;

  // ── Try the live Python service ───────────────────────────────────────────
  try {
    const serviceRes = await fetch(`${PYTHON_SERVICE_URL}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: `demo-${Date.now()}`,
        job_type: serviceJob,
        user_id: user_id ?? "demo-user",
        payload,
      }),
      signal: AbortSignal.timeout(30_000),
    });

    if (!serviceRes.ok) throw new Error(`Service returned HTTP ${serviceRes.status}`);
    const data = await serviceRes.json();
    return NextResponse.json({ ...data, demo: false });
  } catch (err) {
    console.warn("[EchoSelf API] Python service unavailable — returning demo result.", err);
  }

  // ── Demo fallback (Python service offline) ────────────────────────────────
  return NextResponse.json(demoResult(job_type, text_input), { status: 200 });
}

/** Returns a clearly-labelled demo result when the AI service is offline. */
function demoResult(job_type: string, text_input?: string): Record<string, unknown> {
  const base = { demo: true, service_status: "offline" };

  if (job_type === "voice_synthesis" || job_type === "voice_pipeline") {
    // A minimal valid base64 WAV (silent, ~0.1s) so the audio player has something
    const silentWav = makeSilentWavB64(16000, 0.5);
    return {
      ...base,
      status: "success",
      output: {
        audio_base64: silentWav,
        duration_seconds: 0.5,
        watermark_id: "demo-watermark-id",
        detection_result: true,
        stage_latencies: {
          vad: 0.2,
          asr: 0.1,
          nvs: 14.0,
          watermark: 0.3,
          detect: 0.4,
          audit: 0.5,
        },
        total_latency_ms: 15.5,
        real_time_factor: 0.009,
      },
      message:
        "EchoSelf AI service is offline. This is a frontend demonstration — no real voice synthesis was performed.",
    };
  }

  return {
    ...base,
    status: "success",
    output: { generated_text: text_input ?? "Demo response." },
    message: "Demo mode — AI service offline.",
  };
}

/** Build a minimal valid silent 16-bit PCM WAV as a base64 string. */
function makeSilentWavB64(sampleRate: number, durationSec: number): string {
  const numSamples = Math.floor(sampleRate * durationSec);
  const dataSize = numSamples * 2; // 16-bit = 2 bytes per sample
  const buf = Buffer.alloc(44 + dataSize);

  // RIFF header
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8);
  // fmt chunk
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);  // PCM
  buf.writeUInt16LE(1, 22);  // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buf.writeUInt16LE(2, 32);  // block align
  buf.writeUInt16LE(16, 34); // bits per sample
  // data chunk
  buf.write("data", 36);
  buf.writeUInt32LE(dataSize, 40);
  // samples are already 0 (silent)

  return buf.toString("base64");
}
