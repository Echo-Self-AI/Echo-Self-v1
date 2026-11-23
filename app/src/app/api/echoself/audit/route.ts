import { NextResponse } from "next/server";
import { readFileSync } from "fs";
import { join } from "path";

const AUDIT_LOG = join(process.cwd(), "..", "ai", "data", "audit", "audit_log.jsonl");

export async function GET() {
  try {
    const text = readFileSync(AUDIT_LOG, "utf-8");
    const records = text
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        try { return JSON.parse(line); } catch { return null; }
      })
      .filter(Boolean);
    return NextResponse.json({ records, count: records.length });
  } catch {
    // File not found or not yet created
    return NextResponse.json({ records: [], count: 0 });
  }
}
