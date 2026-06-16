#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildChallengeRecord,
  DEFAULT_PURSUIT_CHALLENGE,
  trajectoryPlotSvg,
  validateAttemptPacket,
} from "../src/competition/arcade-engine.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

function main(argv) {
  const args = parseArgs(argv);
  if (!args.attempt) {
    printUsage();
    process.exitCode = 2;
    return;
  }
  const attemptPath = resolve(process.cwd(), args.attempt);
  const challengeConfig = args.challenge
    ? JSON.parse(readFileSync(resolve(process.cwd(), args.challenge), "utf8"))
    : DEFAULT_PURSUIT_CHALLENGE;
  const attempt = JSON.parse(readFileSync(attemptPath, "utf8"));
  const challengeRecord = buildChallengeRecord(challengeConfig);
  const validation = validateAttemptPacket(attempt, challengeRecord, {
    sample_stride_ticks: Number(args.sampleStride || 1),
  });
  const payload = {
    status: validation.status,
    errors: validation.errors,
    warnings: validation.warnings,
    canonical_score: validation.canonical_score,
    canonical_metrics: validation.canonical_metrics,
  };
  console.log(JSON.stringify(payload, null, 2));
  if (validation.replay && args.plotDir) {
    const outputDir = resolve(process.cwd(), args.plotDir);
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(resolve(outputDir, "validated-ri.svg"), trajectoryPlotSvg(validation.replay, "RI"), "utf8");
    writeFileSync(resolve(outputDir, "validated-rc.svg"), trajectoryPlotSvg(validation.replay, "RC"), "utf8");
  }
  if (validation.status === "invalid") process.exitCode = 1;
}

function parseArgs(argv) {
  const parsed = {};
  for (let idx = 0; idx < argv.length; idx += 1) {
    const arg = argv[idx];
    if (arg === "--attempt") parsed.attempt = argv[++idx];
    else if (arg === "--challenge") parsed.challenge = argv[++idx];
    else if (arg === "--plot-dir") parsed.plotDir = argv[++idx];
    else if (arg === "--sample-stride") parsed.sampleStride = argv[++idx];
    else if (arg === "--help" || arg === "-h") parsed.help = true;
  }
  return parsed;
}

function printUsage() {
  const rel = resolve(__dirname, "../fixtures/sample-valid-attempt.json");
  console.log(`Usage:
  node web/rpo-trainer-preview/tools/validate-attempt.mjs --attempt ${rel}

Options:
  --challenge <path>       Challenge config JSON. Defaults to built-in local pursuit challenge.
  --plot-dir <path>        Write validated-ri.svg and validated-rc.svg into an existing directory.
  --sample-stride <ticks>  Replay history sample stride for plot generation.
`);
}

main(process.argv.slice(2));
