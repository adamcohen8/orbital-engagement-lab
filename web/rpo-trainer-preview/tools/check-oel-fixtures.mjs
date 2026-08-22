#!/usr/bin/env node
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const previewRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(previewRoot, "../..");
const generator = resolve(previewRoot, "tools/generate-oel-contract-fixtures.py");
const candidates = [
  process.env.OEL_PYTHON,
  process.env.VIRTUAL_ENV ? resolve(process.env.VIRTUAL_ENV, "bin/python") : null,
  resolve(repoRoot, ".venv/bin/python"),
  resolve(repoRoot, ".venv/Scripts/python.exe"),
  "python3",
  "python",
].filter(Boolean);

function supportedPython(candidate) {
  if (candidate.includes("/") && !existsSync(candidate)) return false;
  const probe = spawnSync(candidate, ["--version"], { encoding: "utf8" });
  if (probe.status !== 0) return false;
  const match = `${probe.stdout || ""}${probe.stderr || ""}`.match(/Python\s+(\d+)\.(\d+)/);
  return Boolean(match && (Number(match[1]) > 3 || (Number(match[1]) === 3 && Number(match[2]) >= 10)));
}

const python = candidates.find(supportedPython);
if (!python) {
  console.error("No OEL-compatible Python 3.10+ interpreter was found. Activate the OEL environment or set OEL_PYTHON.");
  process.exit(1);
}

const result = spawnSync(python, [generator, "--check"], {
  cwd: previewRoot,
  stdio: "inherit",
});
process.exit(result.status ?? 1);
