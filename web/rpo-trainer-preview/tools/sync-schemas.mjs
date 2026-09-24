import { readFileSync, readdirSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const outputRoot = join(repositoryRoot, "web/rpo-trainer-preview/schemas");
const schemaBase = "https://orbital-engineering-lab.vercel.app/schemas/";
const sources = [
  "docs/contracts/schemas/oel-hosted-execution-package-v1.schema.json",
  "docs/contracts/schemas/oel-hosted-pro-package-v1.schema.json",
  "docs/contracts/schemas/oel-orbit-lifetime-v1.schema.json",
  "docs/contracts/schemas/oel-spacecraft-power-v1.schema.json",
  "docs/contracts/schemas/oel-study-lifecycle-v1.schema.json",
  "docs/contracts/schemas/oel-workflow-evidence-v1.schema.json",
  "sim/execution/run_lifecycle/schemas/await_result.schema.json",
  "sim/execution/run_lifecycle/schemas/execution_owner.schema.json",
  "sim/execution/run_lifecycle/schemas/reconcile_result.schema.json",
  "sim/execution/run_lifecycle/schemas/run_event.schema.json",
  "sim/execution/run_lifecycle/schemas/run_handle.schema.json",
  "sim/execution/run_lifecycle/schemas/run_manifest.schema.json",
  "sim/execution/run_lifecycle/schemas/run_state.schema.json",
  "sim/interchange/schemas/oel-completed-run-snapshot-v1.schema.json",
  "sim/interchange/schemas/oel-completed-run-state-v1.schema.json",
  "sim/interchange/schemas/oel-handoff-manifest-v1.schema.json",
  "sim/interchange/schemas/oel-maneuver-detection-v1.schema.json",
  "sim/interchange/schemas/oel-ogp-mean-element-product-v1.schema.json",
  "sim/interchange/schemas/oel-product-envelope-v1.schema.json",
  "sim/interchange/schemas/oel-relative-state-estimate-v1.schema.json",
  "sim/interchange/schemas/oel-satellite-checkpoint-v1.schema.json",
  "sim/interchange/schemas/oel-scenario-patch-v1.schema.json",
  "sim/interchange/schemas/oel-state-estimate-v1.schema.json",
];

const write = process.argv.includes("--write");
if (process.argv.length > 3 || (process.argv.length === 3 && !write)) {
  throw new Error("Usage: node tools/sync-schemas.mjs [--write]");
}

const expected = new Set();
if (write) mkdirSync(outputRoot, { recursive: true });
for (const source of sources) {
  const sourceBytes = readFileSync(join(repositoryRoot, source));
  const schema = JSON.parse(sourceBytes.toString("utf8"));
  const name = new URL(schema.$id).pathname.split("/").at(-1);
  if (schema.$id !== schemaBase + name) {
    throw new Error(`${source}: $id must be ${schemaBase + name}`);
  }
  if (expected.has(name)) throw new Error(`Duplicate published schema filename: ${name}`);
  expected.add(name);
  const publishedPath = join(outputRoot, name);
  if (write) {
    writeFileSync(publishedPath, sourceBytes);
  } else if (!readFileSync(publishedPath).equals(sourceBytes)) {
    throw new Error(`Published schema differs from ${source}: ${name}`);
  }
}
const actual = new Set(readdirSync(outputRoot));
if (actual.size !== expected.size || [...actual].some((name) => !expected.has(name))) {
  throw new Error("Published schema inventory differs from the explicit public list.");
}
console.log(`Verified ${expected.size} public schema files at ${schemaBase}`);
