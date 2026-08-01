#!/usr/bin/env node

import { access, readFile, readdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const [, , method, paramsPath, timeoutMs = "60000", expectFinal = "0", packageDirArg = ""] = process.argv;

if (!method || !paramsPath) {
  console.error(
    "Usage: openclaw_gateway_call.mjs <method> <params.json> [timeoutMs] [expectFinal] [openclawPackageDir]",
  );
  process.exit(2);
}

const packageDir = packageDirArg || process.env.OPENCLAW_PACKAGE_DIR;
if (!packageDir) {
  console.error("Missing OpenClaw package directory. Set OPENCLAW_PACKAGE_DIR or pass openclawPackageDir.");
  process.exit(2);
}

async function existingPaths(paths) {
  const existing = [];
  for (const path of paths) {
    try {
      await access(path, constants.R_OK);
      existing.push(path);
    } catch {
      // Try the next candidate.
    }
  }
  return existing;
}

async function loadCallGateway() {
  const candidates = [];
  try {
    const requireFromPackage = createRequire(pathToFileURL(`${packageDir}/package.json`).href);
    candidates.push(requireFromPackage.resolve("openclaw/plugin-sdk/testing"));
  } catch {
    // Fall through to known built-file layouts.
  }
  candidates.push(
    `${packageDir}/dist/plugin-sdk/testing.js`,
    `${packageDir}/dist/plugin-sdk/testing.mjs`,
    `${packageDir}/plugin-sdk/testing.js`,
    `${packageDir}/plugin-sdk/testing.mjs`,
    `${packageDir}/dist/testing.js`,
    `${packageDir}/dist/testing.mjs`,
  );
  candidates.push(...await internalGatewayCallCandidates());
  const errors = [];
  for (const sdkPath of await existingPaths(candidates)) {
    try {
      const module = await import(pathToFileURL(sdkPath).href);
      const callGateway = resolveCallGatewayExport(module);
      if (typeof callGateway !== "function") {
        errors.push(`${sdkPath}: no callGateway export; exports=${summarizeExports(module)}`);
        continue;
      }
      if (process.env.OPENCLAW_GATEWAY_HELPER_DEBUG === "1") {
        console.error(
          `[openclaw_gateway_call] using ${sdkPath} name=${callGateway.name || "anonymous"}`,
        );
      }
      return callGateway;
    } catch (error) {
      errors.push(`${sdkPath}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  if (!errors.length) {
    if (process.env.OPENCLAW_GATEWAY_CLI_FALLBACK !== "0") {
      return callGatewayViaCli;
    }
    throw new Error(
      [
        "Cannot find OpenClaw Gateway testing SDK.",
        `packageDir=${packageDir}`,
        "Tried:",
        ...candidates.map((candidate) => `  - ${candidate}`),
        "Set OPENCLAW_PACKAGE_DIR to the OpenClaw package root that contains plugin-sdk/testing.",
      ].join("\n"),
    );
  }
  throw new Error(
    [
      "Found OpenClaw Gateway candidate modules, but none exported callGateway.",
      `packageDir=${packageDir}`,
      "Errors:",
      ...errors.map((error) => `  - ${error}`),
    ].join("\n"),
  );
}

function resolveCallGatewayExport(module) {
  if (typeof module.callGateway === "function") {
    return module.callGateway;
  }
  if (typeof module.r === "function" && module.r.name === "callGateway") {
    return module.r;
  }
  for (const value of Object.values(module)) {
    if (typeof value !== "function") {
      continue;
    }
    if (value.name === "callGateway") {
      return value;
    }
    const source = Function.prototype.toString.call(value);
    if (source.startsWith("async function callGateway(")) {
      return value;
    }
  }
  return undefined;
}

function summarizeExports(module) {
  return Object.entries(module)
    .map(([key, value]) => {
      if (typeof value === "function") {
        return `${key}:function:${value.name || "anonymous"}`;
      }
      return `${key}:${typeof value}`;
    })
    .join(", ");
}

async function internalGatewayCallCandidates() {
  const distDir = `${packageDir}/dist`;
  try {
    const entries = await readdir(distDir);
    const candidates = [];
    for (const entry of entries) {
      if (!/^call-[A-Za-z0-9_-]+\.js$/.test(entry) || entry.startsWith("call-status-")) {
        continue;
      }
      const path = `${distDir}/${entry}`;
      let source = "";
      try {
        source = await readFile(path, "utf8");
      } catch {
        continue;
      }
      if (source.includes("async function callGateway(") && source.includes("GatewayClient")) {
        candidates.push(path);
      }
    }
    return candidates;
  } catch {
    return [];
  }
}

function extractJsonObject(text) {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("OpenClaw gateway CLI produced no JSON output.");
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    // The CLI normally emits plain JSON with --json, but tolerate banners/logs.
  }
  for (let index = 0; index < trimmed.length; index += 1) {
    if (trimmed[index] !== "{") {
      continue;
    }
    try {
      return JSON.parse(trimmed.slice(index));
    } catch {
      // Try the next object-like suffix.
    }
  }
  throw new Error(`OpenClaw gateway CLI output was not JSON: ${trimmed.slice(0, 1000)}`);
}

function callGatewayViaCli({ method, params, expectFinal, timeoutMs, url, token }) {
  return new Promise((resolve, reject) => {
    const openclawBin = process.env.OPENCLAW_BIN || "openclaw";
    const args = [
      "gateway",
      "call",
      method,
      "--json",
      "--timeout",
      String(timeoutMs),
      "--params",
      JSON.stringify(params ?? {}),
    ];
    if (expectFinal) {
      args.push("--expect-final");
    }
    if (url) {
      args.push("--url", url);
    }
    if (token) {
      args.push("--token", token);
    }

    const child = spawn(openclawBin, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      const hint = error && error.code === "E2BIG"
        ? " Gateway params were too large for CLI fallback; install an OpenClaw build with plugin-sdk/testing or set OPENCLAW_PACKAGE_DIR to one."
        : "";
      reject(new Error(`OpenClaw gateway CLI fallback failed to start: ${error.message}.${hint}`));
    });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`OpenClaw gateway CLI fallback failed rc=${code}: ${(stderr || stdout).trim().slice(0, 2000)}`));
        return;
      }
      try {
        resolve(extractJsonObject(stdout || stderr));
      } catch (error) {
        reject(error);
      }
    });
  });
}

const callGateway = await loadCallGateway();
const params = JSON.parse(await readFile(paramsPath, "utf8"));
const opts = {
  timeout: String(timeoutMs),
  json: true,
};

if (process.env.OPENCLAW_GATEWAY_URL) {
  opts.url = process.env.OPENCLAW_GATEWAY_URL;
}
if (process.env.OPENCLAW_GATEWAY_TOKEN) {
  opts.token = process.env.OPENCLAW_GATEWAY_TOKEN;
}

try {
  const result = await callGateway({
    method,
    params,
    expectFinal: expectFinal === "1" || expectFinal === "true",
    timeoutMs: Number(timeoutMs),
    ...opts.url ? { url: opts.url } : {},
    ...opts.token ? { token: opts.token } : {},
  });
  process.stdout.write(`${JSON.stringify(result)}\n`);
} catch (error) {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  console.error(message);
  process.exit(1);
}
