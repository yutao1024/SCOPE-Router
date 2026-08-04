#!/usr/bin/env node

const { spawnSync } = require("node:child_process");
const { resolve } = require("node:path");

const script = resolve(__dirname, "../../router_mvp.py");
const result = spawnSync("python", [script, ...process.argv.slice(2)], {
  stdio: "inherit"
});

process.exit(result.status === null ? 1 : result.status);
