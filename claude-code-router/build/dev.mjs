import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import "./build.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const electronCli = path.join(root, "node_modules", "electron", "cli.js");

const child = spawn(process.execPath, [electronCli, root], {
  cwd: root,
  env: {
    ...process.env,
    NODE_ENV: "development",
  },
  stdio: "inherit",
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
