import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const rendererPages = ["home", "browser", "tray"];

const common = {
  bundle: true,
  logLevel: "info",
  sourcemap: process.env.NODE_ENV === "development",
};

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

await buildMain();
await buildRenderer();
await copyStaticAssets();

async function buildMain() {
  await esbuild.build({
    ...common,
    entryPoints: {
      main: path.join(root, "src/main/main.ts"),
      cli: path.join(root, "src/main/cli.ts"),
      preload: path.join(root, "src/main/preload.ts"),
      "browser-preload": path.join(root, "src/main/browser-preload.ts"),
    },
    external: ["electron"],
    format: "cjs",
    outdir: path.join(dist, "main"),
    packages: "external",
    platform: "node",
    target: "node22",
  });

  await esbuild.build({
    ...common,
    entryPoints: {
      "fusion-vision-mcp": path.join(root, "src/server/mcp/fusion-vision-mcp.ts"),
    },
    external: ["electron"],
    format: "cjs",
    outdir: path.join(dist, "main/server/gateway"),
    packages: "external",
    platform: "node",
    target: "node22",
  });
}

async function buildRenderer() {
  for (const page of rendererPages) {
    const sourceDir = path.join(root, "src/renderer/pages", page);
    const outdir = path.join(dist, "renderer/pages", page);
    await mkdir(outdir, { recursive: true });
    await esbuild.build({
      ...common,
      absWorkingDir: root,
      alias: {
        "@": path.join(root, "src/renderer"),
      },
      assetNames: "../../assets/[name]-[hash]",
      entryNames: "[name]",
      entryPoints: [path.join(sourceDir, "main.tsx")],
      format: "esm",
      loader: {
        ".ico": "file",
        ".jpg": "file",
        ".png": "file",
        ".svg": "file",
        ".webp": "file",
      },
      outdir,
      platform: "browser",
      target: "chrome120",
    });

    let html = await readFile(path.join(sourceDir, "index.html"), "utf8");
    html = html.replace("./main.tsx", "./main.js");
    if (existsSync(path.join(outdir, "main.css")) && !html.includes("./main.css")) {
      html = html.replace("</head>", '    <link rel="stylesheet" href="./main.css" />\n  </head>');
    }
    await writeFile(path.join(outdir, "index.html"), html);
  }
}

async function copyStaticAssets() {
  await copyIfExists(path.join(root, "assets"), path.join(dist, "assets"));
  await copyIfExists(path.join(root, "models.json"), path.join(dist, "models.json"));
  await copyIfExists(path.join(root, "examples/plugins"), path.join(dist, "marketplace/plugins"));
}

async function copyIfExists(from, to) {
  if (!existsSync(from)) {
    return;
  }
  await mkdir(path.dirname(to), { recursive: true });
  await cp(from, to, { recursive: true });
}
