import type { AppConfig, GatewayProviderConfig } from "../../shared/app";
import { estimateUsageCostUsd } from "../../main/model-pricing-service";
import { fetchWithSystemProxy } from "../../main/system-proxy-fetch";

type ScopeRouterCandidate = {
  estimated_cost_usd?: number;
  model: string;
  provider: string;
  providerName: string;
  selector: string;
};

export type ScopeRouterRouteDecision = {
  model: string;
  reason: string;
};

type ScopeRouterRouteInput = {
  body: Record<string, unknown>;
  config: AppConfig;
  fallbackModel?: string;
  tokenCount: number;
};

const defaultScopeRouterEndpoint = "http://127.0.0.1:8760/route";
const defaultScopeRouterTimeoutMs = 2000;

export async function resolveScopeRouterRouteDecision(
  input: ScopeRouterRouteInput
): Promise<ScopeRouterRouteDecision | undefined> {
  const config = input.config.ScopeRouter;
  if (!config?.enabled) {
    return undefined;
  }

  const candidates = await collectRouteCandidates(input.config);
  if (candidates.length <= 1) {
    return undefined;
  }

  const candidateBySelector = new Map(candidates.map((candidate) => [candidate.selector.toLowerCase(), candidate]));
  const candidateByModel = new Map(candidates.map((candidate) => [candidate.model.toLowerCase(), candidate]));
  const fallback = normalizeRouteSelector(input.fallbackModel) ?? normalizeRouteSelector(readString(input.body.model));
  const endpoint = config.endpoint?.trim() || defaultScopeRouterEndpoint;
  const timeoutMs = Math.max(1000, config.timeoutMs || defaultScopeRouterTimeoutMs);

  try {
    const response = await callScopeRouter(endpoint, {
      body: input.body,
      candidates,
      fallback_model: fallback,
      source: "claude-code-router",
      token_count: input.tokenCount
    }, timeoutMs);
    if (readBoolean(readPath(response, ["routed"])) !== true) {
      return undefined;
    }
    const model = normalizeRouteSelector(readString(readPath(response, ["model"])))
      ?? normalizeRouteSelector(readString(readPath(response, ["selector"])))
      ?? normalizeRouteSelector(readString(readPath(response, ["selected_model"])));
    if (!model) {
      console.warn("[SCOPE-Router] Ignoring response without a model.");
      return undefined;
    }
    const candidate = candidateBySelector.get(model.toLowerCase()) ?? candidateByModel.get(model.toLowerCase());
    if (!candidate) {
      console.warn(`[SCOPE-Router] Ignoring non-candidate model: ${model}`);
      return undefined;
    }
    return {
      model: candidate.selector,
      reason: readString(readPath(response, ["reason"])) || "scope-router"
    };
  } catch (error) {
    console.warn(`[SCOPE-Router] Route request failed: ${formatError(error)}`);
  }

  return undefined;
}

async function collectRouteCandidates(config: AppConfig): Promise<ScopeRouterCandidate[]> {
  const rawCandidates = config.Providers.flatMap((provider) =>
    provider.models.map((model) => ({
      model: model.trim(),
      provider: provider.name.trim(),
      providerName: provider.name.trim(),
      selector: normalizeRouteSelector(`${provider.name}/${model}`) ?? ""
    }))
  ).filter((candidate) => candidate.model && candidate.providerName && candidate.selector);

  const withCost = await Promise.all(rawCandidates.map(async (candidate) => {
    const cost = await estimateUsageCostUsd({
      inputTokens: 1000,
      model: candidate.model,
      outputTokens: 1000,
      provider: candidate.providerName
    }).catch(() => undefined);
    return {
      ...candidate,
      estimated_cost_usd: cost?.amountUsd
    };
  }));

  const seen = new Set<string>();
  return withCost.filter((candidate) => {
    if (seen.has(candidate.selector.toLowerCase())) {
      return false;
    }
    seen.add(candidate.selector.toLowerCase());
    return true;
  });
}

async function callScopeRouter(endpoint: string, payload: unknown, timeoutMs: number): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchWithSystemProxy(endpoint, {
      body: JSON.stringify(payload),
      headers: { "content-type": "application/json" },
      method: "POST",
      signal: controller.signal
    });
    const text = await response.text();
    let body: unknown;
    try {
      body = text ? JSON.parse(text) as unknown : {};
    } catch {
      body = { text };
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${truncateText(JSON.stringify(body), 500)}`);
    }
    return body;
  } finally {
    clearTimeout(timer);
  }
}

function normalizeRouteSelector(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed || undefined;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function readBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function readPath(value: unknown, path: string[]): unknown {
  let current = value;
  for (const key of path) {
    if (!isObject(current)) {
      return undefined;
    }
    current = current[key];
  }
  return current;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function truncateText(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}...`;
}
