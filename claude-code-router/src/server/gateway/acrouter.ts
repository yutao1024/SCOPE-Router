import type {
  AppConfig,
  GatewayProviderCapability,
  GatewayProviderConfig,
  GatewayProviderProtocol,
  ProviderCredentialConfig
} from "../../shared/app";
import { estimateUsageCostUsd } from "../../main/model-pricing-service";
import { fetchWithSystemProxy } from "../../main/system-proxy-fetch";
import { normalizeProviderBaseUrl } from "../../shared/provider-url";

type ACRouterCandidate = {
  estimatedCostUsd?: number;
  model: string;
  provider: GatewayProviderConfig;
  providerName: string;
  selector: string;
};

type ACRouterInvoker = ACRouterCandidate & {
  apiKey: string;
  baseUrl: string;
  protocol: GatewayProviderProtocol;
};

type ProviderEndpoint = {
  baseUrl: string;
  protocol: GatewayProviderProtocol;
};

export type ACRouterRouteDecision = {
  model: string;
  reason: string;
  routerModel: string;
  routerProvider: string;
};

type ACRouterRouteInput = {
  body: Record<string, unknown>;
  config: AppConfig;
  fallbackModel?: string;
  tokenCount: number;
};

const defaultRouterTimeoutMs = 8000;
const promptPreviewLimit = 12000;

export async function resolveACRouterRouteDecision(
  input: ACRouterRouteInput
): Promise<ACRouterRouteDecision | undefined> {
  const config = input.config.ACRouter;
  if (config.enabled === false) {
    return undefined;
  }

  const candidates = await collectRouteCandidates(input.config);
  if (candidates.length <= 1) {
    return undefined;
  }

  const candidateBySelector = new Map(candidates.map((candidate) => [candidate.selector, candidate]));
  const fallback = normalizeRouteSelector(input.fallbackModel) ?? normalizeRouteSelector(readString(input.body.model));
  const router = selectRouterInvoker(input.config, candidates);
  if (!router) {
    return fallback && candidateBySelector.has(fallback)
      ? {
          model: fallback,
          reason: "fallback-no-supported-router-model",
          routerModel: "",
          routerProvider: ""
        }
      : undefined;
  }

  const prompt = buildRoutingPrompt(input, candidates, fallback);
  const timeoutMs = Math.max(1000, input.config.ACRouter.timeoutMs || defaultRouterTimeoutMs);

  try {
    const text = await callRouterModel(router, prompt, timeoutMs);
    const parsed = parseRouterResponse(text, candidateBySelector);
    if (parsed) {
      return {
        model: parsed.model,
        reason: parsed.reason || `selected-by-${router.selector}`,
        routerModel: router.model,
        routerProvider: router.providerName
      };
    }
    console.warn(`[ACRouter] Ignoring router response without a valid candidate model: ${truncateText(text, 300)}`);
  } catch (error) {
    console.warn(`[ACRouter] Router model call failed: ${formatError(error)}`);
  }

  return fallback && candidateBySelector.has(fallback)
    ? {
        model: fallback,
        reason: "fallback-router-error",
        routerModel: router.model,
        routerProvider: router.providerName
      }
    : undefined;
}

async function collectRouteCandidates(config: AppConfig): Promise<ACRouterCandidate[]> {
  const rawCandidates = config.Providers.flatMap((provider) =>
    provider.models.map((model) => ({
      model: model.trim(),
      provider,
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
      estimatedCostUsd: cost?.amountUsd
    };
  }));

  const seen = new Set<string>();
  return withCost.filter((candidate) => {
    if (seen.has(candidate.selector)) {
      return false;
    }
    seen.add(candidate.selector);
    return true;
  });
}

function selectRouterInvoker(config: AppConfig, candidates: ACRouterCandidate[]): ACRouterInvoker | undefined {
  const preferred = configuredRouterCandidate(config, candidates);
  if (preferred) {
    const invoker = toInvoker(preferred);
    if (invoker) {
      return invoker;
    }
  }

  const invokers = candidates
    .map(toInvoker)
    .filter((candidate): candidate is ACRouterInvoker => Boolean(candidate))
    .sort((left, right) =>
      (left.estimatedCostUsd ?? Number.POSITIVE_INFINITY) - (right.estimatedCostUsd ?? Number.POSITIVE_INFINITY) ||
      left.selector.localeCompare(right.selector)
    );
  return invokers[0];
}

function configuredRouterCandidate(config: AppConfig, candidates: ACRouterCandidate[]): ACRouterCandidate | undefined {
  const model = config.ACRouter.routerModel?.trim();
  if (!model) {
    return undefined;
  }
  const provider = config.ACRouter.routerProvider?.trim();
  if (provider) {
    const selector = normalizeRouteSelector(`${provider}/${model}`);
    return candidates.find((candidate) => candidate.selector.toLowerCase() === selector?.toLowerCase());
  }
  return candidates.find((candidate) =>
    candidate.model.toLowerCase() === model.toLowerCase() ||
    candidate.selector.toLowerCase() === model.toLowerCase()
  );
}

function toInvoker(candidate: ACRouterCandidate): ACRouterInvoker | undefined {
  const apiKey = providerApiKey(candidate.provider);
  const endpoint = providerEndpoint(candidate.provider);
  if (!apiKey || !endpoint) {
    return undefined;
  }
  return {
    ...candidate,
    apiKey,
    baseUrl: endpoint.baseUrl,
    protocol: endpoint.protocol
  };
}

function providerApiKey(provider: GatewayProviderConfig): string | undefined {
  const direct = provider.api_key || provider.apiKey || provider.apikey;
  if (direct?.trim()) {
    return direct.trim();
  }
  const credential = provider.credentials?.find((item) => credentialApiKey(item));
  return credential ? credentialApiKey(credential) : undefined;
}

function credentialApiKey(credential: ProviderCredentialConfig): string | undefined {
  if (credential.enabled === false) {
    return undefined;
  }
  return (credential.api_key || credential.apiKey || credential.apikey)?.trim() || undefined;
}

function providerEndpoint(provider: GatewayProviderConfig): ProviderEndpoint | undefined {
  const capability = normalizedProviderCapabilities(provider)
    .find((item) => routerProtocolIsSupported(item.type));
  if (capability) {
    return {
      baseUrl: capability.baseUrl,
      protocol: capability.type
    };
  }

  const protocol = providerProtocol(provider);
  if (!protocol || !routerProtocolIsSupported(protocol)) {
    return undefined;
  }
  const configured = readProviderBaseUrl(provider);
  const defaultBaseUrl = defaultBaseUrlForProtocol(protocol);
  const baseUrl = configured
    ? normalizeProviderBaseUrl(configured, protocol)
    : defaultBaseUrl;
  return baseUrl ? { baseUrl, protocol } : undefined;
}

function normalizedProviderCapabilities(provider: GatewayProviderConfig): GatewayProviderCapability[] {
  const capabilities = Array.isArray(provider.capabilities) ? provider.capabilities : [];
  const normalized: GatewayProviderCapability[] = [];
  const seen = new Set<string>();
  for (const capability of capabilities) {
    const type = normalizeProtocol(capability.type);
    const baseUrl = capability.baseUrl?.trim();
    if (!type || !baseUrl) {
      continue;
    }
    const normalizedBaseUrl = normalizeProviderBaseUrl(baseUrl, type);
    if (!normalizedBaseUrl) {
      continue;
    }
    const key = `${type}\n${normalizedBaseUrl}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    normalized.push({
      ...capability,
      baseUrl: normalizedBaseUrl,
      type
    });
  }
  return normalized;
}

function defaultBaseUrlForProtocol(protocol: GatewayProviderProtocol): string | undefined {
  if (protocol === "anthropic_messages") return "https://api.anthropic.com";
  if (protocol === "openai_chat_completions" || protocol === "openai_responses") return "https://api.openai.com";
  return undefined;
}

function routerProtocolIsSupported(protocol: GatewayProviderProtocol): boolean {
  return protocol === "anthropic_messages" ||
    protocol === "openai_chat_completions" ||
    protocol === "openai_responses";
}

function providerProtocol(provider: GatewayProviderConfig): GatewayProviderProtocol | undefined {
  const type = normalizeProtocol(provider.type) ?? normalizeProtocol(provider.provider);
  if (type) {
    return type;
  }
  const capability = provider.capabilities
    ?.map((item) => normalizeProtocol(item.type))
    .find((item): item is GatewayProviderProtocol => Boolean(item));
  if (capability) {
    return capability;
  }
  const baseUrl = (readProviderBaseUrl(provider) || provider.provider || provider.name).toLowerCase();
  if (baseUrl.includes("anthropic") || baseUrl.includes("claude")) {
    return "anthropic_messages";
  }
  if (baseUrl.includes("gemini") || baseUrl.includes("google")) {
    return undefined;
  }
  return "openai_chat_completions";
}

function readProviderBaseUrl(provider: GatewayProviderConfig): string | undefined {
  return provider.baseUrl || provider.baseurl || provider.api_base_url;
}

function normalizeProtocol(value: string | undefined): GatewayProviderProtocol | undefined {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }
  if (normalized === "anthropic" || normalized === "anthropic_messages") return "anthropic_messages";
  if (normalized === "openai" || normalized === "openai_chat" || normalized === "openai_chat_completions") return "openai_chat_completions";
  if (normalized === "openai_responses") return "openai_responses";
  return undefined;
}

function buildRoutingPrompt(
  input: ACRouterRouteInput,
  candidates: ACRouterCandidate[],
  fallback: string | undefined
): string {
  const candidateLines = candidates.map((candidate) => {
    const price = candidate.estimatedCostUsd === undefined
      ? "unknown"
      : candidate.estimatedCostUsd.toFixed(6);
    return `- ${candidate.selector} | estimated_cost_1k_in_1k_out_usd=${price}`;
  }).join("\n");

  return [
    "You are ACRouter, a model router for coding-agent requests.",
    "Choose exactly one candidate model selector for the current task.",
    "Prefer the cheapest model that is likely to solve the request. Use stronger models for complex code edits, debugging, architecture changes, tool-heavy tasks, long context, or image/multimodal requests.",
    "Return only compact JSON: {\"model\":\"provider/model\",\"reason\":\"short reason\"}.",
    "",
    `Token estimate: ${input.tokenCount}`,
    fallback ? `Static fallback/default model: ${fallback}` : "Static fallback/default model: none",
    "",
    "Candidates:",
    candidateLines,
    "",
    "Request summary:",
    requestSummary(input.body)
  ].join("\n");
}

function requestSummary(body: Record<string, unknown>): string {
  const summary = {
    max_tokens: body.max_tokens,
    model: body.model,
    system: truncateText(stringifyCompact(body.system), 1600),
    messages: truncateText(stringifyCompact(body.messages), promptPreviewLimit),
    thinking: body.thinking,
    tool_names: extractToolNames(body.tools)
  };
  return stringifyCompact(summary);
}

async function callRouterModel(router: ACRouterInvoker, prompt: string, timeoutMs: number): Promise<string> {
  if (router.protocol === "anthropic_messages") {
    return callAnthropicRouter(router, prompt, timeoutMs);
  }
  if (router.protocol === "openai_responses") {
    return callOpenAIResponsesRouter(router, prompt, timeoutMs);
  }
  return callOpenAIChatRouter(router, prompt, timeoutMs);
}

async function callAnthropicRouter(router: ACRouterInvoker, prompt: string, timeoutMs: number): Promise<string> {
  const response = await fetchJsonWithTimeout(apiUrl(router.baseUrl, "/v1/messages"), {
    body: JSON.stringify({
      max_tokens: 160,
      messages: [{ content: prompt, role: "user" }],
      model: router.model,
      system: "Route coding-agent tasks. Return only JSON.",
      temperature: 0
    }),
    headers: {
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
      "x-api-key": router.apiKey
    },
    method: "POST"
  }, timeoutMs);
  const text = extractAnthropicText(response);
  if (!text) {
    throw new Error("Anthropic router response did not contain text.");
  }
  return text;
}

async function callOpenAIChatRouter(router: ACRouterInvoker, prompt: string, timeoutMs: number): Promise<string> {
  const response = await fetchJsonWithTimeout(apiUrl(router.baseUrl, "/v1/chat/completions"), {
    body: JSON.stringify({
      max_tokens: 160,
      messages: [
        { content: "Route coding-agent tasks. Return only JSON.", role: "system" },
        { content: prompt, role: "user" }
      ],
      model: router.model,
      temperature: 0
    }),
    headers: {
      authorization: `Bearer ${router.apiKey}`,
      "content-type": "application/json"
    },
    method: "POST"
  }, timeoutMs);
  const text = readString(readPath(response, ["choices", "0", "message", "content"]));
  if (!text) {
    throw new Error("OpenAI chat router response did not contain text.");
  }
  return text;
}

async function callOpenAIResponsesRouter(router: ACRouterInvoker, prompt: string, timeoutMs: number): Promise<string> {
  const response = await fetchJsonWithTimeout(apiUrl(router.baseUrl, "/v1/responses"), {
    body: JSON.stringify({
      input: `Route coding-agent tasks. Return only JSON.\n\n${prompt}`,
      max_output_tokens: 160,
      model: router.model,
      temperature: 0
    }),
    headers: {
      authorization: `Bearer ${router.apiKey}`,
      "content-type": "application/json"
    },
    method: "POST"
  }, timeoutMs);
  const text = readString(readPath(response, ["output_text"])) ?? extractResponsesText(response);
  if (!text) {
    throw new Error("OpenAI responses router response did not contain text.");
  }
  return text;
}

async function fetchJsonWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchWithSystemProxy(url, {
      ...init,
      signal: controller.signal
    });
    const text = await response.text();
    let payload: unknown;
    try {
      payload = text ? JSON.parse(text) as unknown : {};
    } catch {
      payload = text;
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${truncateText(typeof payload === "string" ? payload : stringifyCompact(payload), 500)}`);
    }
    return payload;
  } finally {
    clearTimeout(timer);
  }
}

function parseRouterResponse(
  text: string,
  candidates: Map<string, ACRouterCandidate>
): { model: string; reason: string } | undefined {
  const parsed = parseJsonObject(text) ?? parseJsonObject(extractJsonObject(text) ?? "");
  const model = normalizeRouteSelector(readString(parsed?.model));
  if (model && candidates.has(model)) {
    return {
      model,
      reason: readString(parsed?.reason) || readString(parsed?.reasoning) || "selected-by-acrouter"
    };
  }

  const normalizedText = text.toLowerCase();
  const mentioned = [...candidates.keys()].find((selector) => normalizedText.includes(selector.toLowerCase()));
  return mentioned ? { model: mentioned, reason: "extracted-from-router-text" } : undefined;
}

function extractAnthropicText(value: unknown): string | undefined {
  const content = isRecord(value) && Array.isArray(value.content) ? value.content : [];
  const parts = content
    .map((item) => isRecord(item) && item.type === "text" ? readString(item.text) : undefined)
    .filter((item): item is string => Boolean(item));
  return parts.join("\n").trim() || undefined;
}

function extractResponsesText(value: unknown): string | undefined {
  if (!isRecord(value) || !Array.isArray(value.output)) {
    return undefined;
  }
  const parts: string[] = [];
  for (const item of value.output) {
    if (!isRecord(item) || !Array.isArray(item.content)) {
      continue;
    }
    for (const content of item.content) {
      if (isRecord(content) && typeof content.text === "string") {
        parts.push(content.text);
      }
    }
  }
  return parts.join("\n").trim() || undefined;
}

function extractJsonObject(text: string): string | undefined {
  const match = text.match(/\{[\s\S]*\}/);
  return match?.[0];
}

function parseJsonObject(text: string): Record<string, unknown> | undefined {
  try {
    const value = JSON.parse(text) as unknown;
    return isRecord(value) ? value : undefined;
  } catch {
    return undefined;
  }
}

function extractToolNames(tools: unknown): string[] {
  if (!Array.isArray(tools)) {
    return [];
  }
  return tools
    .map((tool) => isRecord(tool) ? readString(tool.name) : undefined)
    .filter((item): item is string => Boolean(item))
    .slice(0, 50);
}

function apiUrl(baseUrl: string, path: string): string {
  const base = baseUrl.trim().replace(/\/+$/, "");
  if (base.endsWith(path)) {
    return base;
  }
  if (base.endsWith("/v1") && path.startsWith("/v1/")) {
    return `${base}${path.slice(3)}`;
  }
  return `${base}${path}`;
}

function readPath(value: unknown, path: string[]): unknown {
  return path.reduce<unknown>((current, part) => {
    if (Array.isArray(current)) {
      const index = Number(part);
      return Number.isInteger(index) ? current[index] : undefined;
    }
    return isRecord(current) ? current[part] : undefined;
  }, value);
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeRouteSelector(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) {
    return undefined;
  }

  const commaIndex = trimmed.indexOf(",");
  if (commaIndex > 0 && commaIndex < trimmed.length - 1) {
    const provider = trimmed.slice(0, commaIndex).trim();
    const model = trimmed.slice(commaIndex + 1).trim();
    return provider && model ? `${provider}/${model}` : undefined;
  }

  return trimmed;
}

function stringifyCompact(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function truncateText(value: string, maxLength: number): string {
  return value.length > maxLength ? `${value.slice(0, maxLength)}...` : value;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
