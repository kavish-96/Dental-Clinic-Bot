const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type PromptType = "system" | "intent" | "response" | "context";

export type PromptItem = {
  content: string;
  is_active: boolean;
  version?: number;
};

export type PromptsResponse = {
  agent_id: string;
  prompts: Record<PromptType, PromptItem>;
};

export type RAGResponse = {
  agent_id: string;
  rewrite_prompt: string;
  answer_instructions: string;
  multi_query_prompt: string;
  focus_keywords: string[];
};

export type SynonymsResponse = {
  agent_id: string;
  synonyms: Record<string, string[]>;
};

export type IntentsResponse = {
  agent_id: string;
  intents: string[];
};

export type ToolAliasesResponse = {
  agent_id: string;
  tool_aliases: Record<string, Record<string, string>>;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

function put<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export const adminApi = {
  getPrompts: (agentId: string) =>
    request<PromptsResponse>(`/admin/prompts/${encodeURIComponent(agentId)}`),
  savePrompts: (agentId: string, prompts: PromptsResponse["prompts"]) =>
    put<PromptsResponse>(`/admin/prompts/${encodeURIComponent(agentId)}`, { prompts }),

  getRag: (agentId: string) =>
    request<RAGResponse>(`/admin/rag/${encodeURIComponent(agentId)}`),
  saveRag: (agentId: string, rag: Omit<RAGResponse, "agent_id">) =>
    put<RAGResponse>(`/admin/rag/${encodeURIComponent(agentId)}`, rag),

  getSynonyms: (agentId: string) =>
    request<SynonymsResponse>(`/admin/synonyms/${encodeURIComponent(agentId)}`),
  saveSynonyms: (agentId: string, synonyms: SynonymsResponse["synonyms"]) =>
    put<SynonymsResponse>(`/admin/synonyms/${encodeURIComponent(agentId)}`, { synonyms }),

  getIntents: (agentId: string) =>
    request<IntentsResponse>(`/admin/intents/${encodeURIComponent(agentId)}`),
  saveIntents: (agentId: string, intents: string[]) =>
    put<IntentsResponse>(`/admin/intents/${encodeURIComponent(agentId)}`, { intents }),

  getToolAliases: (agentId: string) =>
    request<ToolAliasesResponse>(`/admin/tool-aliases/${encodeURIComponent(agentId)}`),
  saveToolAliases: (agentId: string, toolAliases: ToolAliasesResponse["tool_aliases"]) =>
    put<ToolAliasesResponse>(`/admin/tool-aliases/${encodeURIComponent(agentId)}`, {
      tool_aliases: toolAliases,
    }),
};
