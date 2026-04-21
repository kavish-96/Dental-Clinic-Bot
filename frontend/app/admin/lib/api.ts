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

export type KnowledgeStatusResponse = {
  agent_id: string;
  pdf_count: number;
  url_count: number;
  pdf_files: string[];
  urls: string[];
  last_indexed_time: string | null;
  indexing: boolean;
  error: string | null;
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

  getKnowledgeStatus: (agentId: string) =>
    request<KnowledgeStatusResponse>(`/admin/knowledge-status/${encodeURIComponent(agentId)}`),
  rebuildIndex: (agentId: string) =>
    request<{ detail: string; agent_id: string }>(`/admin/rebuild-index/${encodeURIComponent(agentId)}`, {
      method: "POST",
    }),
  addUrl: (agentId: string, url: string) =>
    request<{ detail: string; agent_id: string; url: string }>(`/admin/add-url`, {
      method: "POST",
      body: JSON.stringify({ agent_id: agentId, url }),
      headers: { "Content-Type": "application/json" },
    }),
  deletePdf: (agentId: string, filename: string) =>
    request<{ detail: string; agent_id: string; filename: string }>(
      `/admin/pdf/${encodeURIComponent(agentId)}?filename=${encodeURIComponent(filename)}`,
      { method: "DELETE" },
    ),
  deleteUrl: (agentId: string, url: string) =>
    request<{ detail: string; agent_id: string; url: string }>(
      `/admin/url/${encodeURIComponent(agentId)}?url=${encodeURIComponent(url)}`,
      { method: "DELETE" },
    ),
  uploadPdf: (
    agentId: string,
    file: File,
    onProgress?: (progress: number) => void,
  ) =>
    new Promise<{ detail: string; agent_id: string; filename: string }>((resolve, reject) => {
      const formData = new FormData();
      formData.append("agent_id", agentId);
      formData.append("file", file);

      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_URL}/admin/upload-pdf`);
      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable || !onProgress) return;
        onProgress(Math.round((event.loaded / event.total) * 100));
      };
      xhr.onload = () => {
        try {
          const parsed = JSON.parse(xhr.responseText || "{}");
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(parsed);
            return;
          }
          reject(new Error(parsed.detail || `Request failed with status ${xhr.status}`));
        } catch {
          reject(new Error(`Request failed with status ${xhr.status}`));
        }
      };
      xhr.onerror = () => reject(new Error("Upload failed."));
      xhr.send(formData);
    }),
};
