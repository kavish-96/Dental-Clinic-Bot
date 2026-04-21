"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Card } from "./components/Card";
import { Input, TextArea } from "./components/Input";
import { SectionHeader } from "./components/SectionHeader";
import { TagInput } from "./components/TagInput";
import {
  adminApi,
  type KnowledgeStatusResponse,
  type PromptType,
  type PromptsResponse,
  type RAGResponse,
  type ToolAliasesResponse,
} from "./lib/api";

type Section = "prompts" | "rag" | "synonyms" | "intents" | "tool-aliases" | "knowledge-base";
type SaveState = "idle" | "loading" | "saving" | "saved" | "error";
type KnowledgeTab = "pdfs" | "websites";

const PROMPT_TABS: { key: PromptType; label: string }[] = [
  { key: "system", label: "System" },
  { key: "intent", label: "Intent" },
  { key: "response", label: "Response" },
  { key: "context", label: "Context" },
];

const NAV_ITEMS: { key: Section; label: string; hint: string }[] = [
  { key: "prompts", label: "Prompts", hint: "Core agent language" },
  { key: "rag", label: "RAG Config", hint: "Retrieval instructions" },
  { key: "synonyms", label: "Synonyms", hint: "Search expansion" },
  { key: "intents", label: "Intents", hint: "Routing labels" },
  { key: "tool-aliases", label: "Tool Aliases", hint: "Tool argument mapping" },
  { key: "knowledge-base", label: "Knowledge Base", hint: "PDFs and websites" },
];

const emptyPrompts: PromptsResponse["prompts"] = {
  system: { content: "", is_active: true },
  intent: { content: "", is_active: true },
  response: { content: "", is_active: true },
  context: { content: "", is_active: true },
};

const emptyRag: Omit<RAGResponse, "agent_id"> = {
  rewrite_prompt: "",
  answer_instructions: "",
  multi_query_prompt: "",
  focus_keywords: [],
};

const emptyKnowledgeStatus: KnowledgeStatusResponse = {
  agent_id: "dental_bot",
  pdf_count: 0,
  url_count: 0,
  pdf_files: [],
  urls: [],
  last_indexed_time: null,
  indexing: false,
  error: null,
};

function SaveButton({
  saving,
  onClick,
  label = "Save changes",
}: {
  saving: boolean;
  onClick: () => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={saving}
      className="rounded-lg bg-zinc-950 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-teal-700 disabled:translate-y-0 disabled:cursor-wait disabled:bg-zinc-400"
    >
      {saving ? "Saving..." : label}
    </button>
  );
}

function EditableKeyField({
  initialValue,
  ariaLabel,
  placeholder,
  className,
  onCommit,
}: {
  initialValue: string;
  ariaLabel: string;
  placeholder?: string;
  className?: string;
  onCommit: (value: string) => void;
}) {
  const [draft, setDraft] = useState(initialValue);

  useEffect(() => {
    setDraft(initialValue);
  }, [initialValue]);

  function commit() {
    const nextValue = draft.trim();
    if (!nextValue || nextValue === initialValue) {
      setDraft(initialValue);
      return;
    }
    onCommit(nextValue);
  }

  return (
    <Input
      value={draft}
      aria-label={ariaLabel}
      placeholder={placeholder}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          commit();
        }
        if (event.key === "Escape") {
          setDraft(initialValue);
        }
      }}
      className={className}
    />
  );
}

function TrashButton({
  onClick,
  disabled,
  label,
}: {
  onClick: () => void;
  disabled?: boolean;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className="rounded-lg p-2 text-rose-600 transition hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 6h18" />
        <path d="M8 6V4h8v2" />
        <path d="M19 6l-1 14H6L5 6" />
        <path d="M10 11v6" />
        <path d="M14 11v6" />
      </svg>
    </button>
  );
}

export default function AdminPage() {
  const [agentId, setAgentId] = useState("dental_bot");
  const [activeSection, setActiveSection] = useState<Section>("prompts");
  const [activePrompt, setActivePrompt] = useState<PromptType>("system");
  const [knowledgeTab, setKnowledgeTab] = useState<KnowledgeTab>("pdfs");
  const [saveState, setSaveState] = useState<SaveState>("loading");
  const [toast, setToast] = useState("");
  const [error, setError] = useState("");
  const [knowledge, setKnowledge] = useState<KnowledgeStatusResponse>(emptyKnowledgeStatus);
  const [urlInput, setUrlInput] = useState("");
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploading, setUploading] = useState(false);
  const [addingUrl, setAddingUrl] = useState(false);
  const [removingPdf, setRemovingPdf] = useState<string | null>(null);
  const [removingUrl, setRemovingUrl] = useState<string | null>(null);
  const [rebuilding, setRebuilding] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const wasIndexingRef = useRef(false);

  const [prompts, setPrompts] = useState<PromptsResponse["prompts"]>(emptyPrompts);
  const [rag, setRag] = useState<Omit<RAGResponse, "agent_id">>(emptyRag);
  const [synonyms, setSynonyms] = useState<Record<string, string[]>>({});
  const [intents, setIntents] = useState<string[]>([]);
  const [toolAliases, setToolAliases] = useState<ToolAliasesResponse["tool_aliases"]>({});

  const saving = saveState === "saving";
  const knowledgeBusy =
    knowledge.indexing || uploading || addingUrl || rebuilding || !!removingPdf || !!removingUrl;

  function showToast(message: string) {
    setToast(message);
    window.setTimeout(() => {
      setToast((current) => (current === message ? "" : current));
    }, 2200);
  }

  async function refreshKnowledgeStatus(
    currentAgentId: string,
  ): Promise<KnowledgeStatusResponse> {
    const next = await adminApi.getKnowledgeStatus(currentAgentId);
    setKnowledge(next);
    return next;
  }

  useEffect(() => {
    let alive = true;

    async function loadConfig() {
      setSaveState("loading");
      setError("");

      try {
        const [promptsData, ragData, synonymsData, intentsData, toolAliasesData, knowledgeData] =
          await Promise.all([
            adminApi.getPrompts(agentId),
            adminApi.getRag(agentId),
            adminApi.getSynonyms(agentId),
            adminApi.getIntents(agentId),
            adminApi.getToolAliases(agentId),
            adminApi.getKnowledgeStatus(agentId),
          ]);

        if (!alive) return;
        setPrompts(promptsData.prompts);
        setRag({
          rewrite_prompt: ragData.rewrite_prompt,
          answer_instructions: ragData.answer_instructions,
          multi_query_prompt: ragData.multi_query_prompt,
          focus_keywords: ragData.focus_keywords,
        });
        setSynonyms(synonymsData.synonyms);
        setIntents(intentsData.intents);
        setToolAliases(toolAliasesData.tool_aliases);
        setKnowledge(knowledgeData);
        setSaveState("idle");
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : "Unable to load admin config.");
        setSaveState("error");
      }
    }

    void loadConfig();

    return () => {
      alive = false;
    };
  }, [agentId]);

  function showSaved() {
    showToast("Saved successfully");
    setSaveState("saved");
    window.setTimeout(() => {
      setSaveState("idle");
    }, 2200);
  }

  async function runSave(task: () => Promise<unknown>) {
    setSaveState("saving");
    setError("");
    try {
      await task();
      showSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save changes.");
      setSaveState("error");
    }
  }

  useEffect(() => {
    let intervalId: number | undefined;
    if (knowledge.indexing) {
      intervalId = window.setInterval(() => {
        void refreshKnowledgeStatus(agentId);
      }, 2000);
    }
    if (wasIndexingRef.current && !knowledge.indexing && !knowledge.error) {
      showToast("Index updated");
    }
    wasIndexingRef.current = knowledge.indexing;
    return () => {
      if (intervalId) {
        window.clearInterval(intervalId);
      }
    };
  }, [agentId, knowledge.error, knowledge.indexing]);

  async function handlePdfUpload(file: File | null) {
    if (!file) return;
    setUploading(true);
    setUploadProgress(0);
    setError("");
    try {
      await adminApi.uploadPdf(agentId, file, setUploadProgress);
      showToast("Upload successful");
      await refreshKnowledgeStatus(agentId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
      setUploadProgress(null);
      setDragActive(false);
    }
  }

  async function handleAddUrl() {
    const nextUrl = urlInput.trim();
    if (!nextUrl || knowledgeBusy) return;
    setAddingUrl(true);
    setError("");
    try {
      const response = await adminApi.addUrl(agentId, nextUrl);
      setUrlInput("");
      showToast(response.detail || "URL added successfully");
      await refreshKnowledgeStatus(agentId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to add URL.");
    } finally {
      setAddingUrl(false);
    }
  }

  async function handleDeletePdf(fileName: string) {
    if (knowledgeBusy) return;
    setRemovingPdf(fileName);
    setError("");
    try {
      const response = await adminApi.deletePdf(agentId, fileName);
      showToast(response.detail || "PDF removed");
      await refreshKnowledgeStatus(agentId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to remove PDF.");
    } finally {
      setRemovingPdf(null);
    }
  }

  async function handleDeleteUrl(url: string) {
    if (knowledgeBusy) return;
    setRemovingUrl(url);
    setError("");
    try {
      const response = await adminApi.deleteUrl(agentId, url);
      showToast(response.detail || "URL removed");
      await refreshKnowledgeStatus(agentId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to remove URL.");
    } finally {
      setRemovingUrl(null);
    }
  }

  async function handleRebuildIndex() {
    if (knowledge.indexing || rebuilding) return;
    setRebuilding(true);
    setError("");
    try {
      const response = await adminApi.rebuildIndex(agentId);
      showToast(response.detail || "Index rebuild started");
      await refreshKnowledgeStatus(agentId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to rebuild index.");
    } finally {
      setRebuilding(false);
    }
  }

  function formatIndexedTime(value: string | null) {
    if (!value) return "Not indexed yet";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
  }

  return (
    <div className="h-screen overflow-hidden bg-[#f6f8f7] text-zinc-950">
      {knowledge.indexing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/40 backdrop-blur-[2px]">
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-6 py-4 text-center shadow-xl">
            <p className="text-sm font-semibold text-amber-900">Updating knowledge base...</p>
            <p className="mt-1 text-sm text-amber-800">Please wait until reindexing is complete.</p>
          </div>
        </div>
      )}
      <div className="flex h-full min-h-0">
        <aside className="hidden h-screen w-72 shrink-0 border-r border-zinc-200 bg-white/85 px-4 py-5 shadow-[10px_0_35px_rgba(30,41,59,0.05)] backdrop-blur lg:block">
          <div className="mb-8 px-2">
            <p className="text-sm font-semibold uppercase text-teal-700">AI Admin</p>
            <h1 className="mt-2 text-2xl font-semibold">Control Center</h1>
          </div>
          <nav className="space-y-2">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setActiveSection(item.key)}
                className={`w-full rounded-lg px-3 py-3 text-left transition hover:-translate-y-0.5 hover:bg-zinc-100 ${
                  activeSection === item.key
                    ? "bg-zinc-950 text-white shadow-lg shadow-zinc-950/15"
                    : "text-zinc-700"
                }`}
              >
                <span className="block text-sm font-semibold">{item.label}</span>
                <span
                  className={`mt-1 block text-xs ${
                    activeSection === item.key ? "text-zinc-300" : "text-zinc-400"
                  }`}
                >
                  {item.hint}
                </span>
              </button>
            ))}
          </nav>
          <div className="mt-6 border-t border-zinc-200 pt-5">
            <p className="px-2 text-xs font-semibold uppercase text-zinc-400">
              Conversation
            </p>
            <Link
              href="/"
              className="mt-2 block rounded-lg border border-teal-200 bg-teal-50 px-3 py-3 text-teal-800 transition hover:-translate-y-0.5 hover:border-teal-300 hover:bg-teal-100"
            >
              <span className="block text-sm font-semibold">Back to Chat</span>
              <span className="mt-1 block text-xs text-teal-600">Return to patient assistant</span>
            </Link>
          </div>
        </aside>

        <main className="flex h-screen min-w-0 flex-1 flex-col overflow-hidden">
          <header className="shrink-0 border-b border-zinc-200 bg-white/90 px-4 py-4 backdrop-blur md:px-8">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-sm font-semibold text-teal-700">Production configuration</p>
                <h2 className="mt-1 text-2xl font-semibold">AI Behavior Admin Panel</h2>
              </div>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <label className="text-sm font-medium text-zinc-600">
                  Agent
                  <select
                    value={agentId}
                    onChange={(event) => setAgentId(event.target.value)}
                    className="ml-3 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-950 outline-none transition focus:border-teal-500 focus:ring-4 focus:ring-teal-100"
                  >
                    <option value="dental_bot">dental_bot</option>
                  </select>
                </label>
              </div>
            </div>
            <div className="mt-4 flex gap-2 overflow-x-auto lg:hidden">
              <Link
                href="/"
                className="shrink-0 rounded-lg border border-teal-200 bg-teal-50 px-3 py-2 text-sm font-semibold text-teal-800"
              >
                Back to Chat
              </Link>
              {NAV_ITEMS.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => setActiveSection(item.key)}
                  className={`shrink-0 rounded-lg px-3 py-2 text-sm font-semibold transition ${
                    activeSection === item.key
                      ? "bg-zinc-950 text-white"
                      : "bg-zinc-100 text-zinc-700"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </header>

          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 md:px-8">
            {error && (
              <div className="mb-5 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {error}
              </div>
            )}

            {activeSection === "prompts" && (
              <Card>
                <SectionHeader
                  eyebrow="Prompts"
                  title="Prompt Editor"
                  description="Tune how the assistant reasons, classifies requests, responds, and resolves context."
                  action={
                    <SaveButton
                      saving={saving}
                      onClick={() => runSave(() => adminApi.savePrompts(agentId, prompts))}
                    />
                  }
                />
                <div className="p-5">
                  <div className="mb-4 flex flex-wrap gap-2">
                    {PROMPT_TABS.map((tab) => (
                      <button
                        key={tab.key}
                        type="button"
                        onClick={() => setActivePrompt(tab.key)}
                        className={`rounded-lg px-4 py-2 text-sm font-semibold transition hover:-translate-y-0.5 ${
                          activePrompt === tab.key
                            ? "bg-teal-600 text-white shadow-md shadow-teal-700/20"
                            : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
                        }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                  <div className="mb-4 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
                    <div>
                      <p className="text-sm font-semibold text-zinc-900">
                        {PROMPT_TABS.find((tab) => tab.key === activePrompt)?.label} prompt
                      </p>
                    </div>
                  </div>
                  <TextArea
                    value={prompts[activePrompt]?.content || ""}
                    onChange={(event) =>
                      setPrompts({
                        ...prompts,
                        [activePrompt]: {
                          ...prompts[activePrompt],
                          content: event.target.value,
                        },
                      })
                    }
                    className="min-h-[380px] bg-[#fbfcfc]"
                  />
                </div>
              </Card>
            )}

            {activeSection === "rag" && (
              <Card>
                <SectionHeader
                  eyebrow="RAG"
                  title="Retrieval Configuration"
                  description="Shape query rewriting, answer grounding, multi-query expansion, and focus terms."
                  action={
                    <SaveButton
                      saving={saving}
                      onClick={() => runSave(() => adminApi.saveRag(agentId, rag))}
                    />
                  }
                />
                <div className="grid gap-5 p-5">
                  <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                    <TextArea
                      label="Rewrite prompt"
                      value={rag.rewrite_prompt}
                      onChange={(event) => setRag({ ...rag, rewrite_prompt: event.target.value })}
                    />
                  </div>
                  <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                    <TextArea
                      label="Answer instructions"
                      value={rag.answer_instructions}
                      onChange={(event) => setRag({ ...rag, answer_instructions: event.target.value })}
                    />
                  </div>
                  <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                    <TextArea
                      label="Multi-query prompt"
                      value={rag.multi_query_prompt}
                      onChange={(event) => setRag({ ...rag, multi_query_prompt: event.target.value })}
                    />
                  </div>
                  <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                    <p className="mb-2 text-sm font-medium text-zinc-700">Focus keywords</p>
                    <TagInput
                      values={rag.focus_keywords}
                      onChange={(values) => setRag({ ...rag, focus_keywords: values })}
                      placeholder="Add keyword"
                    />
                  </div>
                </div>
              </Card>
            )}

            {activeSection === "synonyms" && (
              <Card>
                <SectionHeader
                  eyebrow="Synonyms"
                  title="Synonym Categories"
                  description="Manage words the retrieval layer should treat as related."
                  action={
                    <SaveButton
                      saving={saving}
                      onClick={() => runSave(() => adminApi.saveSynonyms(agentId, synonyms))}
                    />
                  }
                />
                <div className="space-y-4 p-5">
                  {Object.entries(synonyms).map(([category, words], index) => (
                    <div key={`${category}-${index}`} className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <EditableKeyField
                          initialValue={category}
                          ariaLabel="Category"
                          onCommit={(nextCategory) => {
                            const next = { ...synonyms };
                            delete next[category];
                            next[nextCategory] = words;
                            setSynonyms(next);
                          }}
                          className="max-w-xs font-semibold"
                        />
                        <button
                          type="button"
                          onClick={() => {
                            const next = { ...synonyms };
                            delete next[category];
                            setSynonyms(next);
                          }}
                          className="rounded-lg px-3 py-2 text-sm font-semibold text-rose-600 transition hover:bg-rose-50"
                        >
                          Remove
                        </button>
                      </div>
                      <TagInput
                        values={words}
                        onChange={(values) => setSynonyms({ ...synonyms, [category]: values })}
                        placeholder="Add synonym"
                      />
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() => setSynonyms({ ...synonyms, new_category: [] })}
                    className="rounded-lg border border-dashed border-zinc-300 px-4 py-3 text-sm font-semibold text-zinc-700 transition hover:border-teal-500 hover:bg-teal-50"
                  >
                    Add category
                  </button>
                </div>
              </Card>
            )}

            {activeSection === "intents" && (
              <Card>
                <SectionHeader
                  eyebrow="Intents"
                  title="Intent Labels"
                  description="Keep the classifier choices aligned with the workflows your assistant supports."
                  action={
                    <SaveButton
                      saving={saving}
                      onClick={() => runSave(() => adminApi.saveIntents(agentId, intents))}
                    />
                  }
                />
                <div className="p-5">
                  <TagInput
                    values={intents}
                    onChange={setIntents}
                    placeholder="Add intent"
                  />
                </div>
              </Card>
            )}

            {activeSection === "tool-aliases" && (
              <Card>
                <SectionHeader
                  eyebrow="Tool aliases"
                  title="Tool Argument Aliases"
                  description="Map phrasing from model output to the canonical argument names your tools execute."
                  action={
                    <SaveButton
                      saving={saving}
                      onClick={() => runSave(() => adminApi.saveToolAliases(agentId, toolAliases))}
                    />
                  }
                />
                <div className="space-y-4 p-5">
                  {Object.entries(toolAliases).map(([toolName, aliases], toolIndex) => (
                    <div key={`${toolName}-${toolIndex}`} className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                      <div className="mb-4 flex items-center justify-between gap-3">
                        <EditableKeyField
                          initialValue={toolName}
                          ariaLabel="Tool name"
                          onCommit={(nextToolName) => {
                            const next = { ...toolAliases };
                            delete next[toolName];
                            next[nextToolName] = aliases;
                            setToolAliases(next);
                          }}
                          className="max-w-sm font-semibold"
                        />
                        <button
                          type="button"
                          onClick={() => {
                            const next = { ...toolAliases };
                            delete next[toolName];
                            setToolAliases(next);
                          }}
                          className="rounded-lg px-3 py-2 text-sm font-semibold text-rose-600 transition hover:bg-rose-50"
                        >
                          Remove
                        </button>
                      </div>
                      <div className="space-y-2">
                        {Object.entries(aliases).map(([alias, target], aliasIndex) => (
                          <div key={`${alias}-${aliasIndex}`} className="grid gap-2 sm:grid-cols-[1fr_1fr_auto]">
                            <EditableKeyField
                              initialValue={alias}
                              ariaLabel="Alias"
                              placeholder="alias"
                              onCommit={(nextAlias) => {
                                const nextAliases = { ...aliases };
                                delete nextAliases[alias];
                                nextAliases[nextAlias] = target;
                                setToolAliases({ ...toolAliases, [toolName]: nextAliases });
                              }}
                            />
                            <Input
                              value={target}
                              aria-label="Canonical argument"
                              placeholder="canonical argument"
                              onChange={(event) =>
                                setToolAliases({
                                  ...toolAliases,
                                  [toolName]: { ...aliases, [alias]: event.target.value },
                                })
                              }
                            />
                            <button
                              type="button"
                              onClick={() => {
                                const nextAliases = { ...aliases };
                                delete nextAliases[alias];
                                setToolAliases({ ...toolAliases, [toolName]: nextAliases });
                              }}
                              className="rounded-lg px-3 py-2 text-sm font-semibold text-rose-600 transition hover:bg-rose-50"
                            >
                              Remove
                            </button>
                          </div>
                        ))}
                      </div>
                      <button
                        type="button"
                        onClick={() =>
                          setToolAliases({
                            ...toolAliases,
                            [toolName]: { ...aliases, new_alias: "canonical_argument" },
                          })
                        }
                        className="mt-3 rounded-lg border border-dashed border-zinc-300 px-3 py-2 text-sm font-semibold text-zinc-700 transition hover:border-teal-500 hover:bg-teal-50"
                      >
                        Add alias
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      setToolAliases({
                        ...toolAliases,
                        new_tool: { new_alias: "canonical_argument" },
                      })
                    }
                    className="rounded-lg border border-dashed border-zinc-300 px-4 py-3 text-sm font-semibold text-zinc-700 transition hover:border-teal-500 hover:bg-teal-50"
                  >
                    Add tool
                  </button>
                </div>
              </Card>
            )}

            {activeSection === "knowledge-base" && (
              <Card>
                <SectionHeader
                  eyebrow="Knowledge"
                  title="Knowledge Base"
                  description="Manage PDFs and website sources for agent-scoped retrieval and automatic FAISS indexing."
                />
                <div className="space-y-5 p-5">
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                      <p className="text-xs font-semibold uppercase text-teal-700">PDFs</p>
                      <p className="mt-2 text-3xl font-semibold text-zinc-950">{knowledge.pdf_count}</p>
                    </div>
                    <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                      <p className="text-xs font-semibold uppercase text-teal-700">Websites</p>
                      <p className="mt-2 text-3xl font-semibold text-zinc-950">{knowledge.url_count}</p>
                    </div>
                    <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                      <p className="text-xs font-semibold uppercase text-teal-700">Last indexed</p>
                      <p className="mt-2 text-sm font-medium text-zinc-700">
                        {formatIndexedTime(knowledge.last_indexed_time)}
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => setKnowledgeTab("pdfs")}
                        disabled={knowledge.indexing}
                        className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${
                          knowledgeTab === "pdfs"
                            ? "bg-teal-600 text-white shadow-md shadow-teal-700/20"
                            : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
                        } disabled:cursor-not-allowed disabled:opacity-60`}
                      >
                        PDFs
                      </button>
                      <button
                        type="button"
                        onClick={() => setKnowledgeTab("websites")}
                        disabled={knowledge.indexing}
                        className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${
                          knowledgeTab === "websites"
                            ? "bg-teal-600 text-white shadow-md shadow-teal-700/20"
                            : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
                        } disabled:cursor-not-allowed disabled:opacity-60`}
                      >
                        Websites
                      </button>
                    </div>
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                      <p className="text-sm text-zinc-500">
                        Changes to PDFs and websites are staged until you rebuild the index.
                      </p>
                      <button
                        type="button"
                        onClick={() => void handleRebuildIndex()}
                        disabled={knowledgeBusy}
                        className="rounded-lg border border-amber-300 bg-amber-100 px-4 py-2 text-sm font-semibold text-amber-900 transition hover:-translate-y-0.5 hover:bg-amber-200 disabled:translate-y-0 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {rebuilding || knowledge.indexing ? "Rebuilding..." : "Reload Index"}
                      </button>
                    </div>
                  </div>

                  {knowledge.indexing && (
                    <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-800">
                      Updating knowledge base...
                    </div>
                  )}
                  {knowledge.error && (
                    <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                      {knowledge.error}
                    </div>
                  )}

                  {knowledgeTab === "pdfs" && (
                    <div className="space-y-4">
                      <label
                        className={`block rounded-lg border-2 border-dashed p-6 text-center transition ${
                          dragActive
                            ? "border-teal-500 bg-teal-50"
                            : "border-teal-200 bg-teal-50/70 hover:border-teal-300"
                        } ${knowledgeBusy ? "cursor-not-allowed opacity-70" : "cursor-pointer"}`}
                        onDragOver={(event) => {
                          event.preventDefault();
                          if (!knowledgeBusy) setDragActive(true);
                        }}
                        onDragLeave={() => setDragActive(false)}
                        onDrop={(event) => {
                          event.preventDefault();
                          if (knowledgeBusy) return;
                          const droppedFile = event.dataTransfer.files?.[0] || null;
                          void handlePdfUpload(droppedFile);
                        }}
                      >
                        <input
                          type="file"
                          accept=".pdf,application/pdf"
                          className="hidden"
                          disabled={knowledgeBusy}
                          onChange={(event) => {
                            const selectedFile = event.target.files?.[0] || null;
                            void handlePdfUpload(selectedFile);
                            event.currentTarget.value = "";
                          }}
                        />
                        <p className="text-base font-semibold text-zinc-950">Drag and drop a PDF here</p>
                        <p className="mt-2 text-sm text-zinc-500">or click to choose a file for {agentId}</p>
                        {uploadProgress !== null && (
                          <div className="mx-auto mt-4 max-w-md">
                            <div className="h-2 rounded-full bg-white">
                              <div
                                className="h-2 rounded-full bg-teal-600 transition-all"
                                style={{ width: `${uploadProgress}%` }}
                              />
                            </div>
                            <p className="mt-2 text-xs font-semibold text-teal-700">{uploadProgress}% uploaded</p>
                          </div>
                        )}
                      </label>

                      <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                        <p className="text-sm font-semibold text-zinc-950">Uploaded files</p>
                        {knowledge.pdf_files.length === 0 ? (
                          <p className="mt-2 text-sm text-zinc-500">No PDFs uploaded yet.</p>
                        ) : (
                          <ul className="mt-3 space-y-2">
                            {knowledge.pdf_files.map((fileName: string) => (
                              <li
                                key={fileName}
                                className="flex items-center justify-between gap-3 rounded-lg border border-white/80 bg-white/80 px-3 py-2 text-sm text-zinc-700"
                              >
                                <span className="truncate">{fileName}</span>
                                <TrashButton
                                  label={`Remove ${fileName}`}
                                  disabled={knowledgeBusy}
                                  onClick={() => void handleDeletePdf(fileName)}
                                />
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </div>
                  )}

                  {knowledgeTab === "websites" && (
                    <div className="space-y-4">
                      <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                        <div className="flex flex-col gap-3 sm:flex-row">
                          <input
                            value={urlInput}
                            onChange={(event) => setUrlInput(event.target.value)}
                            placeholder="https://example.com"
                            className="min-w-0 flex-1 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none transition placeholder:text-zinc-400 focus:border-teal-500 focus:ring-4 focus:ring-teal-100 disabled:cursor-not-allowed disabled:bg-zinc-100"
                          />
                          <button
                            type="button"
                            disabled={knowledgeBusy || !urlInput.trim()}
                            onClick={() => void handleAddUrl()}
                            className="rounded-lg bg-zinc-950 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-teal-700 disabled:translate-y-0 disabled:cursor-not-allowed disabled:bg-zinc-400"
                          >
                            {addingUrl ? "Adding..." : "Add URL"}
                          </button>
                        </div>
                      </div>

                      <div className="rounded-lg border border-teal-200 bg-teal-50/70 p-4">
                        <p className="text-sm font-semibold text-zinc-950">Tracked URLs</p>
                        {knowledge.urls.length === 0 ? (
                          <p className="mt-2 text-sm text-zinc-500">No websites added yet.</p>
                        ) : (
                          <ul className="mt-3 space-y-2">
                            {knowledge.urls.map((url: string) => (
                              <li
                                key={url}
                                className="flex items-center justify-between gap-3 rounded-lg border border-white/80 bg-white/80 px-3 py-2 text-sm text-zinc-700"
                              >
                                <span className="truncate">{url}</span>
                                <TrashButton
                                  label={`Remove ${url}`}
                                  disabled={knowledgeBusy}
                                  onClick={() => void handleDeleteUrl(url)}
                                />
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            )}
          </div>
        </main>
      </div>

      {toast && (
        <div className="fixed bottom-5 right-5 rounded-lg border border-teal-200 bg-white px-4 py-3 text-sm font-semibold text-teal-700 shadow-2xl shadow-zinc-950/15">
          {toast}
        </div>
      )}
    </div>
  );
}
