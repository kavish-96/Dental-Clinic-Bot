"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Message = { role: "user" | "assistant"; content: string };

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    sessionStorage.setItem("session_id", crypto.randomUUID());
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    const nextMessages: Message[] = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setLoading(true);
    setError(null);

    try {
      const sessionId = sessionStorage.getItem("session_id") || crypto.randomUUID();
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages, session_id: sessionId }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const detail = Array.isArray(data.detail) ? data.detail[0]?.msg ?? data.detail : data.detail;
        throw new Error(detail || `Request failed: ${res.status}`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response || "No response." },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Something went wrong.";
      setError(msg);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I ran into a temporary issue while processing that request. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-screen overflow-hidden bg-[#f6f8f7] text-zinc-950">
      <div className="flex h-full min-h-0">
        <aside className="hidden h-screen w-72 shrink-0 border-r border-zinc-200 bg-white/85 px-4 py-5 shadow-[10px_0_35px_rgba(30,41,59,0.05)] backdrop-blur lg:block">
          <div className="mb-8 px-2">
            <p className="text-sm font-semibold uppercase text-teal-700">Dental AI</p>
            <h1 className="mt-2 text-2xl font-semibold">Clinic Console</h1>
          </div>
          <nav className="space-y-2">
            <div className="rounded-lg bg-zinc-950 px-3 py-3 text-white shadow-lg shadow-zinc-950/15">
              <span className="block text-sm font-semibold">Chat Assistant</span>
              <span className="mt-1 block text-xs text-zinc-300">Patient conversation</span>
            </div>
            <Link
              href="/admin"
              className="block rounded-lg px-3 py-3 text-zinc-700 transition hover:-translate-y-0.5 hover:bg-zinc-100"
            >
              <span className="block text-sm font-semibold">Admin Panel</span>
              <span className="mt-1 block text-xs text-zinc-400">AI configuration</span>
            </Link>
          </nav>
        </aside>

        <main className="flex h-screen min-w-0 flex-1 flex-col overflow-hidden">
          <header className="shrink-0 border-b border-zinc-200 bg-white/90 px-4 py-4 backdrop-blur md:px-8">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-sm font-semibold text-teal-700">Live assistant</p>
                <h2 className="mt-1 text-2xl font-semibold">Dental Clinic Assistant</h2>
                <p className="mt-1 text-sm text-zinc-500">
                  Book, cancel, update, or view appointments, and answer clinic questions.
                </p>
              </div>
            </div>
          </header>

          <section className="flex min-h-0 flex-1 flex-col overflow-hidden px-4 py-6 md:px-8">
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-[0_18px_45px_rgba(30,41,59,0.08)]">
              <div className="shrink-0 border-b border-zinc-100 px-5 py-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-lg font-semibold text-zinc-950">Patient Chat</h3>
                    <p className="mt-1 text-sm text-zinc-500">
                      Every response uses the current AI configuration.
                    </p>
                  </div>
                  <div className="hidden rounded-lg bg-teal-50 px-3 py-2 text-xs font-semibold text-teal-700 sm:block">
                    dental_bot
                  </div>
                </div>
              </div>

              <div className="min-h-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,#ffffff_0%,#f8faf9_100%)] px-4 py-5 md:px-6">
                {messages.length === 0 && (
                  <div className="mx-auto flex max-w-2xl flex-col items-center justify-center gap-4 py-16 text-center">
                    <div className="rounded-lg border border-teal-100 bg-teal-50 px-4 py-2 text-sm font-semibold text-teal-700">
                      Start a guided appointment conversation
                    </div>
                    <h4 className="text-2xl font-semibold text-zinc-950">
                      Ask about availability, appointments, or clinic related information and services.
                    </h4>
                    <p className="text-sm leading-6 text-zinc-500">
                      Try: I want to book an appointment tomorrow at 4 PM.
                    </p>
                  </div>
                )}

                <ul className="space-y-4">
                  {messages.map((m, i) => (
                    <li
                      key={i}
                      className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[85%] rounded-lg px-4 py-3 text-sm leading-6 shadow-sm ${
                          m.role === "user"
                            ? "bg-zinc-950 text-white shadow-zinc-950/15"
                            : "border border-zinc-200 bg-white text-zinc-900"
                        }`}
                      >
                        <span className="whitespace-pre-wrap break-words">{m.content}</span>
                      </div>
                    </li>
                  ))}
                </ul>

                {loading && (
                  <div className="mt-4 flex justify-start">
                    <div className="rounded-lg border border-zinc-200 bg-white px-4 py-3 shadow-sm">
                      <span className="text-sm text-zinc-500">Thinking...</span>
                    </div>
                  </div>
                )}
                <div ref={bottomRef} />
              </div>

              {error && (
                <div className="mx-4 mb-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 md:mx-6">
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit} className="shrink-0 border-t border-zinc-200 bg-white p-4 md:p-5">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    className="min-w-0 flex-1 rounded-lg border border-zinc-200 bg-zinc-50 px-4 py-3 text-zinc-900 outline-none transition placeholder:text-zinc-400 focus:border-teal-500 focus:ring-4 focus:ring-teal-100 disabled:cursor-not-allowed disabled:bg-zinc-100"
                    disabled={loading}
                  />
                  <button
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="rounded-lg bg-zinc-950 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-teal-700 disabled:translate-y-0 disabled:cursor-not-allowed disabled:bg-zinc-400"
                  >
                    Send
                  </button>
                </div>
              </form>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
