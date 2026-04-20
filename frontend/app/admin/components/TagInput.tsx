"use client";

import { useState } from "react";

type TagInputProps = {
  values: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
};

export function TagInput({ values, onChange, placeholder = "Add item" }: TagInputProps) {
  const [draft, setDraft] = useState("");

  function addValue(rawValue = draft) {
    const next = rawValue.trim();
    if (!next) return;
    const exists = values.some((value) => value.toLowerCase() === next.toLowerCase());
    if (!exists) {
      onChange([...values, next]);
    }
    setDraft("");
  }

  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-2 transition focus-within:border-teal-500 focus-within:ring-4 focus-within:ring-teal-100">
      <div className="flex flex-wrap gap-2">
        {values.map((value) => (
          <span
            key={value}
            className="inline-flex items-center gap-2 rounded-lg bg-white px-3 py-1.5 text-sm text-zinc-800 shadow-sm ring-1 ring-zinc-200"
          >
            {value}
            <button
              type="button"
              onClick={() => onChange(values.filter((item) => item !== value))}
              className="rounded text-zinc-400 transition hover:text-rose-600"
              aria-label={`Remove ${value}`}
            >
              x
            </button>
          </span>
        ))}
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === ",") {
              event.preventDefault();
              addValue();
            }
            if (event.key === "Backspace" && !draft && values.length) {
              onChange(values.slice(0, -1));
            }
          }}
          onBlur={() => addValue()}
          placeholder={placeholder}
          className="min-w-36 flex-1 bg-transparent px-2 py-1.5 text-sm text-zinc-900 outline-none placeholder:text-zinc-400"
        />
      </div>
    </div>
  );
}
