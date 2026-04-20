import type { InputHTMLAttributes, TextareaHTMLAttributes } from "react";

type TextInputProps = InputHTMLAttributes<HTMLInputElement> & {
  label?: string;
};

type TextAreaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & {
  label?: string;
};

const inputClass =
  "w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none transition focus:border-teal-500 focus:ring-4 focus:ring-teal-100 disabled:cursor-not-allowed disabled:bg-zinc-100";

export function Input({ label, className = "", ...props }: TextInputProps) {
  return (
    <label className="block">
      {label && <span className="mb-2 block text-sm font-medium text-zinc-700">{label}</span>}
      <input className={`${inputClass} ${className}`} {...props} />
    </label>
  );
}

export function TextArea({ label, className = "", ...props }: TextAreaProps) {
  return (
    <label className="block">
      {label && <span className="mb-2 block text-sm font-medium text-zinc-700">{label}</span>}
      <textarea
        className={`${inputClass} min-h-40 resize-y font-mono leading-6 ${className}`}
        {...props}
      />
    </label>
  );
}
