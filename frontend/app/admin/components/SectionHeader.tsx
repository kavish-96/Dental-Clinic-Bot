type SectionHeaderProps = {
  title: string;
  eyebrow?: string;
  description?: string;
  action?: React.ReactNode;
};

export function SectionHeader({ title, eyebrow, description, action }: SectionHeaderProps) {
  return (
    <div className="flex flex-col gap-4 border-b border-zinc-100 p-5 sm:flex-row sm:items-start sm:justify-between">
      <div>
        {eyebrow && (
          <p className="text-xs font-semibold uppercase text-teal-700">{eyebrow}</p>
        )}
        <h2 className="mt-1 text-xl font-semibold text-zinc-950">{title}</h2>
        {description && <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-500">{description}</p>}
      </div>
      {action}
    </div>
  );
}
