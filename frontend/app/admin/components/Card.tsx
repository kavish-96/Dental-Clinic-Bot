import type { ReactNode } from "react";

type CardProps = {
  children: ReactNode;
  className?: string;
};

export function Card({ children, className = "" }: CardProps) {
  return (
    <section
      className={`rounded-lg border border-zinc-200 bg-white shadow-[0_18px_45px_rgba(30,41,59,0.08)] transition-shadow duration-200 hover:shadow-[0_22px_55px_rgba(30,41,59,0.12)] ${className}`}
    >
      {children}
    </section>
  );
}
