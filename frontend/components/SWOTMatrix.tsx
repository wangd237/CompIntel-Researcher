"use client";

import type { CompetitorSwot, SwotItem } from "@/lib/types";

const quadrants: Array<{ key: keyof CompetitorSwot; label: string; color: string }> = [
  { key: "strengths", label: "Strengths", color: "bg-emerald-50 border-emerald-300" },
  { key: "weaknesses", label: "Weaknesses", color: "bg-rose-50 border-rose-300" },
  { key: "opportunities", label: "Opportunities", color: "bg-blue-50 border-blue-300" },
  { key: "threats", label: "Threats", color: "bg-amber-50 border-amber-300" },
];

function SwotList({ items }: { items?: SwotItem[] }) {
  if (!items?.length) {
    return <div className="text-xs italic text-slate-400">No data</div>;
  }
  return (
    <ul className="list-disc pl-4 space-y-1">
      {items.map((item, index) => (
        <li key={index} className="text-sm text-slate-700">{item.text}</li>
      ))}
    </ul>
  );
}

export function SWOTMatrix({ competitors }: { competitors: CompetitorSwot[] }) {
  if (!competitors.length) {
    return (
      <div className="rounded-md border border-dashed border-line p-4 text-sm text-slate-500">
        Waiting for SWOT results...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {competitors.map((competitor) => (
        <div key={competitor.name}>
          <h3 className="mb-3 text-base font-semibold text-slate-900">{competitor.name}</h3>
          <div className="grid grid-cols-2 gap-3">
            {quadrants.map((q) => (
              <div
                key={q.key}
                className={`rounded-lg border-2 p-4 ${q.color}`}
              >
                <div className="mb-2 text-sm font-bold text-slate-800 uppercase tracking-wide">{q.label}</div>
                <SwotList items={competitor[q.key] as SwotItem[] | undefined} />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
