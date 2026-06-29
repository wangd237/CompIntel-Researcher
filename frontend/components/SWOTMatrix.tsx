"use client";

import { useState } from "react";
import type { CompetitorSwot, SwotItem } from "@/lib/types";

const quadrants: Array<{ key: keyof CompetitorSwot; label: string; color: string }> = [
  { key: "strengths", label: "Strengths", color: "bg-emerald-50 border-emerald-300" },
  { key: "weaknesses", label: "Weaknesses", color: "bg-rose-50 border-rose-300" },
  { key: "opportunities", label: "Opportunities", color: "bg-blue-50 border-blue-300" },
  { key: "threats", label: "Threats", color: "bg-amber-50 border-amber-300" },
];

function SwotList({ items }: { items?: SwotItem[] }) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  if (!items?.length) {
    return <div className="text-xs italic text-slate-400">No data</div>;
  }

  return (
    <ul className="space-y-1.5">
      {items.map((item, index) => {
        const hasEvidence = item.evidence && item.evidence.trim().length > 0;
        const isExpanded = expanded[index] ?? false;
        return (
          <li key={index}>
            <div className="text-sm text-slate-700">{item.text}</div>
            {hasEvidence && (
              <>
                <div className="mt-0.5 text-xs text-slate-400 truncate max-w-full">
                  {isExpanded ? (
                    <span>{item.evidence}</span>
                  ) : (
                    <span>{item.evidence!.slice(0, 80)}{item.evidence!.length > 80 ? "…" : ""}</span>
                  )}
                </div>
                {item.evidence!.length > 80 && (
                  <button
                    type="button"
                    className="mt-0.5 text-xs text-blue-600 hover:text-blue-800"
                    onClick={() => setExpanded((prev) => ({ ...prev, [index]: !prev[index] }))}
                  >
                    {isExpanded ? "Show less" : "Show evidence"}
                  </button>
                )}
              </>
            )}
          </li>
        );
      })}
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
