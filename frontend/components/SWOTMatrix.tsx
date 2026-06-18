import type { CompetitorSwot, SwotItem } from "@/lib/types";

const quadrants: Array<{ key: keyof CompetitorSwot; label: string; className: string }> = [
  { key: "strengths", label: "S Strengths", className: "border-emerald-200 bg-emerald-50" },
  { key: "weaknesses", label: "W Weaknesses", className: "border-rose-200 bg-rose-50" },
  { key: "opportunities", label: "O Opportunities", className: "border-blue-200 bg-blue-50" },
  { key: "threats", label: "T Threats", className: "border-amber-200 bg-amber-50" }
];

function SwotList({ items }: { items?: SwotItem[] }) {
  if (!items?.length) {
    return <div className="text-sm text-slate-500">No evidence yet.</div>;
  }
  return (
    <ul className="space-y-2">
      {items.map((item, index) => (
        <li key={`${item.text}-${index}`} className="text-sm text-slate-800">
          <div>{item.text}</div>
          {item.evidence ? <div className="mt-1 pl-3 text-xs text-slate-500">Evidence: {item.evidence}</div> : null}
        </li>
      ))}
    </ul>
  );
}

export function SWOTMatrix({ competitors }: { competitors: CompetitorSwot[] }) {
  if (!competitors.length) {
    return <div className="rounded-md border border-dashed border-line p-4 text-sm text-slate-500">等待 SWOT 结果</div>;
  }

  return (
    <div className="grid gap-4">
      {competitors.map((competitor) => (
        <article key={competitor.name} className="rounded-md border border-line bg-white">
          <div className="border-b border-line px-4 py-3 font-semibold text-slate-900">{competitor.name}</div>
          <div className="grid gap-3 p-4 md:grid-cols-2">
            {quadrants.map((quadrant) => (
              <section key={quadrant.key} className={`min-h-36 rounded-md border p-3 ${quadrant.className}`}>
                <div className="mb-2 text-sm font-semibold text-slate-800">{quadrant.label}</div>
                <SwotList items={competitor[quadrant.key] as SwotItem[] | undefined} />
              </section>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}
