"use client";

import { ArrowDownUp } from "lucide-react";
import { useMemo, useState } from "react";
import type { CompetitorProfile } from "@/lib/types";

type SortKey = "name" | "sources";

export function ComparisonTable({ profiles }: { profiles: CompetitorProfile[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("name");

  const rows = useMemo(() => {
    return [...profiles].sort((a, b) => {
      if (sortKey === "sources") {
        return sourceCount(b) - sourceCount(a);
      }
      return (a.name ?? "").localeCompare(b.name ?? "");
    });
  }, [profiles, sortKey]);

  if (!rows.length) {
    return <div className="rounded-md border border-dashed border-line p-4 text-sm text-slate-500">等待竞品画像</div>;
  }

  return (
    <div className="overflow-x-auto rounded-md border border-line bg-white">
      <table className="min-w-full text-sm">
        <thead className="bg-panel text-left text-slate-600">
          <tr>
            <th className="px-3 py-2">
              <button className="inline-flex items-center gap-1 font-semibold" onClick={() => setSortKey("name")}>
                竞品 <ArrowDownUp className="h-3 w-3" />
              </button>
            </th>
            <th className="px-3 py-2">摘要</th>
            <th className="px-3 py-2">证据等级</th>
            <th className="px-3 py-2 text-right">
              <button className="inline-flex items-center gap-1 font-semibold" onClick={() => setSortKey("sources")}>
                来源数 <ArrowDownUp className="h-3 w-3" />
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((profile) => (
            <tr key={profile.name} className="border-t border-line">
              <td className="px-3 py-3 font-semibold text-slate-900">{profile.name}</td>
              <td className="px-3 py-3 text-slate-700 max-w-xs truncate" title={profile.summary ?? ""}>
                {profile.summary ?? "待补充"}
              </td>
              <td className="px-3 py-3">{evidenceBadge(profile.evidence_grade)}</td>
              <td className="px-3 py-3 text-right text-slate-700">{sourceCount(profile)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function sourceCount(profile: CompetitorProfile): number {
  return (
    (profile.sources?.length ?? 0) +
    (profile.search_results?.length ?? 0) +
    (profile.rag_context?.length ?? 0)
  );
}

function evidenceBadge(grade?: string) {
  const map: Record<string, { label: string; cls: string }> = {
    rich: { label: "丰富", cls: "bg-emerald-100 text-emerald-800" },
    adequate: { label: "充足", cls: "bg-blue-100 text-blue-800" },
    thin: { label: "稀疏", cls: "bg-amber-100 text-amber-800" },
    empty: { label: "空", cls: "bg-rose-100 text-rose-800" },
  };
  const entry = map[grade ?? ""] ?? { label: grade ?? "未知", cls: "bg-slate-100 text-slate-600" };
  return (
    <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${entry.cls}`}>
      {entry.label}
    </span>
  );
}
