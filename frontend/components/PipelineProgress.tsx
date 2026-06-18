"use client";

import { Check, Circle, Loader2 } from "lucide-react";
import type { CompIntelEvent } from "@/lib/types";

const stages = [
  { id: "intent_analyst", label: "意图解析" },
  { id: "research_planner", label: "研究计划" },
  { id: "competitor_profiler", label: "竞品画像" },
  { id: "market_analyst", label: "市场分析" },
  { id: "swot_synthesizer", label: "SWOT" },
  { id: "report_writer", label: "报告生成" },
  { id: "reviewer", label: "质量审核" }
];

type Props = {
  events: CompIntelEvent[];
  isRunning: boolean;
};

export function PipelineProgress({ events, isRunning }: Props) {
  const completed = new Set(events.map((event) => event.phase).filter(Boolean));
  const currentPhase = events.at(-1)?.phase;

  return (
    <section className="h-full border-r border-line bg-white">
      <div className="border-b border-line px-4 py-3">
        <div className="text-sm font-semibold text-slate-900">Pipeline</div>
        <div className="text-xs text-slate-500">完成后事件回放</div>
      </div>
      <div className="space-y-3 p-4">
        {stages.map((stage) => {
          const done = completed.has(stage.id);
          const active = currentPhase === stage.id && isRunning;
          return (
            <div key={stage.id} className="flex items-center gap-3">
              <div className={`flex h-7 w-7 items-center justify-center rounded-full border ${
                done ? "border-emerald-500 bg-emerald-50 text-emerald-700" : "border-line bg-panel text-slate-400"
              }`}>
                {active ? <Loader2 className="h-4 w-4 animate-spin" /> : done ? <Check className="h-4 w-4" /> : <Circle className="h-3 w-3" />}
              </div>
              <div className="min-w-0">
                <div className="text-sm font-medium text-slate-800">{stage.label}</div>
                <div className="text-xs text-slate-500">{stage.id}</div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="border-t border-line p-4">
        <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">事件日志</div>
        <div className="max-h-72 space-y-2 overflow-y-auto text-xs">
          {events.length === 0 ? (
            <div className="rounded border border-dashed border-line p-3 text-slate-500">暂无事件</div>
          ) : (
            events.map((event, index) => (
              <div key={`${event.type}-${index}`} className="rounded border border-line bg-panel p-2">
                <div className="font-semibold text-slate-800">{event.type}</div>
                <div className="text-slate-500">{event.message ?? event.phase ?? "event"}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
