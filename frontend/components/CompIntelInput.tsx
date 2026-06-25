"use client";

import { BarChart3, Boxes, Cpu, DollarSign, FileSearch, Play } from "lucide-react";
import type { AnalysisDepth, AnalysisDimension } from "@/lib/types";

const dimensionOptions: Array<{ id: AnalysisDimension; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { id: "product", label: "产品功能", icon: Boxes },
  { id: "pricing", label: "定价", icon: DollarSign },
  { id: "market", label: "市场策略", icon: BarChart3 },
  { id: "technology", label: "技术架构", icon: Cpu }
];

const depthOptions: Array<{ id: AnalysisDepth; label: string }> = [
  { id: "brief", label: "快速简报" },
  { id: "standard", label: "标准分析" },
  { id: "deep", label: "深度调研" }
];

type Props = {
  competitors: string;
  dimensions: AnalysisDimension[];
  depth: AnalysisDepth;
  isRunning: boolean;
  onCompetitorsChange: (value: string) => void;
  onDimensionsChange: (value: AnalysisDimension[]) => void;
  onDepthChange: (value: AnalysisDepth) => void;
  onSubmit: () => void;
};

export function CompIntelInput({
  competitors,
  dimensions,
  depth,
  isRunning,
  onCompetitorsChange,
  onDimensionsChange,
  onDepthChange,
  onSubmit
}: Props) {
  function toggleDimension(id: AnalysisDimension) {
    onDimensionsChange(
      dimensions.includes(id)
        ? dimensions.filter((item) => item !== id)
        : [...dimensions, id]
    );
  }

  return (
    <section className="border-b border-line bg-white">
      <div className="mx-auto grid max-w-7xl gap-5 px-6 py-5 lg:grid-cols-[minmax(360px,1fr)_460px]">
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-600">
            <FileSearch className="h-4 w-4" />
            CompIntel Research
          </div>
          <textarea
            value={competitors}
            onChange={(event) => onCompetitorsChange(event.target.value)}
            className="min-h-28 w-full resize-none rounded-md border border-line bg-panel px-3 py-3 text-base outline-none transition focus:border-blue-500 focus:shadow-focus placeholder:text-slate-400 placeholder:italic"
            placeholder="输入竞品名称，如：Notion, Coda&#10;每行一个，或以逗号分隔"
          />
        </div>

        <div className="grid gap-4">
          <div>
            <div className="mb-2 text-sm font-semibold text-slate-700">分析维度</div>
            <div className="grid grid-cols-2 gap-2">
              {dimensionOptions.map((option) => {
                const Icon = option.icon;
                const active = dimensions.includes(option.id);
                return (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => toggleDimension(option.id)}
                    className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium transition ${
                      active
                        ? "border-blue-500 bg-blue-50 text-blue-700"
                        : "border-line bg-white text-slate-700 hover:bg-panel"
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="mb-2 text-sm font-semibold text-slate-700">报告深度</div>
            <div className="grid grid-cols-3 rounded-md border border-line bg-panel p-1">
              {depthOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => onDepthChange(option.id)}
                  className={`rounded px-2 py-2 text-sm font-medium transition ${
                    depth === option.id ? "bg-white text-blue-700 shadow-sm" : "text-slate-600"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <button
            type="button"
            onClick={onSubmit}
            disabled={isRunning}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-ink px-4 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            <Play className="h-4 w-4" />
            {isRunning ? "分析中" : "开始分析"}
          </button>
        </div>
      </div>
    </section>
  );
}
