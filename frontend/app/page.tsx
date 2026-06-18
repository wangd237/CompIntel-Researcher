"use client";

import { AlertTriangle, CheckCircle2, FileArchive, Server, XCircle } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { ComparisonTable } from "@/components/ComparisonTable";
import { CompIntelInput } from "@/components/CompIntelInput";
import { PipelineProgress } from "@/components/PipelineProgress";
import { ReportViewer } from "@/components/ReportViewer";
import { SWOTMatrix } from "@/components/SWOTMatrix";
import { buildQuery, runCompIntelAnalysis } from "@/lib/compintel";
import { extractReport, toMarkdown } from "@/lib/markdown";
import type {
  AnalysisDepth,
  AnalysisDimension,
  CompIntelEvent,
  CompIntelResult,
  FinalAnalysisMessage
} from "@/lib/types";

export default function HomePage() {
  const [competitors, setCompetitors] = useState("Notion, Coda");
  const [dimensions, setDimensions] = useState<AnalysisDimension[]>([
    "product",
    "pricing",
    "market",
    "technology"
  ]);
  const [depth, setDepth] = useState<AnalysisDepth>("standard");
  const [events, setEvents] = useState<CompIntelEvent[]>([]);
  const [result, setResult] = useState<CompIntelResult | undefined>();
  const [bundlePath, setBundlePath] = useState("");
  const [reportPath, setReportPath] = useState("");
  const [status, setStatus] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [error, setError] = useState("");
  const socketRef = useRef<WebSocket | null>(null);

  const markdown = useMemo(() => toMarkdown(result), [result]);
  const swot = extractReport(result)?.swot_analysis ?? result?.report?.swot_analysis;
  const query = useMemo(() => buildQuery(competitors, dimensions, depth), [competitors, depth, dimensions]);

  function handleSubmit() {
    socketRef.current?.close();
    setEvents([]);
    setResult(undefined);
    setBundlePath("");
    setReportPath("");
    setError("");
    setStatus("running");

    socketRef.current = runCompIntelAnalysis(query, {
      onReplayEvent: (event) => {
        window.setTimeout(() => {
          setEvents((current) => [...current, event]);
        }, 180);
      },
      onComplete: (message: FinalAnalysisMessage) => {
        window.setTimeout(() => {
          setResult(message.data?.result);
          setBundlePath(message.data?.bundle_path ?? "");
          setReportPath(message.data?.report_path ?? "");
          setStatus("complete");
        }, 220);
      },
      onError: (message) => {
        setError(message);
        setStatus("error");
      }
    });
  }

  const isRunning = status === "running";

  return (
    <main className="min-h-screen bg-panel">
      <CompIntelInput
        competitors={competitors}
        dimensions={dimensions}
        depth={depth}
        isRunning={isRunning}
        onCompetitorsChange={setCompetitors}
        onDepthChange={setDepth}
        onDimensionsChange={setDimensions}
        onSubmit={handleSubmit}
      />

      <div className="mx-auto grid max-w-7xl grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)]">
        <PipelineProgress events={events} isRunning={isRunning} />

        <section className="min-w-0 px-6 py-5">
          <div className="mb-4 grid gap-3 md:grid-cols-3">
            <StatusCard status={status} eventCount={events.length} />
            <PathCard icon={<FileArchive className="h-4 w-4" />} label="Bundle" value={bundlePath || "等待生成"} />
            <PathCard icon={<Server className="h-4 w-4" />} label="Report" value={reportPath || "等待生成"} />
          </div>

          {error ? (
            <div className="mb-4 flex items-start gap-2 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
              <XCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          ) : null}

          <div className="mb-5 rounded-md border border-line bg-white p-4">
            <div className="mb-1 text-xs font-semibold uppercase text-slate-500">当前查询</div>
            <div className="text-sm text-slate-800">{query}</div>
          </div>

          <div className="grid gap-5">
            <Panel title="SWOT 四象限">
              <SWOTMatrix competitors={swot?.competitors ?? []} />
            </Panel>
            <Panel title="竞品对比">
              <ComparisonTable profiles={result?.profiles ?? []} />
            </Panel>
            <Panel title="研究报告">
              <ReportViewer markdown={markdown} />
            </Panel>
          </div>
        </section>
      </div>
    </main>
  );
}

function StatusCard({ status, eventCount }: { status: string; eventCount: number }) {
  const copy = {
    idle: ["待开始", "连接后端后运行分析"],
    running: ["运行中", `已回放 ${eventCount} 个事件`],
    complete: ["已完成", `共回放 ${eventCount} 个事件`],
    error: ["失败", "请检查后端服务或 API Key"]
  }[status] ?? ["待开始", ""];

  return (
    <div className="rounded-md border border-line bg-white p-3">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-900">
        {status === "complete" ? <CheckCircle2 className="h-4 w-4 text-emerald-600" /> : <AlertTriangle className="h-4 w-4 text-blue-600" />}
        {copy[0]}
      </div>
      <div className="text-xs text-slate-500">{copy[1]}</div>
    </div>
  );
}

function PathCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-line bg-white p-3">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-900">
        {icon}
        {label}
      </div>
      <div className="truncate text-xs text-slate-500" title={value}>
        {value}
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="mb-3 text-base font-semibold text-slate-900">{title}</h2>
      {children}
    </section>
  );
}
