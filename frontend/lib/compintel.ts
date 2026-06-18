import type { CompIntelEvent, FinalAnalysisMessage } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_COMPINTEL_API_BASE ?? "http://localhost:8000";

export function websocketUrl(): string {
  const base = API_BASE.replace(/\/$/, "");
  if (base.startsWith("https://")) {
    return `wss://${base.slice("https://".length)}/ws/compintel`;
  }
  if (base.startsWith("http://")) {
    return `ws://${base.slice("http://".length)}/ws/compintel`;
  }
  return `${base}/ws/compintel`;
}

export type RunAnalysisCallbacks = {
  onReplayEvent: (event: CompIntelEvent) => void;
  onComplete: (message: FinalAnalysisMessage) => void;
  onError: (message: string) => void;
};

export function runCompIntelAnalysis(query: string, callbacks: RunAnalysisCallbacks): WebSocket {
  const socket = new WebSocket(websocketUrl());

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ query }));
  });

  socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data) as CompIntelEvent | FinalAnalysisMessage;
      if (payload.type === "analysis_ready" && "data" in payload && payload.data?.result) {
        callbacks.onComplete(payload as FinalAnalysisMessage);
        socket.close();
        return;
      }
      if (payload.type === "execution_failed") {
        callbacks.onError(payload.message ?? "Analysis failed");
        socket.close();
        return;
      }
      callbacks.onReplayEvent(payload as CompIntelEvent);
    } catch (error) {
      callbacks.onError(error instanceof Error ? error.message : "Invalid WebSocket message");
      socket.close();
    }
  });

  socket.addEventListener("error", () => {
    callbacks.onError("WebSocket connection failed. Confirm the FastAPI service is running.");
  });

  return socket;
}

export function buildQuery(competitors: string, dimensions: string[], depth: string): string {
  const names = competitors
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .join("、");
  const dimensionText = dimensions.length ? `，重点关注${dimensions.join("、")}` : "";
  const depthText = depth === "deep" ? "深度调研" : depth === "brief" ? "快速简报" : "标准分析";
  return `分析 ${names || "Notion"} 的竞品格局，输出${depthText}${dimensionText}`;
}
