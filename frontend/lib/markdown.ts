import type { CompIntelResult, ReportShape } from "./types";

export function extractReport(result?: CompIntelResult): ReportShape | undefined {
  return result?.report?.report;
}

export function toMarkdown(result?: CompIntelResult): string {
  const report = extractReport(result);
  if (!report) return "";

  const lines = [
    `# ${report.title ?? "CompIntel Research Report"}`,
    "",
    "## Executive Summary",
    report.executive_summary ?? "",
    "",
    ...(report.sections ?? []).flatMap((section) => [
      `## ${section.title}`,
      section.content,
      "",
      ...(section.key_insights?.length ? ["Key insights:", ...section.key_insights.map((item) => `- ${item}`), ""] : [])
    ]),
    "## Data Gaps",
    ...(report.data_gaps?.length ? report.data_gaps.map((gap) => `- Data Gap: ${gap}`) : ["- No explicit data gaps."]),
    "",
    "## Sources",
    ...(report.sources?.length ? report.sources.map((source, index) => `${index + 1}. [${source}](${source})`) : ["- No traceable sources."]),
    "",
    "## Conclusion",
    report.conclusion ?? ""
  ];

  return lines.join("\n");
}
