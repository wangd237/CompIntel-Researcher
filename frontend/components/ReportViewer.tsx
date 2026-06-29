import { ExternalLink } from "lucide-react";

export function ReportViewer({
  reportUrl,
}: {
  markdown?: string;
  reportUrl?: string;
  reportTitle?: string;
}) {
  if (!reportUrl) {
    return (
      <div className="rounded-md border border-dashed border-line p-4 text-sm text-slate-500">
        Waiting for report...
      </div>
    );
  }

  return (
    <div className="rounded-md border border-line bg-white p-5 text-center">
      <p className="mb-3 text-sm text-slate-600">
        The competitive intelligence report has been generated.
      </p>
      <a
        href={reportUrl}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-2 rounded-md bg-ink px-6 py-2.5 text-sm font-medium text-white hover:bg-slate-800 transition-colors"
      >
        <ExternalLink className="h-4 w-4" />
        Open Full Report
      </a>
    </div>
  );
}
