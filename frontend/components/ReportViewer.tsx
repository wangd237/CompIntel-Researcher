import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function ReportViewer({ markdown }: { markdown: string }) {
  if (!markdown.trim()) {
    return <div className="rounded-md border border-dashed border-line p-4 text-sm text-slate-500">等待报告生成</div>;
  }

  return (
    <article className="prose prose-slate max-w-none rounded-md border border-line bg-white p-5 prose-a:text-blue-700 prose-table:text-sm">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p({ children }) {
            const text = String(children);
            if (text.includes("Data Gap") || text.includes("数据缺口")) {
              return <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-amber-900">{children}</p>;
            }
            return <p>{children}</p>;
          },
          a({ href, children }) {
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          }
        }}
      >
        {markdown}
      </ReactMarkdown>
    </article>
  );
}
