export type AnalysisDepth = "brief" | "standard" | "deep";

export type AnalysisDimension = "product" | "pricing" | "market" | "technology";

export type CompIntelEvent = {
  type: string;
  phase?: string;
  message?: string;
  data?: Record<string, unknown>;
};

export type SwotItem = {
  text: string;
  evidence?: string;
};

export type CompetitorSwot = {
  name: string;
  strengths?: SwotItem[];
  weaknesses?: SwotItem[];
  opportunities?: SwotItem[];
  threats?: SwotItem[];
};

export type CompetitorProfile = {
  name: string;
  website?: string | null;
  summary?: string | null;
  evidence_grade?: string;
  sources?: string[];
  search_results?: Array<{ url?: string; title?: string; snippet?: string; source?: string }>;
  rag_context?: Array<{ source?: string; text?: string }>;
};

export type ReportShape = {
  title?: string;
  executive_summary?: string;
  sections?: Array<{
    title: string;
    content: string;
    key_insights?: string[];
  }>;
  swot_analysis?: {
    summary?: string;
    competitors?: CompetitorSwot[];
    cross_analysis?: Record<string, SwotItem[]>;
  };
  market_analysis?: Record<string, unknown>;
  sources?: string[];
  data_gaps?: string[];
  conclusion?: string;
};

export type CompIntelResult = {
  intent?: {
    target?: string;
    market_segment?: string;
  };
  competitors?: Array<{ name: string; website?: string | null; rationale?: string | null }>;
  profiles?: CompetitorProfile[];
  report?: {
    report?: ReportShape;
    market_analysis?: Record<string, unknown>;
    swot_analysis?: ReportShape["swot_analysis"];
    review_feedback?: Record<string, unknown>;
  };
};

export type FinalAnalysisMessage = {
  type: "analysis_ready";
  message?: string;
  data?: {
    mode?: "replay";
    event_count?: number;
    result?: CompIntelResult;
    report_path?: string;
    bundle_path?: string;
  };
};
