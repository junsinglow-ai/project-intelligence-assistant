import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertTriangle, Check, Copy, FileCheck2, RotateCcw, Sparkles } from "lucide-react";
import AgentBadge from "./AgentBadge.jsx";
import Citations from "./Citations.jsx";

const CITE_HREF = "#cite-";

// The backend renumbers citations so an `[n]` marker in the answer is
// citations[n - 1] (app/skills/evidence.py). Turn in-range markers into links
// the markdown renderer can swap for clickable chips; leave the rest as text.
function linkCitations(text, count) {
  return text.replace(/\[(\d+)\](?![(:])/g, (match, n) =>
    Number(n) >= 1 && Number(n) <= count ? `[${n}](${CITE_HREF}${n})` : match
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button className="icon-btn small" onClick={copy} title="Copy answer" aria-label="Copy answer">
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

function AssistantMessage({ message, agents }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [active, setActive] = useState(null);
  const citations = message.citations ?? [];
  const description = agents.find((a) => a.name === message.agent)?.description;

  const selectCitation = (n) => {
    setSourcesOpen(true);
    setActive(n);
  };

  const components = {
    a({ href, children, ...props }) {
      if (href?.startsWith(CITE_HREF)) {
        const n = Number(href.slice(CITE_HREF.length));
        const c = citations[n - 1];
        return (
          <button
            className={`cite-chip ${active === n ? "active" : ""}`}
            onClick={() => selectCitation(active === n ? null : n)}
            title={c ? [c.source, c.location].filter(Boolean).join(" · ") : undefined}
          >
            {n}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer" {...props}>
          {children}
        </a>
      );
    },
  };

  return (
    <div className="msg assistant">
      <div className="avatar" aria-hidden="true">
        <Sparkles size={16} />
      </div>
      <div className="msg-body">
        <div className="msg-meta">
          <AgentBadge name={message.agent} description={description} />
        </div>
        <div className="markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
            {linkCitations(message.content, citations.length)}
          </ReactMarkdown>
        </div>
        <Citations
          citations={citations}
          open={sourcesOpen}
          onToggle={() => setSourcesOpen((o) => !o)}
          active={active}
          onSelect={setActive}
        />
        <div className="msg-actions">
          <CopyButton text={message.content} />
          {message.traceId && (
            <span className="trace" title="Trace ID (matches the backend logs)">
              {message.traceId}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Message({ message, agents, onRetry }) {
  switch (message.role) {
    case "user":
      return (
        <div className="msg user">
          <div className="bubble">{message.content}</div>
        </div>
      );
    case "notice":
      return (
        <div className="notice">
          <FileCheck2 size={14} aria-hidden="true" />
          {message.content}
        </div>
      );
    case "error":
      return (
        <div className="msg assistant">
          <div className="avatar error" aria-hidden="true">
            <AlertTriangle size={16} />
          </div>
          <div className="msg-body">
            <div className="error-text">{message.content}</div>
            {message.retryQuestion && (
              <button className="retry-btn" onClick={() => onRetry(message)}>
                <RotateCcw size={14} aria-hidden="true" /> Retry
              </button>
            )}
          </div>
        </div>
      );
    default:
      return <AssistantMessage message={message} agents={agents} />;
  }
}
