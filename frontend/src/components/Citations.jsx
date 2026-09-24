import { ChevronDown, Database, FileText } from "lucide-react";

const TABULAR = /\.(csv|xlsx|xlsm)$/i;

// Data-analysis citations carry the SQL that produced the figure as the snippet.
function looksLikeSql(snippet) {
  return /^\s*(with|select)\b/i.test(snippet ?? "");
}

export default function Citations({ citations, open, onToggle, active, onSelect }) {
  if (!citations?.length) return null;

  return (
    <div className="citations">
      <button className="citations-toggle" onClick={onToggle} aria-expanded={open}>
        <ChevronDown size={14} className={open ? "rotated" : ""} aria-hidden="true" />
        {citations.length} {citations.length === 1 ? "source" : "sources"}
      </button>

      {open && (
        <ol className="citation-list">
          {citations.map((c, i) => {
            const n = i + 1;
            const Icon = TABULAR.test(c.source) ? Database : FileText;
            const isActive = active === n;
            return (
              <li
                key={n}
                className={`citation ${isActive ? "active" : ""}`}
                data-cite={n}
              >
                <button
                  className="citation-head"
                  onClick={() => onSelect(isActive ? null : n)}
                  aria-expanded={isActive}
                >
                  <span className="citation-num">{n}</span>
                  <Icon size={14} aria-hidden="true" />
                  <span className="citation-source">{c.source}</span>
                  {c.location && <span className="citation-loc">{c.location}</span>}
                </button>
                {isActive && c.snippet && (
                  looksLikeSql(c.snippet) ? (
                    <pre className="citation-snippet sql"><code>{c.snippet}</code></pre>
                  ) : (
                    <blockquote className="citation-snippet">{c.snippet}</blockquote>
                  )
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
