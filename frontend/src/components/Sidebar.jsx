import { Database, FileText, MessageSquare, SquarePen, Trash2 } from "lucide-react";

const TABULAR = /\.(csv|xlsx|xlsm)$/i;

export default function Sidebar({
  open,
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  documents,
  status,
  onClose,
}) {
  return (
    <>
      <aside className={`sidebar ${open ? "open" : "closed"}`} aria-hidden={!open}>
        <div className="sidebar-top">
          <div className="brand">Project Intelligence</div>
          <button className="new-chat" onClick={onNew}>
            <SquarePen size={16} aria-hidden="true" /> New chat
          </button>
        </div>

        <nav className="sidebar-section conversations" aria-label="Conversations">
          <div className="section-label">Chats</div>
          {conversations.length === 0 && <div className="muted small pad">No chats yet</div>}
          {conversations.map((c) => (
            <div key={c.id} className={`conv ${c.id === activeId ? "active" : ""}`}>
              <button className="conv-title" onClick={() => onSelect(c.id)} title={c.title}>
                <MessageSquare size={14} aria-hidden="true" />
                <span>{c.title}</span>
              </button>
              <button
                className="icon-btn tiny conv-delete"
                onClick={() => onDelete(c.id)}
                aria-label={`Delete ${c.title}`}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </nav>

        {documents.length > 0 && (
          <div className="sidebar-section documents">
            <div className="section-label">Uploaded documents</div>
            {documents.map((d) => {
              const Icon = TABULAR.test(d.filename) ? Database : FileText;
              return (
                <div key={d.filename} className="doc" title={`${d.chunks_indexed} chunks · ${d.doc_type}`}>
                  <Icon size={14} aria-hidden="true" />
                  <span>{d.filename}</span>
                </div>
              );
            })}
          </div>
        )}

        <div className="sidebar-footer">
          <span className={`status-dot ${status}`} aria-hidden="true" />
          {status === "ok" ? "Backend connected" : status === "down" ? "Backend offline" : "Connecting…"}
        </div>
      </aside>
      {open && <div className="scrim" onClick={onClose} aria-hidden="true" />}
    </>
  );
}
