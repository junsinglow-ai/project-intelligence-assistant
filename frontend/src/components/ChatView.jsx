import { useEffect, useRef, useState } from "react";
import { AlertTriangle, PanelLeft, Sparkles, Upload } from "lucide-react";
import Message from "./Message.jsx";
import TypingIndicator from "./TypingIndicator.jsx";
import Composer from "./Composer.jsx";
import AgentBadge from "./AgentBadge.jsx";

const SUGGESTIONS = [
  "Summarise the latest project status report",
  "What is the budget variance by quarter?",
  "Which open risks are rated high?",
  "What can you help me with?",
];

function EmptyState({ agents, onPick }) {
  return (
    <div className="empty">
      <div className="empty-logo" aria-hidden="true">
        <Sparkles size={28} />
      </div>
      <h1>Project Intelligence Assistant</h1>
      <p className="muted">
        Ask about status reports, financials and risk registers. Attach documents with the
        paperclip or drop them anywhere here.
      </p>
      {agents.length > 0 && (
        <div className="agent-row">
          {agents.map((a) => (
            <AgentBadge key={a.name} name={a.name} description={a.description} />
          ))}
        </div>
      )}
      <div className="suggestions">
        {SUGGESTIONS.map((s) => (
          <button key={s} className="suggestion" onClick={() => onPick(s)}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ChatView({
  conversation,
  pending,
  agents,
  backendError,
  onSend,
  onRetry,
  uploads,
  onToggleSidebar,
}) {
  const scrollRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  // dragenter/dragleave fire for every child; count them to know when we left.
  const dragDepth = useRef(0);
  const messages = conversation?.messages ?? [];

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages.length, pending]);

  const hasFiles = (e) => Array.from(e.dataTransfer?.types ?? []).includes("Files");

  const dropHandlers = {
    onDragEnter(e) {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragOver(e) {
      if (hasFiles(e)) e.preventDefault();
    },
    onDragLeave(e) {
      if (!hasFiles(e)) return;
      dragDepth.current -= 1;
      if (dragDepth.current <= 0) setDragging(false);
    },
    onDrop(e) {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      uploads.addFiles(e.dataTransfer.files);
    },
  };

  return (
    <main className="chat" {...dropHandlers}>
      <header className="chat-header">
        <button className="icon-btn" onClick={onToggleSidebar} aria-label="Toggle sidebar">
          <PanelLeft size={18} />
        </button>
        <span className="chat-title">{conversation?.title ?? "New chat"}</span>
      </header>

      {backendError && (
        <div className="banner" role="alert">
          <AlertTriangle size={16} aria-hidden="true" />
          Backend unavailable: {backendError}
        </div>
      )}

      <div className="scroll" ref={scrollRef}>
        <div className="column">
          {messages.length === 0 && !pending ? (
            <EmptyState agents={agents} onPick={onSend} />
          ) : (
            messages.map((m) => (
              <Message key={m.id} message={m} agents={agents} onRetry={onRetry} />
            ))
          )}
          {pending && <TypingIndicator />}
        </div>
      </div>

      <Composer
        onSend={onSend}
        disabled={pending}
        attachments={uploads.attachments}
        onFiles={uploads.addFiles}
        onDismiss={uploads.dismiss}
      />

      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          <Upload size={32} />
          <p>Drop PDF, CSV or Excel files to index them</p>
        </div>
      )}
    </main>
  );
}
