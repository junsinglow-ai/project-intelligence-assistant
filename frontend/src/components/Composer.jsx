import { useLayoutEffect, useRef, useState } from "react";
import { ArrowUp, FileText, Loader2, Paperclip, X } from "lucide-react";
import { UPLOAD_SUFFIXES } from "../api/client.js";

const MAX_ROWS_PX = 200;

function AttachmentChip({ item, onDismiss }) {
  const detail =
    item.status === "uploading"
      ? item.progress < 1
        ? `${Math.round(item.progress * 100)}%`
        : "Indexing…"
      : item.status === "done"
        ? `${item.result.chunks_indexed} chunks · ${item.result.doc_type}`
        : item.error;

  return (
    <div className={`attachment ${item.status}`} title={detail}>
      {item.status === "uploading" ? (
        <Loader2 size={14} className="spin" aria-hidden="true" />
      ) : (
        <FileText size={14} aria-hidden="true" />
      )}
      <span className="attachment-name">{item.name}</span>
      <span className="attachment-detail">{detail}</span>
      {item.status !== "uploading" && (
        <button className="icon-btn tiny" onClick={() => onDismiss(item.id)} aria-label="Dismiss">
          <X size={12} />
        </button>
      )}
      {item.status === "uploading" && (
        <span className="attachment-bar" style={{ width: `${item.progress * 100}%` }} />
      )}
    </div>
  );
}

export default function Composer({ onSend, disabled, attachments, onFiles, onDismiss }) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);
  const fileRef = useRef(null);

  // Grow with the content up to a cap, then scroll.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_ROWS_PX)}px`;
  }, [text]);

  const canSend = text.trim().length > 0 && !disabled;

  const submit = () => {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        {attachments.length > 0 && (
          <div className="attachments">
            {attachments.map((a) => (
              <AttachmentChip key={a.id} item={a} onDismiss={onDismiss} />
            ))}
          </div>
        )}
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about your project documents…"
          aria-label="Message"
          maxLength={4000}
        />
        <div className="composer-row">
          <button
            className="icon-btn"
            onClick={() => fileRef.current?.click()}
            title={`Upload documents (${UPLOAD_SUFFIXES.join(", ")})`}
            aria-label="Upload documents"
          >
            <Paperclip size={18} />
          </button>
          <input
            ref={fileRef}
            type="file"
            hidden
            multiple
            accept={UPLOAD_SUFFIXES.join(",")}
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <span className="composer-hint">PDF, CSV or Excel · Enter to send, Shift+Enter for a new line</span>
          <button className="send-btn" onClick={submit} disabled={!canSend} aria-label="Send">
            <ArrowUp size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
