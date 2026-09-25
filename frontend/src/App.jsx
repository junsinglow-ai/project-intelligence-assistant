import { useCallback, useEffect, useState } from "react";
import { chat, getHealth, listAgents } from "./api/client.js";
import { load, newId, save } from "./lib/storage.js";
import { useUploads } from "./hooks/useUploads.js";
import Sidebar from "./components/Sidebar.jsx";
import ChatView from "./components/ChatView.jsx";

const CONVERSATIONS_KEY = "pia.conversations";
const DOCUMENTS_KEY = "pia.documents";
const HEALTH_POLL_MS = 30000;
const NARROW = "(max-width: 860px)";
const TITLE_LENGTH = 48;

function isNarrow() {
  return typeof window !== "undefined" && window.matchMedia(NARROW).matches;
}

function titleFrom(question) {
  const oneLine = question.replace(/\s+/g, " ").trim();
  return oneLine.length > TITLE_LENGTH ? `${oneLine.slice(0, TITLE_LENGTH)}…` : oneLine;
}

function newConversation() {
  return { id: newId(), title: "New chat", titled: false, sessionId: null, messages: [], updatedAt: Date.now() };
}

export default function App() {
  // Conversations live in the browser: the server's session history has no
  // citations, and a chat should reopen exactly as it was answered.
  const [conversations, setConversations] = useState(() => load(CONVERSATIONS_KEY, []));
  const [activeId, setActiveId] = useState(() => conversations[0]?.id ?? null);
  const [documents, setDocuments] = useState(() => load(DOCUMENTS_KEY, []));
  const [pending, setPending] = useState({});
  const [agents, setAgents] = useState([]);
  const [status, setStatus] = useState("checking");
  const [backendError, setBackendError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => !isNarrow());

  useEffect(() => save(CONVERSATIONS_KEY, conversations), [conversations]);
  useEffect(() => save(DOCUMENTS_KEY, documents), [documents]);

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      getHealth()
        .then(() => (agents.length ? null : listAgents()))
        .then((list) => {
          if (cancelled) return;
          if (list) setAgents(list);
          setStatus("ok");
          setBackendError(null);
        })
        .catch((e) => {
          if (cancelled) return;
          setStatus("down");
          setBackendError(e.message);
        });
    check();
    const timer = setInterval(check, HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [agents.length]);

  const active = conversations.find((c) => c.id === activeId) ?? null;

  const updateConversation = useCallback((id, fn) => {
    setConversations((list) => {
      const updated = list.map((c) => (c.id === id ? { ...fn(c), updatedAt: Date.now() } : c));
      // Most recently touched first, like any chat client.
      return updated.sort((a, b) => b.updatedAt - a.updatedAt);
    });
  }, []);

  // The active conversation, created on first use so an empty "New chat" is
  // never persisted.
  const ensureConversation = useCallback(() => {
    if (active) return active.id;
    const conv = newConversation();
    setConversations((list) => [conv, ...list]);
    setActiveId(conv.id);
    return conv.id;
  }, [active]);

  const appendMessage = useCallback(
    (id, message) => updateConversation(id, (c) => ({ ...c, messages: [...c.messages, { id: newId(), ...message }] })),
    [updateConversation]
  );

  const ask = useCallback(
    async (convId, question, sessionId) => {
      setPending((p) => ({ ...p, [convId]: true }));
      try {
        const res = await chat(question, sessionId);
        updateConversation(convId, (c) => ({
          ...c,
          sessionId: res.session_id,
          messages: [
            ...c.messages,
            {
              id: newId(),
              role: "assistant",
              content: res.answer,
              agent: res.agent,
              model: res.model,
              citations: res.citations,
              traceId: res.trace_id,
            },
          ],
        }));
      } catch (err) {
        appendMessage(convId, { role: "error", content: err.message, retryQuestion: question });
      } finally {
        setPending((p) => {
          const { [convId]: _, ...rest } = p;
          return rest;
        });
      }
    },
    [updateConversation, appendMessage]
  );

  const send = useCallback(
    (question) => {
      const id = ensureConversation();
      updateConversation(id, (c) => ({
        ...c,
        title: c.titled ? c.title : titleFrom(question),
        titled: true,
        messages: [...c.messages, { id: newId(), role: "user", content: question }],
      }));
      ask(id, question, active?.sessionId ?? null);
    },
    [ensureConversation, updateConversation, ask, active]
  );

  const retry = useCallback(
    (errorMessage) => {
      if (!active) return;
      updateConversation(active.id, (c) => ({
        ...c,
        messages: c.messages.filter((m) => m.id !== errorMessage.id),
      }));
      ask(active.id, errorMessage.retryQuestion, active.sessionId);
    },
    [active, updateConversation, ask]
  );

  // Uploads index into the shared corpus, not the chat; the notice just makes
  // it visible in the conversation where the user will ask about it.
  const onUploaded = useCallback(
    (result) => {
      setDocuments((docs) => [result, ...docs.filter((d) => d.filename !== result.filename)]);
      const id = ensureConversation();
      appendMessage(id, {
        role: "notice",
        content: `Indexed ${result.filename} — ${result.chunks_indexed} chunks (${result.doc_type})`,
      });
    },
    [ensureConversation, appendMessage]
  );
  const uploads = useUploads(onUploaded);

  const closeIfNarrow = () => {
    if (isNarrow()) setSidebarOpen(false);
  };

  return (
    <div className={`app ${sidebarOpen ? "sidebar-open" : ""}`}>
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeId}
        onSelect={(id) => {
          setActiveId(id);
          closeIfNarrow();
        }}
        onNew={() => {
          setActiveId(null);
          closeIfNarrow();
        }}
        onDelete={(id) => {
          setConversations((list) => list.filter((c) => c.id !== id));
          if (id === activeId) setActiveId(null);
        }}
        documents={documents}
        status={status}
        onClose={() => setSidebarOpen(false)}
      />
      <ChatView
        conversation={active}
        pending={Boolean(active && pending[active.id])}
        agents={agents}
        backendError={backendError}
        onSend={send}
        onRetry={retry}
        uploads={uploads}
        onToggleSidebar={() => setSidebarOpen((o) => !o)}
      />
    </div>
  );
}
