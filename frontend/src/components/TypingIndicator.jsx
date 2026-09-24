import { Sparkles } from "lucide-react";

export default function TypingIndicator() {
  return (
    <div className="msg assistant" aria-live="polite">
      <div className="avatar" aria-hidden="true">
        <Sparkles size={16} />
      </div>
      <div className="msg-body">
        <div className="typing">
          <span className="dots" aria-hidden="true">
            <i /><i /><i />
          </span>
          Routing to an agent…
        </div>
      </div>
    </div>
  );
}
