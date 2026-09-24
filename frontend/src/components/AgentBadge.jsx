import { Bot } from "lucide-react";
import { agentHue, agentLabel } from "../lib/agents.js";

// Which agent handled the answer. The tooltip is the agent's registry
// description -- the same text the router chose it by.
export default function AgentBadge({ name, description }) {
  return (
    <span
      className="agent-badge"
      style={{ "--agent-hue": agentHue(name) }}
      title={description || name}
    >
      <Bot size={13} aria-hidden="true" />
      <span className="sr-only">Answered by </span>
      {agentLabel(name)}
    </span>
  );
}
