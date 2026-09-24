// Presentation helpers for agent names. Nothing here lists agents: the set comes
// from GET /v1/agents, so a newly registered backend agent needs no frontend edit.

const WORD_OVERRIDES = { qa: "Q&A", sql: "SQL", api: "API" };

export function agentLabel(name) {
  if (!name) return "Assistant";
  return name
    .split(/[_-]+/)
    .map((w) => WORD_OVERRIDES[w] ?? w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// Stable hue per agent name, so colours survive reloads and new agents get one.
export function agentHue(name) {
  let hash = 0;
  for (const ch of name ?? "") hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return hash % 360;
}
