"""Prompts. SYSTEM_CONTEXT is the original ClusterMind system text, with the
node-name examples kept; the risk-score block of the original diagnosis
prompt is removed because the original risk engine consumed synthetic
temperature/memory signals."""

SYSTEM_CONTEXT = (
    "You are ClusterMind, an AI assistant for a shared GPU compute cluster. "
    "Nodes are identified by human-friendly names such as Node-1, Node-2, Node-3, "
    "exactly as they appear in the telemetry. "
    "Always use the exact node name from the provided context. "
    "Never invent node names, job names, metrics, or events. "
    "Respond in plain text only — no markdown, no asterisks, no bold, no bullet points."
)


def diagnosis_system(evidence_text):
    # evidence first => identical prefix across questions of a snapshot (KV-cache reuse)
    return f"{SYSTEM_CONTEXT}\n\nTelemetry:\n{evidence_text}"


def diagnosis_user(question, style="contract"):
    if style == "contract":
        return (f"Question: {question}\n\n"
                "Answer in 1-3 short plain-text sentences. Name the exact node(s) and quote the "
                "metric values exactly as they appear in the telemetry. If the question asks about "
                "a node that is not in the telemetry, say that the node is not present in the telemetry.")
    if style == "original":  # the original ClusterMind Issue/Recommendation template
        return (f"Question: {question}\n\n"
                "Issue: <2 concise plain-text sentences naming the exact node, metric value, and problem>\n"
                "Recommendation: <1 concise plain-text sentence with a single clear action>")
    raise KeyError(style)


def action_user(instruction, allowed_actions):
    acts = ", ".join(allowed_actions)
    return (f"Respond ONLY with valid JSON. No explanation, no markdown, no code fences.\n\n"
            f'{{"action": "<one of: {acts}>", "node": "<exact node name from context e.g. Node-1>", "reason": "<short reason>"}}\n\n'
            f'If no clear action applies: {{"action": "none", "node": null, "reason": "<why>"}}\n\n'
            f"Instruction: {instruction}")


JUDGE_SYSTEM_PREFIX = (
    "You are a strict verifier for a GPU-cluster assistant. You are given the telemetry "
    "records that were shown to the assistant. Decide whether EVERY factual claim in the "
    "assistant's answer (node names, numbers, and states such as idle/busy) is supported by "
    "the telemetry. An answer that correctly says a node is not present in the telemetry is SUPPORTED."
)


def judge_system(evidence_text):
    return f"{JUDGE_SYSTEM_PREFIX}\n\nTelemetry:\n{evidence_text}"


def judge_user(question, answer):
    return (f"Question: {question}\nAssistant answer: {answer}\n\n"
            "Is every claim in the answer supported by the telemetry? Reply with exactly one word: SUPPORTED or UNSUPPORTED.")
