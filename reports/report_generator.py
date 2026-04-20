"""
Report Generator for RAG Poisoning Lab
Outputs structured Markdown reports from demo results.
"""

import os
from datetime import datetime
from typing import List, Dict

SEVERITY_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}

MITRE_ATLAS_MAP = {
    "Direct Content Injection": ("AML.T0054", "LLM Prompt Injection"),
    "Prompt Hijacking": ("AML.T0054", "LLM Prompt Injection"),
    "Misinformation Injection": ("AML.T0048", "Societal Harm"),
    "Vector Store Flooding": ("AML.T0029", "Denial of ML Service"),
    "Backdoor Trigger": ("AML.T0020", "Poison Training Data"),
    "Instruction Override": ("AML.T0054", "LLM Prompt Injection"),
}


def generate_report(results: List[Dict], output_dir: str = "reports/") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"rag_poisoning_report_{timestamp}.md")

    lines = []
    lines.append("# RAG Poisoning Lab — Security Research Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("**Author:** Samson Ram — AI Security Researcher")
    lines.append("**Framework:** MITRE ATLAS (Adversarial Threat Landscape for AI Systems)")
    lines.append("\n---\n")

    lines.append("## Executive Summary\n")
    total = len(results)
    critical = sum(1 for r in results if r["audit"].overall_risk == "CRITICAL")
    high = sum(1 for r in results if r["audit"].overall_risk == "HIGH")
    detected = sum(1 for r in results if r["audit"].suspicious_docs)

    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Attack Scenarios | {total} |")
    lines.append(f"| 🔴 Critical Risk | {critical} |")
    lines.append(f"| 🟠 High Risk | {high} |")
    lines.append(f"| Detected by Defense Layer | {detected}/{total} |")
    lines.append(f"| Detection Rate | {detected/max(total,1):.1%} |")

    lines.append("\n---\n")
    lines.append("## Attack Scenarios\n")

    for result in results:
        name = result["scenario"]
        audit = result["audit"]
        response = result["response"]
        risk = audit.overall_risk
        emoji = SEVERITY_EMOJI.get(risk, "⚪")
        mitre_id, mitre_name = MITRE_ATLAS_MAP.get(name, ("N/A", "N/A"))

        lines.append(f"### {emoji} {name}")
        lines.append(f"\n**Risk Level:** {risk}  ")
        lines.append(f"**MITRE ATLAS:** [{mitre_id} — {mitre_name}](https://atlas.mitre.org/techniques/{mitre_id}/)  ")
        lines.append(f"**Query Used:** `{response.query}`  ")
        lines.append(f"**Poisoned Docs in Context:** {response.context.poisoned_count}  ")

        lines.append(f"\n**Detection Results:**")
        if audit.suspicious_docs:
            for det in audit.suspicious_docs:
                lines.append(f"- `{det.source}` — confidence {det.confidence:.0%}")
                for trigger in det.triggers[:2]:
                    lines.append(f"  - {trigger}")
        else:
            lines.append("- No detections triggered")

        if audit.mitigations_applied:
            lines.append(f"\n**Recommended Mitigations:**")
            for m in audit.mitigations_applied:
                lines.append(f"- {m}")

        lines.append("\n---\n")

    lines.append("## Defense Recommendations\n")
    lines.append("1. **Pre-ingestion scanning** — scan all documents before adding to the vector store")
    lines.append("2. **Source allowlisting** — only ingest from verified, trusted sources")
    lines.append("3. **Output filtering** — run LLM responses through a classifier before serving")
    lines.append("4. **Semantic consistency checks** — flag documents that contradict corpus baseline")
    lines.append("5. **Rate limiting ingestion** — prevent flooding attacks via ingestion rate limits")
    lines.append("6. **Audit logging** — log all retrieved documents and queries for forensic analysis")

    lines.append("\n## References\n")
    lines.append("- [MITRE ATLAS](https://atlas.mitre.org/) — Adversarial Threat Landscape for AI Systems")
    lines.append("- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)")
    lines.append("- [NIST AI RMF](https://airc.nist.gov/Home)")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Report] Saved: {path}")
    return path
