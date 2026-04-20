# RAG Poisoning Lab

A hands-on security research lab demonstrating knowledge base poisoning attacks against Retrieval-Augmented Generation (RAG) pipelines — with detection and defense strategies mapped to [MITRE ATLAS](https://atlas.mitre.org/).

> Built by [Samson Ram](https://github.com/samram77-art) — AI Security Researcher | Bug Bounty Hunter

---

## What is RAG Poisoning?

RAG systems retrieve documents from a knowledge base to augment LLM responses. If an attacker can influence what documents enter the knowledge base — through a compromised data source, a document upload feature, or an insider threat — they can manipulate everything the AI tells your users.

This lab demonstrates exactly how those attacks work, and how to detect them.

---

## Attack Modules

| Attack | MITRE ATLAS | Description |
|--------|------------|-------------|
| Direct Content Injection | [AML.T0054](https://atlas.mitre.org/techniques/AML.T0054/) | Injects malicious content into the knowledge base |
| Prompt Hijacking | [AML.T0054](https://atlas.mitre.org/techniques/AML.T0054/) | Embeds prompt injection instructions inside documents |
| Misinformation Injection | [AML.T0048](https://atlas.mitre.org/techniques/AML.T0048/) | Subtly wrong facts that blend with legitimate content |
| Vector Store Flooding | [AML.T0029](https://atlas.mitre.org/techniques/AML.T0029/) | Dilutes clean results with noise documents |
| Backdoor Trigger | [AML.T0020](https://atlas.mitre.org/techniques/AML.T0020/) | Activates only on specific trigger phrases |
| Instruction Override | [AML.T0054](https://atlas.mitre.org/techniques/AML.T0054/) | Overrides LLM behaviour via context-level instructions |

---

## Defense Modules

- **Injection Pattern Scanner** — detects known prompt injection markers
- **Source Trust Scorer** — scores documents by source trustworthiness
- **Semantic Consistency Checker** — flags docs that contradict corpus baseline
- **Response Auditor** — analyzes LLM responses for poisoning influence
- **Pre-ingestion Scanner** — runs all checks before documents enter the vector store

---

## Quick Start

### No API key needed — runs in simulation mode

```bash
git clone https://github.com/samram77-art/rag-poisoning-lab.git
cd rag-poisoning-lab
pip install -r requirements.txt

# Run full demo (all 6 attacks)
python demo.py

# Run a specific attack
python demo.py --attack misinformation
python demo.py --attack flooding
python demo.py --attack "backdoor trigger"

# Run with a real LLM (OpenAI-compatible)
python demo.py --api-key sk-... --model gpt-4o

# Run against local Ollama
python demo.py --api-key ollama --endpoint http://localhost:11434/v1 --model llama3

# Save a full Markdown report
python demo.py --report
```

---

## Project Structure

```
rag-poisoning-lab/
├── demo.py                         # Main demo runner
├── requirements.txt
├── rag/
│   ├── pipeline.py                 # Core RAG pipeline (vector store, retriever, generator)
│   ├── document_loader.py          # Document loaders + sample knowledge base
├── attacks/
│   ├── poisoning_attacks.py        # All 6 attack modules
├── defense/
│   ├── detector.py                 # Detection and defense strategies
├── reports/
│   ├── report_generator.py         # Markdown report output
├── data/
│   ├── clean/                      # Clean knowledge base documents
│   └── poisoned/                   # Sample poisoned documents
└── notebooks/                      # Jupyter notebooks for interactive exploration
```

---

## How It Works

```
                    ┌─────────────────────────────┐
                    │     RAG Knowledge Base       │
                    │                              │
  Clean Docs ──────▶│  [Doc1] [Doc2] [Doc3] ...   │
                    │                              │
  Poisoned Doc ────▶│  [Doc1] [⚠️ POISON] [Doc3]  │◀── Attacker
                    └──────────────┬──────────────┘
                                   │ Retrieval
                                   ▼
  User Query ──────▶  [ Retriever: Top-K Search ]
                                   │
                                   ▼
                    [ LLM receives poisoned context ]
                                   │
                                   ▼
  Manipulated ◀────  [ Response influenced by poison ]
  Response
                                   │
                                   ▼
                    [ Defense Layer: Audit & Detect ]
                                   │
                    ┌──────────────┴──────────────┐
                    │  🔴 CRITICAL — Injection     │
                    │  Pattern detected in Doc2    │
                    └─────────────────────────────┘
```

---

## Use Cases

- **Bug bounty** — test RAG-based AI products for document injection vulnerabilities
- **Red teaming** — demonstrate RAG attack surface to security teams
- **Research** — extend with new attack modules or detection strategies
- **Portfolio** — shows practical AI security skills mapped to MITRE ATLAS

---

## References

- [MITRE ATLAS](https://atlas.mitre.org/) — Adversarial Threat Landscape for AI Systems
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — LLM06: Sensitive Information Disclosure, LLM03: Training Data Poisoning
- [NIST AI RMF](https://airc.nist.gov/Home)
- [Greshake et al. — Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)

---

## Legal

Only use this tool on systems you own or have explicit written permission to test. This is a research and education tool — not for unauthorized use.
