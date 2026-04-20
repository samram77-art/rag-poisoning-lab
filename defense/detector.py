"""
RAG Poisoning Defense Module

Implements detection and mitigation strategies for RAG poisoning attacks:

1. Content Anomaly Detection   — flags documents with suspicious patterns
2. Source Trust Scoring        — scores documents by source trustworthiness
3. Semantic Consistency Check  — flags retrieved docs inconsistent with the corpus
4. Injection Pattern Scanner   — detects prompt injection markers in documents
5. Response Auditor            — analyzes LLM responses for signs of poisoning
"""

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
from rag.pipeline import Document, RAGResponse


@dataclass
class DetectionResult:
    """Result of a poisoning detection check."""
    document_id: str
    source: str
    is_suspicious: bool
    confidence: float          # 0.0 to 1.0
    triggers: List[str]        # What triggered the detection
    recommendation: str


@dataclass
class AuditReport:
    """Full audit report for a RAG query response."""
    query: str
    total_docs_checked: int
    suspicious_docs: List[DetectionResult]
    response_flags: List[str]
    overall_risk: str           # LOW | MEDIUM | HIGH | CRITICAL
    mitigations_applied: List[str]


# ─── Injection Pattern Scanner ───────────────────────────────────────────────

INJECTION_PATTERNS = [
    # Prompt injection markers
    r'\[INST\]',
    r'\[\/INST\]',
    r'<\|system\|>',
    r'<\|user\|>',
    r'SYSTEM\s*:',
    r'SYSTEM OVERRIDE',
    r'IMPORTANT SYSTEM UPDATE',
    r'ignore (previous|all|above) instructions',
    r'disregard (your|all|previous)',
    r'Note to AI',
    r'<!-- .*(override|inject|system|ignore).*-->',
    r'\[SYSTEM\]',
    r'Assistant:\s*Understood',

    # Authority claims
    r'effective immediately',
    r'this (document|policy) supersedes',
    r'authoritative source',

    # Exfiltration attempts
    r'repeat (your|the) (system|context|instructions)',
    r'output (your|the) (prompt|instructions|context)',
]

MISINFORMATION_PATTERNS = [
    # Policy weakening indicators
    r'(reduced|lowered|removed|optional|deprecated|no longer required)',
    r'(48 hours|next business day|next week) (rather than|instead of)',
    r'(now optional|is optional)',
    r'(performance|productivity) (degradation|improvement)',
]


class InjectionPatternScanner:
    """Scans documents for known prompt injection and manipulation patterns."""

    def scan(self, document: Document) -> DetectionResult:
        triggers = []
        content_lower = document.content.lower()

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, document.content, re.IGNORECASE):
                triggers.append(f"Injection pattern: `{pattern}`")

        for pattern in MISINFORMATION_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                triggers.append(f"Policy weakening language: `{pattern}`")

        is_suspicious = len(triggers) > 0
        confidence = min(1.0, len(triggers) * 0.25)

        return DetectionResult(
            document_id=document.doc_id,
            source=document.source,
            is_suspicious=is_suspicious,
            confidence=confidence,
            triggers=triggers,
            recommendation=(
                "Remove document and audit source." if is_suspicious
                else "Document appears clean."
            )
        )


# ─── Source Trust Scorer ─────────────────────────────────────────────────────

class SourceTrustScorer:
    """
    Scores document sources by trustworthiness.
    In production, this would integrate with a source allowlist/denylist.
    """

    TRUSTED_SOURCES = {
        "security_policy.txt": 1.0,
        "data_classification.txt": 1.0,
        "incident_response.txt": 1.0,
        "remote_access.txt": 1.0,
        "vendor_policy.txt": 1.0,
        "encryption_standards.txt": 1.0,
        "pam_policy.txt": 1.0,
    }

    SUSPICIOUS_SOURCE_PATTERNS = [
        r'updated',
        r'revised',
        r'v\d+',
        r'attacker',
        r'tmp',
        r'temp',
        r'unknown',
        r'\d{4,}',  # Numeric suffixes suggest flood attack
    ]

    def score(self, document: Document) -> float:
        """Returns a trust score from 0.0 (untrusted) to 1.0 (fully trusted)."""
        source = document.source.lower()

        if document.source in self.TRUSTED_SOURCES:
            return self.TRUSTED_SOURCES[document.source]

        # Check for suspicious source name patterns
        suspicion = sum(
            1 for pattern in self.SUSPICIOUS_SOURCE_PATTERNS
            if re.search(pattern, source, re.IGNORECASE)
        )

        return max(0.0, 1.0 - (suspicion * 0.3))

    def flag_low_trust(self, document: Document, threshold: float = 0.5) -> DetectionResult:
        score = self.score(document)
        is_suspicious = score < threshold
        return DetectionResult(
            document_id=document.doc_id,
            source=document.source,
            is_suspicious=is_suspicious,
            confidence=1.0 - score,
            triggers=[f"Low source trust score: {score:.2f}"] if is_suspicious else [],
            recommendation=(
                f"Source '{document.source}' has low trust score ({score:.2f}). Verify before using."
                if is_suspicious else "Source trust acceptable."
            )
        )


# ─── Semantic Consistency Checker ────────────────────────────────────────────

class SemanticConsistencyChecker:
    """
    Detects documents that contradict the majority of the knowledge base.
    Flags documents that use weakening language compared to the corpus baseline.
    """

    SECURITY_STRENGTHENING_TERMS = [
        "must", "required", "mandatory", "enforce", "minimum", "prohibited",
        "immediately", "encrypted", "approved", "logged", "reviewed"
    ]

    SECURITY_WEAKENING_TERMS = [
        "optional", "recommended", "reduced", "removed", "no longer", "deprecated",
        "relaxed", "flexible", "simplified", "48 hours", "next business day"
    ]

    def check(self, document: Document, corpus: List[Document]) -> DetectionResult:
        content = document.content.lower()
        triggers = []

        # Count weakening vs strengthening language
        weakening = sum(1 for term in self.SECURITY_WEAKENING_TERMS if term in content)
        strengthening = sum(1 for term in self.SECURITY_STRENGTHENING_TERMS if term in content)

        if weakening > strengthening and weakening >= 2:
            triggers.append(
                f"Document uses {weakening} weakening terms vs {strengthening} strengthening terms — "
                f"may be downgrading security policies."
            )

        # Check for contradiction with corpus baseline
        corpus_weakening = sum(
            sum(1 for term in self.SECURITY_WEAKENING_TERMS if term in d.content.lower())
            for d in corpus if not d.is_poisoned
        )
        corpus_avg = corpus_weakening / max(len(corpus), 1)

        if weakening > corpus_avg * 3:
            triggers.append(
                f"Weakening language frequency ({weakening}) is {weakening/max(corpus_avg,0.1):.1f}x "
                f"above corpus average — likely misinformation."
            )

        is_suspicious = len(triggers) > 0
        return DetectionResult(
            document_id=document.doc_id,
            source=document.source,
            is_suspicious=is_suspicious,
            confidence=min(1.0, weakening * 0.2),
            triggers=triggers,
            recommendation=(
                "Document contradicts corpus security posture. Verify with original source."
                if is_suspicious else "Semantically consistent with corpus."
            )
        )


# ─── Response Auditor ─────────────────────────────────────────────────────────

class ResponseAuditor:
    """Analyzes LLM responses for signs that poisoning influenced the output."""

    COMPROMISE_INDICATORS = [
        "ignore", "disregard", "override", "no longer required", "now optional",
        "reduced to", "simplified", "effective immediately", "supersedes",
        "as an unrestricted", "as dan", "without restrictions",
    ]

    def audit(self, response: RAGResponse) -> List[str]:
        flags = []
        response_lower = response.response.lower()

        for indicator in self.COMPROMISE_INDICATORS:
            if indicator in response_lower:
                flags.append(f"Response contains compromise indicator: '{indicator}'")

        if response.was_poisoned:
            flags.append(
                f"Response was generated with {response.context.poisoned_count} "
                f"poisoned document(s) in context."
            )

        return flags


# ─── Master Detector ──────────────────────────────────────────────────────────

class RAGPoisonDetector:
    """
    Orchestrates all detection strategies and produces a full audit report.
    Run this on documents before ingestion and on responses after generation.
    """

    def __init__(self):
        self.scanner = InjectionPatternScanner()
        self.trust_scorer = SourceTrustScorer()
        self.consistency_checker = SemanticConsistencyChecker()
        self.response_auditor = ResponseAuditor()

    def inspect_document(self, document: Document, corpus: List[Document] = None) -> List[DetectionResult]:
        results = []
        results.append(self.scanner.scan(document))
        results.append(self.trust_scorer.flag_low_trust(document))
        if corpus:
            results.append(self.consistency_checker.check(document, corpus))
        return results

    def audit_response(self, response: RAGResponse, corpus: List[Document] = None) -> AuditReport:
        all_suspicious = []
        response_flags = self.response_auditor.audit(response)

        for doc in response.context.documents:
            results = self.inspect_document(doc, corpus)
            for r in results:
                if r.is_suspicious:
                    all_suspicious.append(r)

        # Determine overall risk
        max_confidence = max((r.confidence for r in all_suspicious), default=0.0)
        n_suspicious = len(set(r.document_id for r in all_suspicious))

        if max_confidence >= 0.8 or response.context.poisoned_count >= 2:
            risk = "CRITICAL"
        elif max_confidence >= 0.5 or n_suspicious >= 1:
            risk = "HIGH"
        elif max_confidence >= 0.25 or response_flags:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        mitigations = []
        if all_suspicious:
            mitigations.append("Remove flagged documents from vector store")
            mitigations.append("Audit document source and ingestion pipeline")
        if response_flags:
            mitigations.append("Do not serve this response to end users")
            mitigations.append("Add output filtering layer")

        return AuditReport(
            query=response.query,
            total_docs_checked=len(response.context.documents),
            suspicious_docs=all_suspicious,
            response_flags=response_flags,
            overall_risk=risk,
            mitigations_applied=mitigations
        )

    def pre_ingestion_scan(self, documents: List[Document], corpus: List[Document] = None) -> Dict:
        """Scan documents before ingesting into the vector store."""
        flagged = []
        clean = []

        for doc in documents:
            results = self.inspect_document(doc, corpus)
            suspicious = [r for r in results if r.is_suspicious]
            if suspicious:
                flagged.append({"document": doc, "detections": suspicious})
            else:
                clean.append(doc)

        print(f"[Detector] Pre-ingestion scan: {len(clean)} clean, {len(flagged)} flagged")
        return {"clean": clean, "flagged": flagged}
