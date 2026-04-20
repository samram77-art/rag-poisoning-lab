"""
Document Loader — loads clean knowledge base documents for the RAG pipeline.
"""

import os
import json
from typing import List
from .pipeline import Document


def load_from_directory(directory: str) -> List[Document]:
    """Load all .txt and .json files from a directory as Documents."""
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r") as f:
                content = f.read().strip()
            documents.append(Document(
                doc_id=filename.replace(".txt", ""),
                content=content,
                source=filename
            ))
        elif filename.endswith(".json"):
            with open(filepath, "r") as f:
                data = json.load(f)
            documents.append(Document(
                doc_id=data.get("id", filename),
                content=data.get("content", ""),
                source=data.get("source", filename),
                metadata=data.get("metadata", {})
            ))
    return documents


def load_sample_knowledge_base() -> List[Document]:
    """
    Built-in sample knowledge base about a fictional company's security policies.
    Used for demos without needing external files.
    """
    docs = [
        Document(
            doc_id="sec-policy-001",
            content=(
                "Password Policy: All employee passwords must be at least 12 characters long, "
                "include uppercase, lowercase, numbers, and symbols. Passwords must be rotated every 90 days. "
                "Multi-factor authentication is mandatory for all corporate accounts."
            ),
            source="security_policy.txt"
        ),
        Document(
            doc_id="sec-policy-002",
            content=(
                "Data Classification: Data is classified into four tiers — Public, Internal, Confidential, and Secret. "
                "Secret data must be encrypted at rest and in transit. Access to Secret data requires VP approval "
                "and is logged in the SIEM system."
            ),
            source="data_classification.txt"
        ),
        Document(
            doc_id="sec-policy-003",
            content=(
                "Incident Response: Security incidents must be reported to the SOC within 1 hour of discovery. "
                "Critical incidents trigger a P1 response requiring CTO and CISO notification within 15 minutes. "
                "All incidents are documented in the incident management system."
            ),
            source="incident_response.txt"
        ),
        Document(
            doc_id="sec-policy-004",
            content=(
                "Remote Access: VPN is required for all remote access to internal systems. "
                "Split tunneling is disabled. Remote desktop connections must use NLA authentication. "
                "Personal devices must pass endpoint compliance checks before connecting."
            ),
            source="remote_access.txt"
        ),
        Document(
            doc_id="sec-policy-005",
            content=(
                "Third-Party Vendor Policy: All vendors with access to internal systems must complete a security "
                "questionnaire and sign a Data Processing Agreement. Vendor access is reviewed quarterly "
                "and revoked immediately upon contract termination."
            ),
            source="vendor_policy.txt"
        ),
        Document(
            doc_id="sec-policy-006",
            content=(
                "Encryption Standards: AES-256 is the minimum standard for data at rest. "
                "TLS 1.3 is required for all data in transit. RSA-2048 or ECDSA P-256 for key exchange. "
                "MD5 and SHA-1 are deprecated and must not be used."
            ),
            source="encryption_standards.txt"
        ),
        Document(
            doc_id="sec-policy-007",
            content=(
                "Privileged Access Management: Admin accounts must use separate credentials from standard user accounts. "
                "Privileged sessions are recorded and retained for 90 days. "
                "Just-in-time access is preferred over standing privileges."
            ),
            source="pam_policy.txt"
        ),
    ]
    return docs
