"""
Ontology of the knowledge management system (Phase II, expanded version).

Classes
    Category  - the two document types of the repository
    Document  - a policy, procedure, guide or support ticket
    Concept   - an IT topic mentioned by documents (VPN, Password, Malware...)
    Domain    - a group of related concepts (Security, Network, Hardware...)

Relations
    Document  --belongs_to-->      Category
    Document  --mentions-->        Concept
    Concept   --part_of-->         Domain
    Document  --similar_to-->      Document   (semantic similarity above a threshold)
    Incident  --governed_by-->     Policy     (closest policy for a support ticket)
    Concept   --co_occurs_with-->  Concept    (mentioned together in 2+ documents)

Concepts are detected with a controlled vocabulary: each concept has a list of
terms (already in clean form: lowercase, no punctuation). The same vocabulary
is used to expand search queries, so the ontology also improves retrieval.
"""
import re

CLASSES = {
    "Category": "Type of document in the repository (Policy/Process or Support/Incidents).",
    "Document": "A single knowledge item: policy, procedure, guide or resolved support ticket.",
    "Concept": "An IT topic that documents talk about (e.g. VPN, Password, Malware).",
    "Domain": "A group of related concepts (e.g. Security, Network, Hardware).",
}

RELATIONS = {
    "belongs_to": ("Document", "Category", "The document is classified in this category."),
    "mentions": ("Document", "Concept", "The document talks about this concept."),
    "part_of": ("Concept", "Domain", "The concept belongs to this knowledge domain."),
    "similar_to": ("Document", "Document", "Both documents are semantically similar."),
    "governed_by": ("Document", "Document", "A support ticket is covered by this policy or procedure."),
    "co_occurs_with": ("Concept", "Concept", "Both concepts appear together in two or more documents."),
}

DOMAINS = [
    "Security", "Identity & Access", "Network", "Hardware",
    "Software", "Communication", "Data & Storage", "IT Operations", "People & HR",
]

# concept -> (domain, terms in clean form)
CONCEPTS = {
    # Security
    "Malware": ("Security", ["malware", "virus", "viruses", "ransomware", "antivirus", "malicious", "infected"]),
    "Phishing": ("Security", ["phishing", "fraudulent", "unknown sender", "attachment"]),
    "Firewall": ("Security", ["firewall"]),
    "Security Patching": ("Security", ["patch", "patches", "patching", "security update", "vulnerabilities"]),
    "Security Incident": ("Security", ["security breach", "security incident", "cybersecurity", "incident response", "isolate", "device was isolated"]),
    "Security Awareness": ("Security", ["awareness", "phishing simulations", "phishing test", "cybersecurity training"]),
    # Identity & Access
    "Password": ("Identity & Access", ["password", "passwords", "credentials", "master password"]),
    "Multi-Factor Authentication": ("Identity & Access", ["multi factor authentication", "mfa", "authenticator", "authentication code", "sms codes"]),
    "User Account": ("Identity & Access", ["active directory", "user account", "account", "accounts", "log in", "login", "sign in", "locked"]),
    "Access Permissions": ("Identity & Access", ["permissions", "permission", "roles", "security group", "revocation", "access was granted", "unauthorized access"]),
    # Network
    "VPN": ("Network", ["vpn"]),
    "Wi-Fi": ("Network", ["wi fi", "wireless network", "wireless router", "access point", "router", "guest network"]),
    "Network Connectivity": ("Network", ["network cable", "connection refused", "connectivity", "disconnected", "network traffic", "private network", "corporate network"]),
    # Hardware
    "Computer Equipment": ("Hardware", ["laptop", "laptops", "computer", "computers", "computer equipment", "workstation", "equipment", "hardware"]),
    "Printer": ("Hardware", ["printer", "printers", "printing", "printed", "paper jam", "scanned", "scanning"]),
    "Peripherals": ("Hardware", ["keyboard", "mouse", "monitor", "monitors", "peripheral", "peripherals", "hdmi", "bluetooth", "webcam", "camera", "charger", "battery", "batteries", "cable"]),
    "Mobile Device": ("Hardware", ["phone", "phones", "mobile device", "mobile device management", "mdm"]),
    "Disk Storage": ("Hardware", ["disk", "hard drive", "free space", "disk space", "disk cleanup"]),
    # Software
    "Software Installation": ("Software", ["software", "install", "installation", "installed", "uninstalled", "program"]),
    "Software License": ("Software", ["license", "licenses"]),
    "Operating System": ("Software", ["windows", "blue screen", "driver", "restarting", "startup", "boot", "booting"]),
    "Business Application": ("Software", ["application", "payroll", "sales application", "administrative system", "portal"]),
    # Communication
    "Email": ("Communication", ["email", "emails", "mailbox", "inbox", "outlook", "email client"]),
    "Video Conference": ("Communication", ["video conference", "video call", "audio", "speakers", "boardroom", "presentation"]),
    # Data & Storage
    "Backup & Recovery": ("Data & Storage", ["backup", "backups", "restore", "restored", "recovered", "recovery", "disaster recovery", "checkpoint"]),
    "Cloud Storage": ("Data & Storage", ["cloud", "shared folder", "shared drive", "dropbox", "google drive", "archive", "archived"]),
    "Database": ("Data & Storage", ["database", "databases", "query", "index", "records"]),
    "Data Classification": ("Data & Storage", ["confidential", "restricted", "encrypted", "encryption", "classified"]),
    # IT Operations
    "Help Desk": ("IT Operations", ["help desk", "ticket", "tickets", "technical support", "support levels", "escalated", "escalation"]),
    "Change Management": ("IT Operations", ["change request", "change advisory board", "maintenance window", "rollback"]),
    "Asset Inventory": ("IT Operations", ["inventory", "asset tag", "asset", "audit", "warehouse"]),
    "Monitoring": ("IT Operations", ["monitoring", "alert", "alerted", "log files"]),
    # People & HR
    "Onboarding": ("People & HR", ["onboarding", "new employee", "new employees", "new hire", "first day"]),
    "Offboarding": ("People & HR", ["offboarding", "leaves the organization"]),
    "Remote Work": ("People & HR", ["remote", "working from home", "work from home"]),
    "Vacation & Absences": ("People & HR", ["vacation", "absences", "absence"]),
    "Expense Reimbursement": ("People & HR", ["reimbursement", "tax invoice", "budget", "purchases"]),
}

_PATTERNS = {
    name: re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b")
    for name, (_, terms) in CONCEPTS.items()
}


def detect_concepts(clean):
    """Concepts mentioned in an already-cleaned text."""
    return [name for name, pattern in _PATTERNS.items() if pattern.search(clean)]


def expand_query(clean):
    """Add the vocabulary of every detected concept to the query (ontology-based query expansion)."""
    concepts = detect_concepts(clean)
    extra = []
    for name in concepts:
        extra.extend(CONCEPTS[name][1])
    return (clean + " " + " ".join(extra)).strip(), concepts


def concept_domain(name):
    return CONCEPTS[name][0]
