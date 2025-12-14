#!/usr/bin/env python3
"""
### MOD: Completely expanded tagger
- Richer 5GC/Open5GS/K8s-aware tags (timestamps, PFCP/NAS/NGAP, HTTP, IDs, K8s metadata, etc.)
- ANSI escape stripping before tokenization
- First-match-wins regex bank (ordered)
- Same public API: FieldTag, TaggedToken, LogTagger, build_tokenizer()

NOTE: If you already log/tag extra fields, just add regexes to PATTERN_LIST.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizer

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# TAG ENUM
# ---------------------------------------------------------------------

class FieldTag(IntEnum):
    # Time
    TIMESTAMP_ISO = auto()
    TIMESTAMP_SHORT = auto()

    # Severity / module / infra
    SEVERITY = auto()
    NF = auto()
    MODULE = auto()
    K8S_POD = auto()
    POD_NAMESPACE = auto()
    CONTAINER = auto()
    PID = auto()
    TID = auto()

    # Code / src
    FILE_LOC = auto()
    FUNC = auto()
    FUNC_CALL = auto()

    # HTTP / SBI
    URI = auto()
    HTTP_METHOD = auto()
    HTTP_CODE = auto()
    SBI_SERVICE = auto()

    # Network / IDs
    IPV4 = auto()
    IPV6 = auto()
    IMSI = auto()
    SUPI = auto()
    SUCI = auto()
    GUTI = auto()
    S_NSSAI = auto()
    DNN = auto()
    APN = auto()
    PDU_SESS_ID = auto()
    TEID = auto()
    FSEID = auto()
    PDRID = auto()
    FARID = auto()
    QERID = auto()

    # Causes / states
    CAUSE_CODE = auto()
    RESULT_CODE = auto()
    PFCP_CAUSE = auto()
    STATE = auto()

    # Protocol messages / steps
    NAS_MSG = auto()
    NGAP_MSG = auto()
    PFCP_MSG = auto()
    CALLFLOW_STEP = auto()
    PROCEDURE_NAME = auto()

    # Misc
    REQ_ID = auto()
    UUID = auto()
    LAT_MS = auto()
    HEX = auto()
    NUMBER = auto()
    JSON_LINE = auto()
    ANSI = auto()

    UNK = auto()


# ---------------------------------------------------------------------
# REGEX BANK  (ordered; first match wins)
# ---------------------------------------------------------------------

ANSI_RE = re.compile(r"\x1B\[[0-9;]*m")

PATTERN_LIST = [
    (FieldTag.TIMESTAMP_ISO,  re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+(?:Z|[+-]\d{2}:\d{2})\b")),
    (FieldTag.TIMESTAMP_SHORT,re.compile(r"\b\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\b")),

    (FieldTag.SEVERITY,       re.compile(r"\b(INFO|ERROR|WARNING|WARN|DEBUG|CRITICAL|TRACE|I|E|W|D|C)\b", re.IGNORECASE)),
    (FieldTag.NF,             re.compile(r"\b(amf|smf|upf|ausf|pcf|udm|udr|nrf|nssf|bsf|scp|mongodb|metrics|sbi|ngap|pfcp)\b", re.IGNORECASE)),
    (FieldTag.MODULE,         re.compile(r"\[[a-z0-9_+-]+\]", re.IGNORECASE)),
    (FieldTag.K8S_POD,        re.compile(r"\b[a-z0-9-]+-(deployment|statefulset)-[a-z0-9-]+\b")),
    (FieldTag.POD_NAMESPACE,  re.compile(r"\bnamespace[:=]\s*[\w-]+\b", re.IGNORECASE)),
    (FieldTag.CONTAINER,      re.compile(r"\bcontainer[:=]\s*[\w-]+\b", re.IGNORECASE)),
    (FieldTag.PID,            re.compile(r"\bpid[:=]?\s*\d+\b", re.IGNORECASE)),
    (FieldTag.TID,            re.compile(r"\btid[:=]?\s*\d+\b", re.IGNORECASE)),

    (FieldTag.FILE_LOC,       re.compile(r"\([^()]+:\d+\)")),
    (FieldTag.FUNC,           re.compile(r"\b[a-zA-Z_][\w]*\s*\([^)]*\)\s*(?=\{|\:)", re.MULTILINE)),
    (FieldTag.FUNC_CALL,      re.compile(r"\b[a-zA-Z_][\w]*\([^()]*\)")),

    (FieldTag.URI,            re.compile(r"/(?:[A-Za-z0-9._{}-]+)(?:/[A-Za-z0-9._{}-]+)+")),
    (FieldTag.HTTP_METHOD,    re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\b")),
    (FieldTag.HTTP_CODE,      re.compile(r"\b[1-5]\d{2}\b")),
    (FieldTag.SBI_SERVICE,    re.compile(r"\b(namf-[\w-]+|nsmf-[\w-]+|nscp-[\w-]+|nbsf-[\w-]+|nnssf-[\w-]+|nausf-[\w-]+|nudr-[\w-]+|nudm-[\w-]+|nnrf-[\w-]+|npcf-[\w-]+)\b", re.IGNORECASE)),

    (FieldTag.IPV4,           re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    (FieldTag.IPV6,           re.compile(r"\b(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}\b")),
    (FieldTag.IMSI,           re.compile(r"\bIMSI[-:]?\d{5,}\b|\b\d{15}\b")),
    (FieldTag.SUPI,           re.compile(r"\bSUPI[-:]?[A-Za-z0-9:_-]+\b", re.IGNORECASE)),
    (FieldTag.SUCI,           re.compile(r"\bSUCI[-:]?[A-Za-z0-9:_-]+\b", re.IGNORECASE)),
    (FieldTag.GUTI,           re.compile(r"\bGUTI[-:]?[A-Za-z0-9:_-]+\b", re.IGNORECASE)),
    (FieldTag.S_NSSAI,        re.compile(r"\bS-?NSSAI[-:=]?\s*[0-9a-fA-F/]+\b", re.IGNORECASE)),
    (FieldTag.DNN,            re.compile(r"\bDNN[-:=]?\s*[\w.-]+\b", re.IGNORECASE)),
    (FieldTag.APN,            re.compile(r"\bAPN[-:=]?\s*[\w.-]+\b", re.IGNORECASE)),
    (FieldTag.PDU_SESS_ID,    re.compile(r"pdu\s*session\s*id[:=\[]?\s*\d+", re.IGNORECASE)),
    (FieldTag.TEID,           re.compile(r"\bTEID[- :]?(?:0x)?[0-9A-Fa-f]+\b")),
    (FieldTag.FSEID,          re.compile(r"\bF-?SEID[- :]?(?:0x)?[0-9A-Fa-f]+\b", re.IGNORECASE)),
    (FieldTag.PDRID,          re.compile(r"\bPDR[- ]?ID[- :]?(?:0x)?[0-9A-Fa-f]+\b", re.IGNORECASE)),
    (FieldTag.FARID,          re.compile(r"\bFAR[- ]?ID[- :]?(?:0x)?[0-9A-Fa-f]+\b", re.IGNORECASE)),
    (FieldTag.QERID,          re.compile(r"\bQER[- ]?ID[- :]?(?:0x)?[0-9A-Fa-f]+\b", re.IGNORECASE)),

    (FieldTag.CAUSE_CODE,     re.compile(r"\b[Cc]ause(?:\s*[:=\[])?\s*[\w-]+\b")),
    (FieldTag.RESULT_CODE,    re.compile(r"\bresult(?:_code)?[:=]\s*[\w-]+\b", re.IGNORECASE)),
    (FieldTag.PFCP_CAUSE,     re.compile(r"\bPFCP[- ]?Cause[:=]?\s*\w+\b", re.IGNORECASE)),
    (FieldTag.STATE,          re.compile(r"\bstate[:=]\s*\w+\b", re.IGNORECASE)),

    (FieldTag.NAS_MSG,        re.compile(r"\b(NAS|DLNASTransport|RegistrationRequest|ServiceRequest)\b", re.IGNORECASE)),
    (FieldTag.NGAP_MSG,       re.compile(r"\b(NGAP|InitialContextSetupRequest|UEContextRelease)\b", re.IGNORECASE)),
    (FieldTag.PFCP_MSG,       re.compile(r"\b(PFCP|SessionEstablishmentRequest|SessionReport)\b", re.IGNORECASE)),
    (FieldTag.CALLFLOW_STEP,  re.compile(r"\b(registration|deregistration|authentication|service\s+request|pdu\s+session\s+(establishment|release|modify)|configuration\s+update|handover|context\s+release)\b", re.IGNORECASE)),
    (FieldTag.PROCEDURE_NAME, re.compile(r"\bUE-[A-Z-]+-PROCEDURE\b")),

    (FieldTag.REQ_ID,         re.compile(r"\brequest[_-]?id[:=]?\s*[0-9a-f-]{6,}\b", re.IGNORECASE)),
    (FieldTag.UUID,           re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[089ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)),
    (FieldTag.LAT_MS,         re.compile(r"\b\d+(?:\.\d+)?\s*ms\b")),

    (FieldTag.HEX,            re.compile(r"\b0x[0-9A-Fa-f]+\b")),
    (FieldTag.NUMBER,         re.compile(r"\b\d+\b")),
    (FieldTag.JSON_LINE,      re.compile(r"^\s*\{.*\}\s*$")),
    (FieldTag.ANSI,           ANSI_RE),
]

# ---------------------------------------------------------------------

@dataclass
class TaggedToken:
    token_id: int
    tag_id: int


class LogTagger:
    """Tokenises log lines and assigns field tags."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def tag_line(self, line: str) -> List[TaggedToken]:
        clean = ANSI_RE.sub("", line)
        tokens = self.tokenizer.tokenize(clean)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tags = [self._tag_for_token(tok) for tok in tokens]
        return [TaggedToken(tid, tag) for tid, tag in zip(token_ids, tags)]

    def tag_lines(self, lines: Iterable[str]) -> List[TaggedToken]:
        out: List[TaggedToken] = []
        for ln in lines:
            out.extend(self.tag_line(ln))
        return out

    def _tag_for_token(self, token: str) -> int:
        for tag, pattern in PATTERN_LIST:
            if pattern.search(token):
                return int(tag)
        return int(FieldTag.UNK)


def build_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load tokenizer and ensure pad token exists."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token
    return tok