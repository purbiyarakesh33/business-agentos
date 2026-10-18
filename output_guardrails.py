"""Local-only privacy checks for user-visible business reports and evidence."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class OutputGuardDecision:
    action: str
    model: str
    reasons: tuple[str, ...]


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-])\d{3}[\s.-]\d{4}(?!\w)")
_RECORD_ID_RE = re.compile(r"\b(?:CUST|CUSTOMER|USER|ORD|ORDER|RET|REF|PAY)[-_ ]?\d{3,}\b", re.IGNORECASE)
_SYSTEM_PROMPT_MARKERS = (
    "you are a friendly general assistant in a business application.",
    "never guess or offer generic possible causes for company-specific",
    "this app routes supported analysis requests to that workflow.",
)
_SECRET_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{15,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|password|access[_-]?token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}", re.IGNORECASE),
)
_SENSITIVE_FIELDS = {
    "customerid", "userid", "personid", "contactid", "customername", "fullname",
    "firstname", "lastname", "email", "emailaddress", "phone", "phonenumber",
    "mobile", "address", "streetaddress", "dateofbirth", "dob", "ssn",
    "socialsecuritynumber", "cardnumber", "creditcardnumber", "paymentcard",
    "bankaccount", "bankaccountnumber", "paymenttoken", "authtoken", "accesstoken",
    "password", "apikey", "secretkey", "orderid", "refundid", "paymentid",
}


def _normal_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _has_sensitive_fields(value) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if _normal_key(key) in _SENSITIVE_FIELDS and _has_value(child):
                return True
            if _has_sensitive_fields(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_sensitive_fields(child) for child in value)
    return False


def _luhn_valid(candidate: str) -> bool:
    digits = [int(char) for char in candidate if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _sensitive_table_columns(evidence: list[dict]) -> bool:
    """Catch private values in list-of-lists SQL results using their column names."""
    for item in evidence:
        if not isinstance(item, dict):
            continue
        columns = item.get("columns")
        rows = item.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            continue
        sensitive_indexes = [
            index for index, column in enumerate(columns)
            if _normal_key(column) in _SENSITIVE_FIELDS
        ]
        for row in rows:
            if isinstance(row, (list, tuple)) and any(
                index < len(row) and _has_value(row[index]) for index in sensitive_indexes
            ):
                return True
            if isinstance(row, dict) and _has_sensitive_fields(row):
                return True
    return False


def screen_business_output(report: str, evidence: list[dict] | None = None) -> OutputGuardDecision:
    """Block user-visible reports/evidence with direct identifiers or credentials.

    This function is deliberately local-only: it never sends report data to an API.
    It only catches known direct-disclosure patterns; it is not a complete privacy audit.
    """
    evidence = evidence if isinstance(evidence, list) else []
    if not isinstance(report, str) or not report.strip():
        return OutputGuardDecision("REVIEW", "local_rules_v1", ("empty_output",))

    reasons: set[str] = set()
    if _sensitive_table_columns(evidence) or _has_sensitive_fields(evidence):
        reasons.add("raw_sensitive_record_field")

    text = "\n".join([report, *_strings(evidence)])
    if _EMAIL_RE.search(text) or _SSN_RE.search(text) or _PHONE_RE.search(text):
        reasons.add("personal_contact_or_identity_data")
    normalized_report = report.casefold()
    if any(marker in normalized_report for marker in _SYSTEM_PROMPT_MARKERS):
        reasons.add("system_prompt_disclosure")
    if _RECORD_ID_RE.search(text):
        reasons.add("direct_customer_or_transaction_identifier")
    if any(pattern.search(text) for pattern in _SECRET_RES):
        reasons.add("credential_or_access_token")
    if any(_luhn_valid(candidate) for candidate in re.findall(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", text)):
        reasons.add("payment_card_number")

    if reasons:
        return OutputGuardDecision("BLOCK", "local_rules_v1", tuple(sorted(reasons)))
    return OutputGuardDecision("ALLOW", "local_rules_v1", ())
