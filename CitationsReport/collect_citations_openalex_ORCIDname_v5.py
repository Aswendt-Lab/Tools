#!/usr/bin/env python3
"""
OpenAlex citation report generator.

Updates in v4:
- seaborn "colorblind" palette
- every publication starts on a new PDF page
- full PubMed-style citation for each source publication, DOI clickable
- citations-per-year plot starts at 2010 and has NO trend fit
- citation summary next to each citations-per-year plot
- country pie charts use the colorblind palette and include a legend
- explicit "Cited by" heading
- cited-by publications are rendered in PubMed style with DOI links
- authors are stored in CSV files
- journal impact-like metrics are matched from a local SCImagoJR 2025 CSV
  (Citations / Doc. (2years), plus SJR and quartile)
- matplotlib uses the non-interactive Agg backend (safe on macOS/headless)

Journal metric note
-------------------
This version reads the local SCImagoJR export ``scimagojr 2025.csv``.
SCImago does NOT contain the official Clarivate Journal Impact Factor.
The script therefore stores ``Citations / Doc. (2years)`` in the requested
``impact_factor`` column as an impact-like 2-year citation metric and records
its exact source in ``impact_factor_source``. It also stores SJR and quartile.
This avoids silently claiming that the SCImago value is an official JIF.
"""

import argparse
import datetime as dt
import html
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from difflib import SequenceMatcher

import requests
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image as PILImage

try:
    import seaborn as sns
except ImportError as exc:
    raise SystemExit("This version requires seaborn: pip install seaborn") from exc

# ReportLab imports are kept here so PDF-related failures occur early.
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

BASE_URL = "https://api.openalex.org"
OUTDIR = Path("openalex_citations_output")
OUTDIR.mkdir(parents=True, exist_ok=True)

# SCImago journal metric file. By default the script looks beside itself and
# in the current working directory. It can also be overridden with --scimago_csv.
SCRIPT_DIR = Path(__file__).resolve().parent
SCIMAGO_DEFAULT_NAME = "scimagojr 2025.csv"
SCIMAGO_FILE: Optional[Path] = None
SCIMAGO_YEAR = 2025

CURRENT_YEAR = dt.datetime.now().year
FIXED_X_START_YEAR = 2010

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Requested palette.
sns.set_theme(style="whitegrid", palette="colorblind")
COLORBLIND = sns.color_palette("colorblind", 10)

# Cache journal metrics because the same journal occurs repeatedly.
_JOURNAL_METRIC_CACHE: Dict[str, Dict[str, Any]] = {}
_SCIMAGO_DF: Optional[pd.DataFrame] = None
_SCIMAGO_WARNING_PRINTED = False
_CROSSREF_CACHE: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# GENERAL HELPERS
# -----------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """Unescape HTML, remove tags, collapse whitespace, and normalize missing values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_para(value: Any) -> str:
    """Escape externally sourced text before inserting it into ReportLab Paragraph HTML."""
    return html.escape(clean_text(value), quote=False)


def safe_filename(value: Any) -> str:
    text = clean_text(value) or "unnamed"
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("._") or "unnamed"


def doi_text_and_url(value: Any) -> Tuple[str, str]:
    doi = clean_text(value)
    if not doi:
        return "", ""
    doi = re.sub(r"^doi\s*:\s*", "", doi, flags=re.I)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    return doi, f"https://doi.org/{doi}"


def pmid_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    matches = re.findall(r"\d+", text)
    return matches[-1] if matches else text


def get_json(url: str, params: Optional[dict] = None, max_retries: int = 5, pause: float = 1.0):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=45)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                wait = pause * (attempt + 1)
                print(f"⏱️ OpenAlex rate limit. Waiting {wait:.1f} s...")
                time.sleep(wait)
                continue
            print(f"⚠️ HTTP {response.status_code}: {response.url}")
            return None
        except requests.RequestException as exc:
            print(f"⚠️ Request failed ({attempt + 1}/{max_retries}): {exc}")
            time.sleep(pause * (attempt + 1))
    return None


# -----------------------------------------------------------------------------
# AUTHOR AND PUBMED-STYLE CITATION HELPERS
# -----------------------------------------------------------------------------

def format_author_nlm(display_name: Any) -> str:
    """Convert an OpenAlex display name to NLM/PubMed-like 'Surname AB'."""
    name = clean_text(display_name)
    if not name:
        return ""

    if "," in name:
        surname, given = [part.strip() for part in name.split(",", 1)]
        initials = "".join(
            token[0].upper()
            for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", given)
            if token
        )
        return f"{surname} {initials}".strip()

    parts = name.split()
    if len(parts) == 1:
        return parts[0]

    # Preserve common surname particles with the last token where possible.
    particles = {"de", "del", "della", "der", "van", "von", "da", "di", "la", "le"}
    surname_parts = [parts[-1]]
    idx = len(parts) - 2
    while idx >= 0 and parts[idx].lower() in particles:
        surname_parts.insert(0, parts[idx])
        idx -= 1
    given_parts = parts[: idx + 1]
    surname = " ".join(surname_parts)
    initials = "".join(p[0].upper() for p in given_parts if p)
    return f"{surname} {initials}".strip()


def format_authors_nlm(authorships: Iterable[dict]) -> str:
    names: List[str] = []
    for authorship in authorships or []:
        display_name = ((authorship or {}).get("author") or {}).get("display_name")
        formatted = format_author_nlm(display_name)
        if formatted:
            names.append(formatted)
    return ", ".join(names)


def extract_source(work: dict) -> dict:
    """Return the best available source/venue from an OpenAlex work."""
    candidates = [
        (work.get("primary_location") or {}).get("source") or {},
        (work.get("best_oa_location") or {}).get("source") or {},
    ]
    for loc in work.get("locations") or []:
        candidates.append((loc or {}).get("source") or {})
    for source in candidates:
        if clean_text(source.get("display_name")):
            return source
    return {}


def extract_pages(biblio: dict) -> str:
    first_page = clean_text((biblio or {}).get("first_page"))
    last_page = clean_text((biblio or {}).get("last_page"))
    if first_page and last_page and first_page != last_page:
        return f"{first_page}-{last_page}"
    return first_page or last_page


def parse_publication_date(value: Any) -> Tuple[str, str, str]:
    text = clean_text(value)
    if not text:
        return "", "", ""
    try:
        d = dt.date.fromisoformat(text[:10])
        return str(d.year), d.strftime("%b"), str(d.day)
    except ValueError:
        return "", "", ""


def format_pubmed_citation_plain(meta: dict, include_identifiers: bool = True) -> str:
    """Create a readable PubMed/NLM-style citation as plain text."""
    authors = clean_text(meta.get("authors"))
    title = clean_text(meta.get("title"))
    journal = clean_text(meta.get("journal"))
    year = clean_text(meta.get("pub_year") or meta.get("year"))
    month = clean_text(meta.get("pub_month"))
    day = clean_text(meta.get("pub_day"))
    volume = clean_text(meta.get("volume"))
    issue = clean_text(meta.get("issue"))
    pages = clean_text(meta.get("pages"))
    doi, _ = doi_text_and_url(meta.get("doi"))
    pmid = pmid_text(meta.get("pmid"))

    chunks: List[str] = []
    if authors:
        chunks.append(authors.rstrip(".") + ".")
    if title:
        chunks.append(title.rstrip(".") + ".")
    if journal:
        chunks.append(journal.rstrip(".") + ".")

    tail = ""
    if year:
        tail += year
        if month:
            tail += f" {month}"
        if day:
            tail += f" {day}"
        tail += ";"

    if volume:
        tail += volume
        if issue:
            tail += f"({issue})"
    elif issue:
        tail += f"({issue})"

    if pages:
        tail += f":{pages}"
    if tail:
        chunks.append(tail.rstrip(".") + ".")

    if include_identifiers and doi:
        chunks.append(f"doi: {doi}.")
    if include_identifiers and pmid:
        chunks.append(f"PMID: {pmid}.")

    return " ".join(chunks).replace("..", ".")


def format_pubmed_citation_html(meta: dict, include_identifiers: bool = True) -> str:
    """PubMed-style citation with a clickable DOI for ReportLab."""
    authors = safe_para(meta.get("authors"))
    title = safe_para(meta.get("title"))
    journal = safe_para(meta.get("journal"))
    year = safe_para(meta.get("pub_year") or meta.get("year"))
    month = safe_para(meta.get("pub_month"))
    day = safe_para(meta.get("pub_day"))
    volume = safe_para(meta.get("volume"))
    issue = safe_para(meta.get("issue"))
    pages = safe_para(meta.get("pages"))
    doi, doi_url = doi_text_and_url(meta.get("doi"))
    pmid = pmid_text(meta.get("pmid"))

    parts: List[str] = []
    if authors:
        parts.append(authors.rstrip(".") + ".")
    if title:
        parts.append(title.rstrip(".") + ".")
    if journal:
        parts.append(f"<i>{journal.rstrip('.')}</i>.")

    tail = ""
    if year:
        tail += year
        if month:
            tail += f" {month}"
        if day:
            tail += f" {day}"
        tail += ";"
    if volume:
        tail += volume
        if issue:
            tail += f"({issue})"
    elif issue:
        tail += f"({issue})"
    if pages:
        tail += f":{pages}"
    if tail:
        parts.append(tail.rstrip(".") + ".")

    if include_identifiers and doi:
        parts.append(
            f"doi: <a href='{html.escape(doi_url, quote=True)}'>{safe_para(doi)}</a>."
        )
    if include_identifiers and pmid:
        parts.append(f"PMID: {safe_para(pmid)}.")

    return " ".join(parts).replace("..", ".")


# -----------------------------------------------------------------------------
# SCIMAGO JOURNAL METRICS + BIBLIOGRAPHIC FALLBACK
# -----------------------------------------------------------------------------

def normalize_issn(value: Any) -> str:
    return re.sub(r"[^0-9Xx]", "", clean_text(value)).upper()


def normalize_journal_name(value: Any) -> str:
    text = clean_text(value).casefold()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_scimago_file() -> Optional[Path]:
    if SCIMAGO_FILE:
        p = Path(SCIMAGO_FILE)
        return p if p.exists() else None
    candidates = [Path.cwd() / SCIMAGO_DEFAULT_NAME, SCRIPT_DIR / SCIMAGO_DEFAULT_NAME]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_scimago_file() -> Optional[pd.DataFrame]:
    global _SCIMAGO_DF, _SCIMAGO_WARNING_PRINTED
    if _SCIMAGO_DF is not None:
        return _SCIMAGO_DF

    path = _find_scimago_file()
    if not path:
        if not _SCIMAGO_WARNING_PRINTED:
            print(
                f"⚠️ SCImago file '{SCIMAGO_DEFAULT_NAME}' not found. Journal metric columns will be blank. "
                "Place it beside the script or use --scimago_csv PATH."
            )
            _SCIMAGO_WARNING_PRINTED = True
        return None

    try:
        # Official SCImago exports are semicolon-separated and use decimal commas.
        df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        if "Title" not in df.columns:
            raise ValueError("SCImago CSV does not contain a 'Title' column")

        df["_journal_norm"] = df["Title"].map(normalize_journal_name)
        if "Issn" in df.columns:
            df["_issn_set"] = df["Issn"].fillna("").map(
                lambda x: {normalize_issn(v) for v in str(x).split(",") if normalize_issn(v)}
            )
        else:
            df["_issn_set"] = [set() for _ in range(len(df))]

        _SCIMAGO_DF = df
        print(f"📊 Loaded SCImago journal metrics: {path} ({len(df):,} journals)")
        return df
    except Exception as exc:
        print(f"⚠️ Could not read SCImago file {path}: {exc}")
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return float(value)
    txt = clean_text(value).replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


def lookup_scimago_metrics(
    journal: str,
    issn_l: str = "",
    issns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Match an OpenAlex journal to the SCImago 2025 export.

    Matching priority:
      1) exact ISSN match
      2) exact normalized journal title
      3) conservative fuzzy title match (>= 0.93)

    ``impact_factor`` is populated from SCImago's "Citations / Doc. (2years)"
    because the user requested an impact-factor-like value from this file. This
    is NOT the official Clarivate JIF; the source column says so explicitly.
    """
    journal = clean_text(journal)
    query_issns = {normalize_issn(issn_l)} if normalize_issn(issn_l) else set()
    query_issns |= {normalize_issn(x) for x in (issns or []) if normalize_issn(x)}
    cache_key = "|".join(sorted(query_issns)) or normalize_journal_name(journal)
    if not cache_key:
        return {}
    if cache_key in _JOURNAL_METRIC_CACHE:
        return dict(_JOURNAL_METRIC_CACHE[cache_key])

    df = _load_scimago_file()
    if df is None or df.empty:
        return {}

    match = None
    match_method = ""

    if query_issns:
        mask = df["_issn_set"].map(lambda s: bool(s & query_issns))
        candidates = df[mask]
        if not candidates.empty:
            match = candidates.iloc[0]
            match_method = "ISSN"

    qnorm = normalize_journal_name(journal)
    if match is None and qnorm:
        exact = df[df["_journal_norm"] == qnorm]
        if not exact.empty:
            match = exact.iloc[0]
            match_method = "title-exact"

    if match is None and qnorm:
        # Conservative fallback for capitalization/punctuation/abbreviation differences.
        # To keep runtime reasonable, prefilter by first alphanumeric token.
        first = qnorm.split()[0] if qnorm.split() else ""
        candidates = df[df["_journal_norm"].str.startswith(first, na=False)] if first else df
        best_ratio = 0.0
        best_row = None
        for _, rec in candidates.iterrows():
            ratio = SequenceMatcher(None, qnorm, rec["_journal_norm"]).ratio()
            if ratio > best_ratio:
                best_ratio, best_row = ratio, rec
        if best_row is not None and best_ratio >= 0.93:
            match = best_row
            match_method = f"title-fuzzy:{best_ratio:.3f}"

    if match is None:
        result = {
            "impact_factor": None,
            "impact_factor_year": SCIMAGO_YEAR,
            "impact_factor_source": "SCImagoJR 2025: Citations / Doc. (2years); not Clarivate JIF",
            "sjr_2025": None,
            "sjr_quartile_2025": "",
            "scimago_title": "",
            "scimago_match_method": "unmatched",
        }
    else:
        result = {
            "impact_factor": _to_float(match.get("Citations / Doc. (2years)")),
            "impact_factor_year": SCIMAGO_YEAR,
            "impact_factor_source": "SCImagoJR 2025: Citations / Doc. (2years); not Clarivate JIF",
            "sjr_2025": _to_float(match.get("SJR")),
            "sjr_quartile_2025": clean_text(match.get("SJR Best Quartile")),
            "scimago_title": clean_text(match.get("Title")),
            "scimago_match_method": match_method,
        }

    _JOURNAL_METRIC_CACHE[cache_key] = dict(result)
    return result


def crossref_enrich_metadata(meta: dict) -> dict:
    """Fill missing bibliographic fields from Crossref when a DOI is available.

    OpenAlex usually contains all required information, but some citing works have
    incomplete ``primary_location`` data. Crossref is only queried when one or more
    core citation fields are missing, and never overwrites existing values.
    """
    doi, _ = doi_text_and_url(meta.get("doi"))
    missing_core = not clean_text(meta.get("journal")) or not clean_text(meta.get("authors"))
    missing_biblio = not clean_text(meta.get("volume")) and not clean_text(meta.get("pages"))
    if not doi or not (missing_core or missing_biblio):
        return meta

    if doi in _CROSSREF_CACHE:
        message = _CROSSREF_CACHE[doi]
    else:
        try:
            url = f"https://api.crossref.org/works/{requests.utils.quote(doi, safe='')}"
            response = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "OpenAlexCitationReport/5.0 (bibliographic metadata enrichment)"},
            )
            if response.status_code != 200:
                return meta
            message = (response.json() or {}).get("message") or {}
            _CROSSREF_CACHE[doi] = message
            time.sleep(0.05)
        except requests.RequestException:
            return meta

    if not clean_text(meta.get("journal")):
        containers = message.get("container-title") or []
        if containers:
            meta["journal"] = clean_text(containers[0])
    if not clean_text(meta.get("title")):
        titles = message.get("title") or []
        if titles:
            meta["title"] = clean_text(titles[0])
    if not clean_text(meta.get("authors")):
        names = []
        for a in message.get("author") or []:
            family = clean_text(a.get("family"))
            given = clean_text(a.get("given"))
            initials = "".join(token[0].upper() for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", given))
            name = f"{family} {initials}".strip()
            if name:
                names.append(name)
        if names:
            meta["authors"] = ", ".join(names)
    if not clean_text(meta.get("volume")):
        meta["volume"] = clean_text(message.get("volume"))
    if not clean_text(meta.get("issue")):
        meta["issue"] = clean_text(message.get("issue"))
    if not clean_text(meta.get("pages")):
        meta["pages"] = clean_text(message.get("page") or message.get("article-number"))

    # Crossref date fallback if OpenAlex has no date.
    if not clean_text(meta.get("pub_year")):
        date_parts = ((message.get("published-print") or message.get("published-online") or {}).get("date-parts") or [])
        if date_parts and date_parts[0]:
            vals = date_parts[0]
            meta["pub_year"] = str(vals[0]) if len(vals) >= 1 else ""
            if len(vals) >= 2:
                try:
                    meta["pub_month"] = dt.date(2000, int(vals[1]), 1).strftime("%b")
                except Exception:
                    pass
            if len(vals) >= 3:
                meta["pub_day"] = str(vals[2])

    return meta

# -----------------------------------------------------------------------------
# OPENALEX AUTHOR + WORK RETRIEVAL
# -----------------------------------------------------------------------------

def find_author_by_orcid(orcid: str):
    orcid = clean_text(orcid)
    if not orcid:
        return None
    data = get_json(f"{BASE_URL}/authors", {"filter": f"orcid:{orcid}"})
    results = (data or {}).get("results") or []
    return results[0] if results else None


def find_authors_by_name(name: str, max_results: int = 5):
    data = get_json(
        f"{BASE_URL}/authors",
        {"search": name, "per-page": max_results, "sort": "cited_by_count:desc"},
    )
    return (data or {}).get("results") or []


def pick_best_author(name: str, orcid: str = ""):
    if orcid:
        print(f"🔍 Searching OpenAlex by ORCID {orcid} ...")
        author = find_author_by_orcid(orcid)
        if author:
            print(f"✅ Found author: {author.get('display_name')}")
            return author
        print("⚠️ ORCID not found; falling back to name search.")

    print(f"🔍 Searching OpenAlex by name '{name}' ...")
    candidates = find_authors_by_name(name)
    if not candidates:
        return None

    print("\nPossible matches:")
    for i, author in enumerate(candidates):
        print(
            f"  [{i}] {author.get('display_name', 'Unknown')} | "
            f"ORCID={author.get('orcid', 'n/a')} | works={author.get('works_count', 0)} | "
            f"cited_by={author.get('cited_by_count', 0)}"
        )
    selected = candidates[0]
    print(f"➡️ Selecting {selected.get('display_name')}.")
    return selected


def fetch_works_for_author(author_id: str, per_page: int = 200) -> List[dict]:
    works: List[dict] = []
    page = 1
    while True:
        data = get_json(
            f"{BASE_URL}/works",
            {
                "filter": f"authorships.author.id:{author_id}",
                "per-page": per_page,
                "page": page,
                "sort": "publication_year:asc",
            },
        )
        results = (data or {}).get("results") or []
        if not results:
            break
        works.extend(results)
        print(f"  Retrieved page {page}: {len(results)} works; total {len(works)}")
        if len(results) < per_page:
            break
        page += 1
    return works


def fetch_citing_works(openalex_id: str, per_page: int = 200) -> List[dict]:
    citing: List[dict] = []
    page = 1
    short_id = clean_text(openalex_id).split("/")[-1]
    print(f"\n🔎 Analyzing citing works for: {short_id}")
    while True:
        data = get_json(
            f"{BASE_URL}/works",
            {
                "filter": f"cites:{openalex_id}",
                "per-page": per_page,
                "page": page,
                "sort": "publication_year:asc",
            },
        )
        results = (data or {}).get("results") or []
        if not results:
            break
        citing.extend(results)
        print(f"    citing page {page}: {len(results)} works (total so far: {len(citing)})")
        if len(results) < per_page:
            break
        page += 1
    return citing


def work_to_metadata(work: dict, collect_jif: bool = True) -> dict:
    source = extract_source(work)
    biblio = work.get("biblio") or {}
    pub_year, pub_month, pub_day = parse_publication_date(work.get("publication_date"))
    if not pub_year:
        pub_year = clean_text(work.get("publication_year"))

    ids = work.get("ids") or {}
    doi = work.get("doi") or ids.get("doi") or ""
    pmid = ids.get("pmid") or ""

    journal = clean_text(source.get("display_name"))
    issn_l = clean_text(source.get("issn_l"))
    issns = source.get("issn") or []
    if isinstance(issns, str):
        issns = [issns]

    meta = {
        "openalex_id": clean_text(work.get("id")),
        "title": clean_text(work.get("title") or "Untitled"),
        "authors": format_authors_nlm(work.get("authorships") or []),
        "year": clean_text(work.get("publication_year")),
        "publication_date": clean_text(work.get("publication_date")),
        "pub_year": pub_year,
        "pub_month": pub_month,
        "pub_day": pub_day,
        "journal": journal,
        "journal_issn_l": issn_l,
        "journal_issns": ";".join(clean_text(x) for x in issns if clean_text(x)),
        "volume": clean_text(biblio.get("volume")),
        "issue": clean_text(biblio.get("issue")),
        "pages": extract_pages(biblio),
        "doi": doi_text_and_url(doi)[0],
        "doi_link": doi_text_and_url(doi)[1],
        "pmid": pmid_text(pmid),
        "cited_by_count": int(work.get("cited_by_count") or 0),
    }

    # Fill bibliographic holes that otherwise produce incomplete "Cited by" entries.
    meta = crossref_enrich_metadata(meta)

    if collect_jif and clean_text(meta.get("journal")):
        journal_metrics = lookup_scimago_metrics(
            meta.get("journal", ""),
            meta.get("journal_issn_l", ""),
            [x for x in str(meta.get("journal_issns", "")).split(";") if x],
        )
    else:
        journal_metrics = {
            "impact_factor": None,
            "impact_factor_year": SCIMAGO_YEAR,
            "impact_factor_source": "",
            "sjr_2025": None,
            "sjr_quartile_2025": "",
            "scimago_title": "",
            "scimago_match_method": "",
        }
    meta.update(journal_metrics)

    meta["pubmed_citation"] = format_pubmed_citation_plain(meta)
    return meta


# -----------------------------------------------------------------------------
# CITATION ANALYSIS
# -----------------------------------------------------------------------------

def analyze_citations_over_time(citing_works: List[dict]) -> pd.DataFrame:
    rows = [
        {"year": int(w["publication_year"]), "count": 1}
        for w in citing_works
        if w.get("publication_year") is not None
    ]
    if not rows:
        return pd.DataFrame(columns=["year", "count"])
    return (
        pd.DataFrame(rows)
        .groupby("year", as_index=False)["count"]
        .sum()
        .sort_values("year")
    )


def first_author_country(work: dict) -> str:
    # Use the first authorship that contains a country-coded institution.
    for authorship in work.get("authorships") or []:
        for institution in authorship.get("institutions") or []:
            country = clean_text(institution.get("country_code") or institution.get("country"))
            if country:
                return country
    return "Unknown"


def analyze_citations_by_country(citing_works: List[dict]) -> pd.DataFrame:
    if not citing_works:
        return pd.DataFrame(columns=["country", "count"])
    df = pd.DataFrame(
        [{"country": first_author_country(w), "count": 1} for w in citing_works]
    )
    return (
        df.groupby("country", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )


def citation_summary(total: int, publication_year: Any) -> dict:
    try:
        pub_year = int(float(publication_year))
    except (TypeError, ValueError):
        pub_year = CURRENT_YEAR
    years = max(CURRENT_YEAR - pub_year + 1, 1)
    avg = total / years
    return {
        "total": int(total),
        "publication_year": pub_year,
        "years_observed": years,
        "average_per_year": avg,
    }


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------

def plot_citations_per_year(df_year: pd.DataFrame, title_stub: str, outdir: Path = OUTDIR):
    if df_year is None or df_year.empty:
        return None

    df_year = df_year.sort_values("year")
    years = df_year["year"].astype(int).to_numpy()
    counts = df_year["count"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=170)
    ax.bar(years, counts, width=0.72, color=COLORBLIND[0])
    ax.set_xlabel("Year")
    ax.set_ylabel("Citations")
    ax.set_title("Citations per year")

    end_year = max(CURRENT_YEAR, int(years.max()))
    ax.set_xlim(FIXED_X_START_YEAR - 0.5, end_year + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))

    ymax = max(int(counts.max()), 1)
    ax.set_ylim(0, max(1.0, ymax * 1.12))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)

    path = outdir / f"{safe_filename(title_stub)}_citations_per_year.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_citations_by_country(df_country: pd.DataFrame, title_stub: str, outdir: Path = OUTDIR):
    """Compact country pie + separate legend panel; pie remains perfectly circular."""
    if df_country is None or df_country.empty:
        return None

    df = df_country.copy().sort_values("count", ascending=False).reset_index(drop=True)
    if len(df) > 9:
        top = df.iloc[:8].copy()
        other_count = int(df.iloc[8:]["count"].sum())
        df = pd.concat(
            [top, pd.DataFrame([{"country": "Other", "count": other_count}])],
            ignore_index=True,
        )

    palette = sns.color_palette("colorblind", n_colors=max(len(df), 3))
    fig = plt.figure(figsize=(6.2, 3.25), dpi=170)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    wedges, _texts, _autotexts = ax.pie(
        df["count"],
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
        startangle=90,
        colors=palette[: len(df)],
        pctdistance=0.67,
        radius=0.93,
        textprops={"fontsize": 8},
        wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
    )
    ax.set_title("Citations by country", fontsize=11, pad=5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    total = int(df["count"].sum())
    legend_labels = [
        f"{country}: {count} ({count / total * 100:.1f}%)"
        for country, count in zip(df["country"], df["count"])
    ]
    handles = [Patch(facecolor=palette[i], edgecolor="none") for i in range(len(df))]
    ax_leg.legend(
        handles,
        legend_labels,
        title="Country",
        loc="center left",
        frameon=False,
        fontsize=8.2,
        title_fontsize=9.2,
        borderaxespad=0,
        handlelength=1.5,
    )
    ax_leg.axis("off")

    path = outdir / f"{safe_filename(title_stub)}_citations_by_country_pie.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return str(path)


# -----------------------------------------------------------------------------
# PDF REPORT
# -----------------------------------------------------------------------------

def _image_if_exists(path: Any, width_mm: float, height_mm: float):
    """Fit an image inside a box without changing its aspect ratio."""
    if isinstance(path, str) and os.path.exists(path):
        try:
            with PILImage.open(path) as im:
                px_w, px_h = im.size
            if px_w > 0 and px_h > 0:
                scale = min((width_mm * mm) / px_w, (height_mm * mm) / px_h)
                return Image(path, width=px_w * scale, height=px_h * scale)
        except Exception:
            pass
        return Image(path, width=width_mm * mm, height=height_mm * mm)
    return Spacer(1, 1)


def create_pdf_report(
    author_name: str,
    orcid: str,
    summary_df: pd.DataFrame,
    agg_year_plot: Optional[str],
    agg_country_plot: Optional[str],
    total_citations: int,
):
    clean_orcid = safe_filename(orcid or "noORCID")
    clean_name = safe_filename(author_name)
    pdf_path = OUTDIR / f"Report_Citations_{clean_orcid}_{clean_name}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"Citation report - {author_name}",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PublicationLabel",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CitationText",
            parent=styles["Normal"],
            fontSize=9.3,
            leading=12.2,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CitedByCitation",
            parent=styles["Normal"],
            fontSize=8.2,
            leading=10.4,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricBox",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
        )
    )

    elements: List[Any] = []

    # ---------------- COVER PAGE ----------------
    elements.extend(
        [
            Paragraph("Author Citation Report", styles["Title"]),
            Spacer(1, 5 * mm),
            Paragraph(f"<b>Author:</b> {safe_para(author_name)}", styles["Normal"]),
            Paragraph(f"<b>ORCID:</b> {safe_para(orcid or 'n/a')}", styles["Normal"]),
            Paragraph(f"<b>Total citations:</b> {int(total_citations)}", styles["Normal"]),
            Paragraph(
                f"<b>Generated:</b> {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                styles["Normal"],
            ),
            Spacer(1, 7 * mm),
        ]
    )

    if agg_year_plot and os.path.exists(agg_year_plot):
        # Average across the interval starting at the first included publication year.
        valid_years = pd.to_numeric(summary_df.get("year"), errors="coerce").dropna()
        first_pub_year = int(valid_years.min()) if len(valid_years) else CURRENT_YEAR
        stats = citation_summary(total_citations, first_pub_year)
        metric_html = (
            f"<b>Summary</b><br/>"
            f"Total citations: <b>{stats['total']}</b><br/>"
            f"Average/year: <b>{stats['average_per_year']:.2f}</b><br/>"
            f"Period: {stats['publication_year']}-{CURRENT_YEAR}"
        )
        cover_table = Table(
            [[_image_if_exists(agg_year_plot, 128, 62), Paragraph(metric_html, styles["MetricBox"]) ]],
            colWidths=[135 * mm, 42 * mm],
            hAlign="LEFT",
        )
        cover_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#BBBBBB")),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F7F7F7")),
                    ("LEFTPADDING", (1, 0), (1, 0), 6),
                    ("RIGHTPADDING", (1, 0), (1, 0), 6),
                    ("TOPPADDING", (1, 0), (1, 0), 6),
                    ("BOTTOMPADDING", (1, 0), (1, 0), 6),
                ]
            )
        )
        elements.append(cover_table)
        elements.append(Spacer(1, 5 * mm))

    if agg_country_plot and os.path.exists(agg_country_plot):
        elements.append(_image_if_exists(agg_country_plot, 150, 78))

    # Every publication begins on a new page.
    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df.iterrows():
            elements.append(PageBreak())

            meta = {
                "authors": row.get("authors", ""),
                "title": row.get("title", ""),
                "journal": row.get("journal", ""),
                "pub_year": row.get("pub_year", row.get("year", "")),
                "pub_month": row.get("pub_month", ""),
                "pub_day": row.get("pub_day", ""),
                "volume": row.get("volume", ""),
                "issue": row.get("issue", ""),
                "pages": row.get("pages", ""),
                "doi": row.get("doi", ""),
                "pmid": row.get("pmid", ""),
            }

            citation_html = format_pubmed_citation_html(meta)
            total_pub = int(row.get("cited_by_count") or 0)
            stats = citation_summary(total_pub, row.get("pub_year") or row.get("year"))
            jif = row.get("impact_factor")
            jif_year = row.get("impact_factor_year")

            metric_lines = [
                "<b>Citation summary</b>",
                f"Total citations: <b>{stats['total']}</b>",
                f"Average/year: <b>{stats['average_per_year']:.2f}</b>",
                f"Period: {stats['publication_year']}-{CURRENT_YEAR}",
            ]
            if pd.notna(jif) and clean_text(jif):
                jif_label = f"SCImago cites/doc 2y ({int(jif_year)}):" if pd.notna(jif_year) else "SCImago cites/doc 2y:"
                metric_lines.append(f"{jif_label} <b>{float(jif):.3g}</b>")
                sjr = row.get("sjr_2025")
                quartile = clean_text(row.get("sjr_quartile_2025"))
                if pd.notna(sjr) and clean_text(sjr):
                    metric_lines.append(f"SJR 2025: <b>{float(sjr):.3g}</b>{f' ({quartile})' if quartile else ''}")
            metric_html = "<br/>".join(metric_lines)

            year_plot = row.get("year_plot")
            country_plot = row.get("country_plot")

            year_summary_table = Table(
                [[_image_if_exists(year_plot, 128, 62), Paragraph(metric_html, styles["MetricBox"]) ]],
                colWidths=[135 * mm, 42 * mm],
                hAlign="LEFT",
            )
            year_summary_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#BBBBBB")),
                        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F7F7F7")),
                        ("LEFTPADDING", (1, 0), (1, 0), 6),
                        ("RIGHTPADDING", (1, 0), (1, 0), 6),
                        ("TOPPADDING", (1, 0), (1, 0), 6),
                        ("BOTTOMPADDING", (1, 0), (1, 0), 6),
                    ]
                )
            )

            # Every publication already starts on a fresh page. Keep only the citation
            # and year-summary block together; allow the compact pie to flow naturally.
            elements.append(KeepTogether([
                Paragraph("Publication", styles["PublicationLabel"]),
                Paragraph(citation_html, styles["CitationText"]),
                Spacer(1, 2 * mm),
                year_summary_table,
                Spacer(1, 3 * mm),
            ]))
            if isinstance(country_plot, str) and os.path.exists(country_plot):
                elements.append(_image_if_exists(country_plot, 150, 78))
                elements.append(Spacer(1, 3 * mm))

            # Cited-by section with an explicit heading.
            elements.append(Paragraph("Cited by", styles["Heading2"]))
            elements.append(Spacer(1, 1.5 * mm))

            detailed_csv = row.get("detailed_csv")
            if isinstance(detailed_csv, str) and os.path.exists(detailed_csv):
                try:
                    cited_df = pd.read_csv(detailed_csv)
                except Exception as exc:
                    elements.append(Paragraph(f"Could not read cited-by CSV: {safe_para(exc)}", styles["Normal"]))
                    continue

                if cited_df.empty:
                    elements.append(Paragraph("No citing works found.", styles["Normal"]))
                else:
                    for i, cited in cited_df.iterrows():
                        cited_meta = {
                            "authors": cited.get("authors", ""),
                            "title": cited.get("title", ""),
                            "journal": cited.get("journal", "") or "[journal/source unavailable]",
                            "pub_year": cited.get("pub_year", cited.get("year", "")),
                            "pub_month": cited.get("pub_month", ""),
                            "pub_day": cited.get("pub_day", ""),
                            "volume": cited.get("volume", ""),
                            "issue": cited.get("issue", ""),
                            "pages": cited.get("pages", ""),
                            "doi": cited.get("doi", ""),
                            "pmid": cited.get("pmid", ""),
                        }
                        cited_html = format_pubmed_citation_html(cited_meta)
                        elements.append(
                            Paragraph(f"<b>{i + 1}.</b> {cited_html}", styles["CitedByCitation"])
                        )
            else:
                elements.append(Paragraph("No citing-works file available.", styles["Normal"]))

    doc.build(elements)
    print(f"\n📄 PDF report created: {pdf_path}")
    return str(pdf_path)


# -----------------------------------------------------------------------------
# MARKDOWN EMBED FOR GITHUB / GITHUB PAGES
# -----------------------------------------------------------------------------

def create_markdown_embed(pdf_path: str, author_name: str, total_citations: int) -> str:
    """Create a Markdown page next to the PDF with an inline GitHub Pages embed."""
    pdf = Path(pdf_path)
    md_path = pdf.with_suffix(".md")
    pdf_name = pdf.name
    content = f"""# Citation Report — {author_name}

**Total citations in report:** {int(total_citations)}

[Open / download the PDF](./{pdf_name})

<object data="./{pdf_name}" type="application/pdf" width="100%" height="900px">
  <p>Your browser cannot display the embedded PDF. <a href="./{pdf_name}">Open the PDF</a>.</p>
</object>

> On the regular GitHub repository view, the inline PDF object may not render. The link above still works. On GitHub Pages, the PDF can be displayed inline by the browser.
"""
    md_path.write_text(content, encoding="utf-8")
    print(f"📝 Markdown embed created: {md_path}")
    return str(md_path)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenAlex citation report generator")
    parser.add_argument("--name", type=str, help="Author display name")
    parser.add_argument("--orcid", type=str, help="Author ORCID")
    parser.add_argument(
        "--min_citations",
        type=int,
        default=1,
        help="Minimum OpenAlex cited_by_count for a work to include in detailed reporting",
    )
    parser.add_argument(
        "--skip_impact_factor",
        action="store_true",
        help="Do not merge SCImago journal metrics",
    )
    parser.add_argument(
        "--scimago_csv",
        type=str,
        default="",
        help=f"Path to SCImago CSV (default: '{SCIMAGO_DEFAULT_NAME}' beside script/current directory)",
    )
    args = parser.parse_args()

    global SCIMAGO_FILE
    if clean_text(args.scimago_csv):
        SCIMAGO_FILE = Path(args.scimago_csv).expanduser().resolve()

    print("🚀 OpenAlex citation collector starting...\n")

    name = clean_text(args.name)
    orcid = clean_text(args.orcid)
    if not name and not orcid:
        name = input("Author name (leave empty for ORCID-only search): ").strip()
        if not name:
            orcid = input("ORCID: ").strip()
    if not name and not orcid:
        raise SystemExit("No author name or ORCID provided.")

    author = pick_best_author(name, orcid)
    if not author:
        raise SystemExit("No matching OpenAlex author found.")

    author_name = clean_text(author.get("display_name")) or name or "Unknown Author"
    author_orcid = clean_text(author.get("orcid")) or orcid
    author_id = clean_text(author.get("id"))
    print(f"\n=== Selected author: {author_name} ({author_orcid or 'no ORCID'}) ===")

    print("\n📚 Fetching author works...")
    works = fetch_works_for_author(author_id)
    if not works:
        raise SystemExit("No works found for this author.")

    collect_jif = not args.skip_impact_factor
    works_meta = [work_to_metadata(w, collect_jif=collect_jif) for w in works]
    works_df = pd.DataFrame(works_meta)

    works_csv = OUTDIR / f"works_summary_{safe_filename(author_name)}.csv"
    works_df.to_csv(works_csv, index=False)
    print(f"💾 Saved works summary to {works_csv}")

    selected_df = works_df[
        pd.to_numeric(works_df["cited_by_count"], errors="coerce").fillna(0) >= args.min_citations
    ].copy()

    agg_year_frames: List[pd.DataFrame] = []
    agg_country_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    # Map OpenAlex ID back to full raw work, avoiding another API request.
    raw_by_id = {clean_text(w.get("id")): w for w in works}

    for _, work_meta_row in selected_df.iterrows():
        openalex_id = clean_text(work_meta_row.get("openalex_id"))
        title = clean_text(work_meta_row.get("title"))
        print(f"\n=== Processing work: {title} ===")

        citing_works = fetch_citing_works(openalex_id)
        if not citing_works:
            print(f"⚠️ No citing works found for {openalex_id}")
            continue

        year_df = analyze_citations_over_time(citing_works)
        country_df = analyze_citations_by_country(citing_works)
        if not year_df.empty:
            agg_year_frames.append(year_df)
        if not country_df.empty:
            agg_country_frames.append(country_df)

        # Stable unique label: journal + year + OpenAlex work id.
        journal_stub = safe_filename(work_meta_row.get("journal") or "journal")[:35]
        year_stub = clean_text(work_meta_row.get("year")) or "year"
        work_id_stub = openalex_id.split("/")[-1]
        label = f"{journal_stub}_{year_stub}_{work_id_stub}"

        detailed_rows: List[dict] = []
        for citing_work in citing_works:
            cited_meta = work_to_metadata(citing_work, collect_jif=collect_jif)
            cited_meta["country"] = first_author_country(citing_work)
            detailed_rows.append(cited_meta)

        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv = OUTDIR / f"{safe_filename(label)}_citations_detailed.csv"
        detailed_df.to_csv(detailed_csv, index=False)
        print(f"  💾 Saved detailed citing-works CSV: {detailed_csv}")

        year_plot = plot_citations_per_year(year_df, label)
        country_plot = plot_citations_by_country(country_df, label)

        summary = dict(work_meta_row)
        summary.update(
            {
                "label": label,
                "year_plot": year_plot,
                "country_plot": country_plot,
                "detailed_csv": str(detailed_csv),
                # Use the number actually returned by the citing query for report stats.
                "cited_by_count": len(citing_works),
            }
        )
        summary_rows.append(summary)

    if not summary_rows:
        raise SystemExit("No detailed citation data available for the selected works.")

    summary_df = pd.DataFrame(summary_rows)

    agg_label = f"{safe_filename(author_name)}_AGGREGATED"
    if agg_year_frames:
        agg_year_df = pd.concat(agg_year_frames, ignore_index=True)
        agg_year_df = agg_year_df.groupby("year", as_index=False)["count"].sum()
        total_citations = int(agg_year_df["count"].sum())
        agg_year_plot = plot_citations_per_year(agg_year_df, agg_label)
    else:
        total_citations = 0
        agg_year_plot = None

    if agg_country_frames:
        agg_country_df = pd.concat(agg_country_frames, ignore_index=True)
        agg_country_df = agg_country_df.groupby("country", as_index=False)["count"].sum()
        agg_country_plot = plot_citations_by_country(agg_country_df, agg_label)
    else:
        agg_country_plot = None

    pdf_path = create_pdf_report(
        author_name,
        author_orcid,
        summary_df,
        agg_year_plot,
        agg_country_plot,
        total_citations,
    )
    create_markdown_embed(pdf_path, author_name, total_citations)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
