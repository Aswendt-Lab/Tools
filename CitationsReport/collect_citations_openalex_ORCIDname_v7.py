#!/usr/bin/env python3
"""
OpenAlex citation report generator.

Updates in v6.2:
- optional per-publication reference-list export to Papers-compatible BibTeX
- reference fallback chain: OpenAlex → Europe PMC → Crossref
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
# REFERENCE-LIST / BIBTEX EXPORT
# -----------------------------------------------------------------------------

REFERENCE_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "from",
    "with", "by", "after", "before", "under", "using", "based", "between",
    "during", "via", "its", "their", "our",
}

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _openalex_short_id(value: Any) -> str:
    """Return W123... from either a full OpenAlex URL or an already-short ID."""
    text = clean_text(value)
    if not text:
        return ""
    return text.rstrip("/").split("/")[-1]


def _bibtex_escape(value: Any) -> str:
    """Conservative BibTeX escaping while keeping Unicode author/title text intact."""
    text = clean_text(value)
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _bibtex_authors(authorships: Iterable[dict]) -> str:
    """Return BibTeX author syntax: 'Author One and Author Two'."""
    names: List[str] = []
    for authorship in authorships or []:
        display_name = clean_text(((authorship or {}).get("author") or {}).get("display_name"))
        if display_name:
            names.append(display_name)
    return " and ".join(names)


def _authors_for_bibtex(work: dict) -> str:
    """Use structured OpenAlex/Crossref authors, with a fallback preformatted string."""
    structured = _bibtex_authors(work.get("authorships") or [])
    if structured:
        return structured
    return clean_text(work.get("bibtex_authors"))


def _first_author_surname_from_work(work: dict) -> str:
    authorships = work.get("authorships") or []
    name = ""
    if authorships:
        name = clean_text(((authorships[0] or {}).get("author") or {}).get("display_name"))
    if not name:
        pre = clean_text(work.get("bibtex_authors"))
        if pre:
            name = pre.split(" and ", 1)[0]
    if not name:
        return "Unknown"
    if "," in name:
        surname = name.split(",", 1)[0].strip()
    else:
        parts = name.split()
        surname = parts[-1] if parts else "Unknown"
    return safe_filename(surname)


def _abbreviated_title_for_filename(title: Any, max_words: int = 7, max_chars: int = 72) -> str:
    """Create a readable, stable title stub for per-publication .bib filenames."""
    words = re.findall(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", clean_text(title))
    kept: List[str] = []
    for word in words:
        if len(kept) >= max_words:
            break
        if kept and word.lower() in REFERENCE_STOPWORDS:
            continue
        kept.append(word)
    stub = "_".join(kept) if kept else "Untitled"
    stub = safe_filename(stub)
    return stub[:max_chars].rstrip("_") or "Untitled"


REFERENCE_JOURNAL_ABBREVIATIONS = {
    "translational stroke research": "TranslStrokeRes",
    "frontiers in neuroscience": "FrontNeurosci",
    "frontiers in cellular neuroscience": "FrontCellNeurosci",
    "frontiers in neuroinformatics": "FrontNeuroinform",
    "journal of cerebral blood flow & metabolism": "JCBFM",
    "journal of cerebral blood flow and metabolism": "JCBFM",
    "cns neuroscience & therapeutics": "CNSNeurosciTher",
    "cns neuroscience and therapeutics": "CNSNeurosciTher",
    "neural regeneration research": "NeuralRegenRes",
    "neuroimage": "NeuroImage",
    "neuroimage: clinical": "NeuroImageClin",
    "nature communications": "NatCommun",
    "nature reviews neuroscience": "NatRevNeurosci",
    "proceedings of the national academy of sciences": "PNAS",
    "proceedings of the national academy of sciences of the united states of america": "PNAS",
    "journal of neuroscience": "JNeurosci",
    "cerebral cortex": "CerebCortex",
    "brain structure and function": "BrainStructFunct",
    "brain imaging and behavior": "BrainImagingBehav",
    "european journal of neurology": "EurJNeurol",
    "plos one": "PLoSOne",
    "plos biology": "PLoSBiol",
    "cell": "Cell",
    "neuron": "Neuron",
    "neuroscience": "Neuroscience",
}


def _abbreviate_journal_name(journal: Any, max_chars: int = 22) -> str:
    """Create a short, readable journal token for Papers list/folder names."""
    raw = clean_text(journal)
    if not raw:
        return "UnknownJournal"

    key = raw.lower().strip()
    if key in REFERENCE_JOURNAL_ABBREVIATIONS:
        return REFERENCE_JOURNAL_ABBREVIATIONS[key]

    # Generic fallback: keep informative words and compress common journal terms.
    replacements = {
        "Journal": "J",
        "International": "Int",
        "European": "Eur",
        "American": "Am",
        "British": "Br",
        "Clinical": "Clin",
        "Experimental": "Exp",
        "Molecular": "Mol",
        "Cellular": "Cell",
        "Neuroscience": "Neurosci",
        "Neurology": "Neurol",
        "Neurobiology": "Neurobiol",
        "Neuroimaging": "Neuroimaging",
        "Research": "Res",
        "Therapy": "Ther",
        "Therapeutics": "Ther",
        "Medicine": "Med",
        "Magnetic": "Magn",
        "Resonance": "Reson",
        "Imaging": "Imaging",
        "Functional": "Funct",
        "Structural": "Struct",
        "Brain": "Brain",
        "Stroke": "Stroke",
    }
    stop = {"of", "the", "and", "&", "in", "for", "on", "a", "an"}
    words = re.findall(r"[A-Za-z0-9]+", raw)
    out = []
    for w in words:
        if w.lower() in stop:
            continue
        out.append(replacements.get(w, replacements.get(w.title(), w)))
    token = "".join(out) if out else "UnknownJournal"
    token = safe_filename(token).replace("_", "")
    return token[:max_chars] or "UnknownJournal"


def _source_publication_tag(source_work: dict) -> str:
    """Short Papers grouping tag: YEAR_FirstAuthor_AbbreviatedJournal."""
    year = clean_text(source_work.get("publication_year")) or "nd"
    first = _first_author_surname_from_work(source_work)
    source = extract_source(source_work)
    journal = clean_text(source.get("display_name") or source_work.get("journal"))
    journal_short = _abbreviate_journal_name(journal)
    return f"{year}_{first}_{journal_short}"


def _reference_bib_filename(source_work: dict) -> str:
    """Per-publication BibTeX filename, aligned with the short Papers list name."""
    return f"{_source_publication_tag(source_work)}.bib"


def _reference_bib_key(work: dict, index: int) -> str:
    """Generate a reasonably stable BibTeX key for a referenced work."""
    year = clean_text(work.get("publication_year")) or "nd"
    surname = _first_author_surname_from_work(work)
    title_words = re.findall(r"[A-Za-z0-9]+", clean_text(work.get("title")))
    token = "".join(title_words[:2])[:24] or "Work"
    key = f"{surname}{year}{token}{index}"
    key = re.sub(r"[^A-Za-z0-9_]", "", key)
    return key or f"Reference{index}"


def _bibtex_entry_from_openalex_work(work: dict, index: int) -> str:
    """Convert OpenAlex-like work metadata to a Papers-friendly BibTeX entry."""
    source = extract_source(work)
    biblio = work.get("biblio") or {}
    ids = work.get("ids") or {}
    doi_raw = work.get("doi") or ids.get("doi") or ""
    doi, doi_url = doi_text_and_url(doi_raw)
    pmid = pmid_text(ids.get("pmid") or work.get("pmid") or "")
    openalex_id = clean_text(work.get("id")) if "openalex.org" in clean_text(work.get("id")) else ""
    title = clean_text(work.get("title") or work.get("raw_reference") or "Untitled reference")
    authors = _authors_for_bibtex(work)
    journal = clean_text(source.get("display_name") or work.get("journal"))
    year = clean_text(work.get("publication_year"))
    volume = clean_text(biblio.get("volume") or work.get("volume"))
    issue = clean_text(biblio.get("issue") or work.get("issue"))
    pages = extract_pages(biblio) or clean_text(work.get("pages"))
    work_type = clean_text(work.get("type")).lower()

    type_map = {
        "book": "book",
        "book-chapter": "incollection",
        "dissertation": "phdthesis",
        "report": "techreport",
        "proceedings-article": "inproceedings",
    }
    bib_type = type_map.get(work_type, "article")
    key = _reference_bib_key(work, index)

    lines = [f"@{bib_type}{{{key},"]
    if authors:
        lines.append(f"  author = {{{_bibtex_escape(authors)}}},")
    if title:
        lines.append(f"  title = {{{_bibtex_escape(title)}}},")
    if journal and bib_type == "article":
        lines.append(f"  journal = {{{_bibtex_escape(journal)}}},")
    elif journal:
        lines.append(f"  booktitle = {{{_bibtex_escape(journal)}}},")
    if year:
        lines.append(f"  year = {{{_bibtex_escape(year)}}},")
    if volume:
        lines.append(f"  volume = {{{_bibtex_escape(volume)}}},")
    if issue:
        lines.append(f"  number = {{{_bibtex_escape(issue)}}},")
    if pages:
        lines.append(f"  pages = {{{_bibtex_escape(pages.replace('-', '--'))}}},")
    if doi:
        lines.append(f"  doi = {{{_bibtex_escape(doi)}}},")
        lines.append(f"  url = {{{_bibtex_escape(doi_url)}}},")
    elif openalex_id:
        lines.append(f"  url = {{{_bibtex_escape(openalex_id)}}},")
    if pmid:
        lines.append(f"  pmid = {{{_bibtex_escape(pmid)}}},")
    if openalex_id:
        lines.append(f"  openalex = {{{_bibtex_escape(_openalex_short_id(openalex_id))}}},")
    if clean_text(work.get("reference_source")):
        lines.append(f"  x_reference_source = {{{_bibtex_escape(work.get('reference_source'))}}},")

    keywords = work.get("papers_keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = [clean_text(x) for x in keywords if clean_text(x)]
    if keywords:
        # Standard BibTeX keyword field. Papers can use this as the basis for Smart Lists.
        lines.append(f"  keywords = {{{_bibtex_escape(', '.join(sorted(set(keywords))))}}},")

    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def fetch_openalex_works_by_ids(openalex_ids: Iterable[str], batch_size: int = 50) -> Dict[str, dict]:
    """Retrieve full OpenAlex metadata for many W-IDs using OR filters."""
    short_ids = []
    seen = set()
    for value in openalex_ids or []:
        sid = _openalex_short_id(value)
        if sid and sid not in seen:
            seen.add(sid)
            short_ids.append(sid)

    found: Dict[str, dict] = {}
    for start in range(0, len(short_ids), batch_size):
        batch = short_ids[start:start + batch_size]
        data = get_json(
            f"{BASE_URL}/works",
            {
                "filter": "openalex_id:" + "|".join(batch),
                "per-page": min(200, max(len(batch), 1)),
            },
        )
        for work in (data or {}).get("results") or []:
            sid = _openalex_short_id(work.get("id"))
            if sid:
                found[sid] = work
        print(
            f"    reference metadata batch {start // batch_size + 1}: "
            f"{len(batch)} requested, {sum(1 for x in batch if x in found)} available"
        )
    return found


def _reference_identity(work: dict) -> str:
    """Stable key for deduplication across OpenAlex, Europe PMC, and Crossref."""
    ids = work.get("ids") or {}
    doi, _ = doi_text_and_url(work.get("doi") or ids.get("doi") or "")
    if doi:
        return "doi:" + doi.lower()
    pmid = pmid_text(ids.get("pmid") or work.get("pmid") or "")
    if pmid:
        return "pmid:" + pmid
    oid = clean_text(work.get("id"))
    if oid and "openalex.org" in oid:
        return "openalex:" + _openalex_short_id(oid)
    title = clean_text(work.get("title") or work.get("raw_reference")).casefold()
    title = re.sub(r"[^a-z0-9]+", " ", title).strip()
    year = clean_text(work.get("publication_year"))
    if title:
        return f"title:{title}|{year}"
    return ""


def _author_string_to_bibtex(author_string: Any) -> str:
    """Best-effort conversion of Europe PMC display authors to BibTeX author syntax."""
    s = clean_text(author_string)
    if not s:
        return ""
    # Europe PMC commonly returns 'Smith J, Jones AB, Doe C'.
    if "," in s and " and " not in s.lower():
        bits = [x.strip() for x in s.split(",") if x.strip()]
        if len(bits) > 1:
            return " and ".join(bits)
    return s


def _crossref_message_for_doi(doi_value: Any) -> dict:
    """Fetch a Crossref work message, sharing the script-wide DOI cache."""
    doi, _ = doi_text_and_url(doi_value)
    if not doi:
        return {}
    if doi in _CROSSREF_CACHE:
        return _CROSSREF_CACHE[doi]
    try:
        url = f"https://api.crossref.org/works/{requests.utils.quote(doi, safe='')}"
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "OpenAlexCitationReport/6.2 (reference-list fallback)"},
        )
        if response.status_code != 200:
            return {}
        message = (response.json() or {}).get("message") or {}
        _CROSSREF_CACHE[doi] = message
        time.sleep(0.05)
        return message
    except requests.RequestException:
        return {}


def _crossref_message_to_work(message: dict, doi_hint: str = "") -> dict:
    """Convert a Crossref work message to the OpenAlex-like structure used by BibTeX export."""
    titles = message.get("title") or []
    containers = message.get("container-title") or []
    authorships = []
    for a in message.get("author") or []:
        family = clean_text(a.get("family"))
        given = clean_text(a.get("given"))
        display = " ".join(x for x in (given, family) if x).strip()
        if display:
            authorships.append({"author": {"display_name": display}})

    year = ""
    for fld in ("published-print", "published-online", "issued", "created"):
        parts = ((message.get(fld) or {}).get("date-parts") or [])
        if parts and parts[0]:
            year = str(parts[0][0])
            break

    doi = clean_text(message.get("DOI") or doi_hint)
    page = clean_text(message.get("page") or message.get("article-number"))
    first_page, last_page = "", ""
    if page:
        pbits = re.split(r"[-–]", page, maxsplit=1)
        first_page = pbits[0]
        last_page = pbits[1] if len(pbits) > 1 else ""

    return {
        "id": "",
        "title": clean_text(titles[0] if titles else ""),
        "publication_year": year,
        "authorships": authorships,
        "doi": doi,
        "ids": {"doi": doi} if doi else {},
        "primary_location": {"source": {"display_name": clean_text(containers[0] if containers else "")}},
        "biblio": {
            "volume": clean_text(message.get("volume")),
            "issue": clean_text(message.get("issue")),
            "first_page": first_page,
            "last_page": last_page,
        },
        "type": clean_text(message.get("type")),
        "reference_source": "Crossref",
    }


def _europe_pmc_source_pmid(source_work: dict) -> str:
    """Get the source PMID from OpenAlex IDs, or resolve the source DOI through Europe PMC."""
    ids = source_work.get("ids") or {}
    pmid = pmid_text(ids.get("pmid") or source_work.get("pmid") or "")
    if pmid:
        return pmid

    doi, _ = doi_text_and_url(source_work.get("doi") or ids.get("doi") or "")
    if not doi:
        return ""
    try:
        response = requests.get(
            f"{EUROPE_PMC_BASE}/search",
            params={"query": f'DOI:"{doi}"', "format": "json", "pageSize": 5},
            timeout=30,
            headers={"User-Agent": "OpenAlexCitationReport/6.2 (Europe PMC reference fallback)"},
        )
        if response.status_code != 200:
            return ""
        for item in ((response.json() or {}).get("resultList") or {}).get("result") or []:
            candidate = pmid_text(item.get("pmid") or item.get("id") or "")
            if candidate:
                return candidate
    except requests.RequestException:
        return ""
    return ""


def fetch_europe_pmc_references(source_work: dict) -> List[dict]:
    """Retrieve a source publication's bibliography from Europe PMC when available."""
    pmid = _europe_pmc_source_pmid(source_work)
    if not pmid:
        return []
    try:
        response = requests.get(
            f"{EUROPE_PMC_BASE}/MED/{pmid}/references",
            params={"format": "json", "pageSize": 1000},
            timeout=45,
            headers={"User-Agent": "OpenAlexCitationReport/6.2 (Europe PMC reference fallback)"},
        )
        if response.status_code != 200:
            return []
        items = (((response.json() or {}).get("referenceList") or {}).get("reference") or [])
    except requests.RequestException:
        return []

    refs: List[dict] = []
    for item in items:
        doi = clean_text(item.get("doi"))
        ref_pmid = ""
        if clean_text(item.get("source")).upper() == "MED":
            ref_pmid = pmid_text(item.get("id"))
        page_info = clean_text(item.get("pageInfo"))
        first_page, last_page = "", ""
        if page_info:
            pbits = re.split(r"[-–]", page_info, maxsplit=1)
            first_page = pbits[0]
            last_page = pbits[1] if len(pbits) > 1 else ""
        refs.append({
            "id": "",
            "title": clean_text(item.get("title") or item.get("text")),
            "publication_year": clean_text(item.get("pubYear")),
            "authorships": [],
            "bibtex_authors": _author_string_to_bibtex(item.get("authorString")),
            "doi": doi,
            "pmid": ref_pmid,
            "ids": {k: v for k, v in {"doi": doi, "pmid": ref_pmid}.items() if v},
            "primary_location": {
                "source": {
                    "display_name": clean_text(item.get("journalTitle") or item.get("journalAbbreviation"))
                }
            },
            "biblio": {
                "volume": clean_text(item.get("volume")),
                "issue": clean_text(item.get("issue")),
                "first_page": first_page,
                "last_page": last_page,
            },
            "type": "article",
            "raw_reference": clean_text(item.get("text")),
            "reference_source": "Europe PMC",
        })
    return refs


def fetch_crossref_references(source_work: dict) -> List[dict]:
    """Retrieve references deposited by the publisher with Crossref."""
    ids = source_work.get("ids") or {}
    doi, _ = doi_text_and_url(source_work.get("doi") or ids.get("doi") or "")
    if not doi:
        return []
    message = _crossref_message_for_doi(doi)
    if not message:
        return []

    refs: List[dict] = []
    for item in message.get("reference") or []:
        ref_doi = clean_text(item.get("DOI"))
        if ref_doi:
            resolved = _crossref_message_for_doi(ref_doi)
            if resolved:
                refs.append(_crossref_message_to_work(resolved, ref_doi))
                continue

        title = clean_text(item.get("article-title") or item.get("volume-title") or item.get("unstructured"))
        author = clean_text(item.get("author"))
        journal = clean_text(item.get("journal-title"))
        year = clean_text(item.get("year"))
        first_page = clean_text(item.get("first-page"))
        refs.append({
            "id": "",
            "title": title,
            "publication_year": year,
            "authorships": [],
            "bibtex_authors": author,
            "doi": ref_doi,
            "ids": {"doi": ref_doi} if ref_doi else {},
            "primary_location": {"source": {"display_name": journal}},
            "biblio": {
                "volume": clean_text(item.get("volume")),
                "issue": "",
                "first_page": first_page,
                "last_page": "",
            },
            "type": "article",
            "raw_reference": clean_text(item.get("unstructured")),
            "reference_source": "Crossref",
        })
    return refs


def _deduplicate_reference_works(refs: Iterable[dict]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for ref in refs or []:
        key = _reference_identity(ref)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def export_reference_bibliographies(
    author_name: str,
    works: List[dict],
    outdir: Path = OUTDIR,
) -> dict:
    """
    Export one BibTeX bibliography per source publication plus a deduplicated
    master bibliography containing all references across the author's works.

    Retrieval order for each publication:
        1. OpenAlex `referenced_works`
        2. Europe PMC reference list if OpenAlex yields no usable references
        3. Crossref deposited references if Europe PMC also yields none

    Per-publication filename format:
        YEAR_FirstAuthor_AbbreviatedJournal.bib
    """
    ref_root = Path(outdir) / "references"
    per_pub_dir = ref_root / "by_publication"
    per_pub_dir.mkdir(parents=True, exist_ok=True)

    print("\n📖 Exporting reference lists from the author's publications ...")

    all_ref_ids: List[str] = []
    for work in works:
        all_ref_ids.extend(work.get("referenced_works") or [])
    ref_meta = fetch_openalex_works_by_ids(all_ref_ids)

    manifest_rows: List[dict] = []
    master_refs: List[dict] = []
    master_seen = set()

    for pos, source_work in enumerate(works, 1):
        ref_ids = [
            _openalex_short_id(x)
            for x in (source_work.get("referenced_works") or [])
            if _openalex_short_id(x)
        ]
        openalex_refs = [ref_meta[rid] for rid in ref_ids if rid in ref_meta]
        unique_refs = _deduplicate_reference_works(openalex_refs)
        reference_source = "OpenAlex" if unique_refs else ""

        europe_pmc_count = 0
        crossref_count = 0

        if not unique_refs:
            print(
                f"  ↪ [{pos}/{len(works)}] OpenAlex has no usable references for "
                f"'{clean_text(source_work.get('title'))[:75]}'; trying Europe PMC ..."
            )
            epmc_refs = _deduplicate_reference_works(fetch_europe_pmc_references(source_work))
            europe_pmc_count = len(epmc_refs)
            if epmc_refs:
                unique_refs = epmc_refs
                reference_source = "Europe PMC"
            else:
                print("      Europe PMC yielded none; trying Crossref ...")
                cr_refs = _deduplicate_reference_works(fetch_crossref_references(source_work))
                crossref_count = len(cr_refs)
                if cr_refs:
                    unique_refs = cr_refs
                    reference_source = "Crossref"

        source_tag = _source_publication_tag(source_work)

        # Add the source-publication tag to every reference in this publication.
        # This allows Papers Smart Lists to group imported references automatically.
        for ref in unique_refs:
            existing = ref.get("papers_keywords") or []
            if isinstance(existing, str):
                existing = [existing]
            ref["papers_keywords"] = sorted(set(existing + [source_tag]))

        # Build the deduplicated master list while preserving *all* source tags
        # when the same cited paper occurs in more than one publication.
        master_index = {
            _reference_identity(existing_ref): existing_ref
            for existing_ref in master_refs
            if _reference_identity(existing_ref)
        }
        for ref in unique_refs:
            key = _reference_identity(ref)
            if not key:
                continue
            if key not in master_seen:
                master_seen.add(key)
                master_refs.append(ref)
                master_index[key] = ref
            else:
                target = master_index.get(key)
                if target is not None:
                    old_kw = target.get("papers_keywords") or []
                    if isinstance(old_kw, str):
                        old_kw = [old_kw]
                    new_kw = ref.get("papers_keywords") or []
                    if isinstance(new_kw, str):
                        new_kw = [new_kw]
                    target["papers_keywords"] = sorted(set(old_kw + new_kw))

        bib_name = _reference_bib_filename(source_work)
        bib_path = per_pub_dir / bib_name
        header = (
            f"% References cited by: {clean_text(source_work.get('title'))}\n"
            f"% Source OpenAlex ID: {clean_text(source_work.get('id'))}\n"
            f"% Reference-list source: {reference_source or 'none'}\n"
            f"% Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n\n"
        )
        entries = [
            _bibtex_entry_from_openalex_work(ref, i + 1)
            for i, ref in enumerate(unique_refs)
        ]

        if entries:
            bib_path.write_text(header + "\n\n".join(entries) + "\n", encoding="utf-8")
            bib_file_value = str(bib_path)
            export_status = "OK"
        else:
            if bib_path.exists():
                bib_path.unlink()
            bib_file_value = ""
            export_status = "NO_REFERENCE_LIST_FOUND"

        missing_openalex = max(len(ref_ids) - len(openalex_refs), 0)
        manifest_rows.append(
            {
                "source_year": clean_text(source_work.get("publication_year")),
                "source_first_author": _first_author_surname_from_work(source_work),
                "source_title": clean_text(source_work.get("title")),
                "papers_list_tag": source_tag,
                "source_openalex_id": clean_text(source_work.get("id")),
                "source_doi": doi_text_and_url(
                    source_work.get("doi") or (source_work.get("ids") or {}).get("doi") or ""
                )[0],
                "references_listed_by_openalex": len(ref_ids),
                "openalex_references_with_metadata": len(openalex_refs),
                "europe_pmc_references": europe_pmc_count,
                "crossref_references": crossref_count,
                "references_exported": len(unique_refs),
                "references_missing_openalex_metadata": missing_openalex,
                "reference_list_source": reference_source,
                "status": export_status,
                "bib_file": bib_file_value,
            }
        )

    master_path = ref_root / f"ALL_{safe_filename(author_name)}_references_deduplicated.bib"
    master_header = (
        f"% Deduplicated references cited across publications by {author_name}\n"
        f"% Sources: OpenAlex, Europe PMC fallback, Crossref fallback\n"
        f"% Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n\n"
    )
    master_entries = [
        _bibtex_entry_from_openalex_work(ref, i + 1)
        for i, ref in enumerate(master_refs)
    ]
    if master_entries:
        master_path.write_text(master_header + "\n\n".join(master_entries) + "\n", encoding="utf-8")
    elif master_path.exists():
        master_path.unlink()

    manifest_path = ref_root / "reference_export_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    summary = {
        "source_publications": len(works),
        "reference_links_reported_by_openalex": sum(
            int(row["references_listed_by_openalex"]) for row in manifest_rows
        ),
        "publications_using_openalex": sum(1 for row in manifest_rows if row.get("reference_list_source") == "OpenAlex"),
        "publications_using_europe_pmc_fallback": sum(1 for row in manifest_rows if row.get("reference_list_source") == "Europe PMC"),
        "publications_using_crossref_fallback": sum(1 for row in manifest_rows if row.get("reference_list_source") == "Crossref"),
        "publications_without_retrievable_reference_list": sum(1 for row in manifest_rows if row.get("status") != "OK"),
        "unique_references_exported": len(master_refs),
        "master_bib": str(master_path) if master_entries else "",
        "manifest_csv": str(manifest_path),
        "per_publication_directory": str(per_pub_dir),
    }
    summary_path = ref_root / "reference_export_summary.txt"
    summary_path.write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n",
        encoding="utf-8",
    )

    print(f"  ✅ Per-publication BibTeX files: {per_pub_dir}")
    if master_entries:
        print(f"  ✅ Deduplicated master BibTeX: {master_path}")
    print(f"  ✅ Reference export manifest: {manifest_path}")
    print(f"  ✅ Unique references exported: {len(master_refs)}")
    print(
        "  Sources used: "
        f"OpenAlex={summary['publications_using_openalex']}, "
        f"Europe PMC fallback={summary['publications_using_europe_pmc_fallback']}, "
        f"Crossref fallback={summary['publications_using_crossref_fallback']}, "
        f"unavailable={summary['publications_without_retrievable_reference_list']}"
    )
    return summary


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
    parser.add_argument(
        "--export_references",
        action="store_true",
        help=(
            "Export the reference list of every author publication as BibTeX. "
            "Also creates a deduplicated master .bib in OUTDIR/references/."
        ),
    )
    parser.add_argument(
        "--references_only",
        action="store_true",
        help=(
            "Export publication reference lists and stop without running the "
            "citation analysis/PDF report. Implies --export_references."
        ),
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

    if args.export_references or args.references_only:
        export_reference_bibliographies(author_name, works, outdir=OUTDIR)
        if args.references_only:
            print("\n✅ Reference export done.")
            return

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
