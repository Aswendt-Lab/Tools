#!/usr/bin/env python3
"""
Collect citations from OpenAlex for an author (by name or ORCID) and
generate:
- CSV with the author’s works
- per-work CSVs with citing works
- per-work plots (citations per year + by country)
- one PDF report with all plots and clickable DOI links
"""

import os
import sys
import time
import math
import logging
import warnings
import html
import re
import datetime

from urllib.parse import urlencode

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === BASIC CONFIG ===

BASE_URL = "https://api.openalex.org"
OUTDIR = "openalex_citations_output"
os.makedirs(OUTDIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# === OPTIONAL COLOR PALETTE ===
try:
    import seaborn as sns
    COLOR_PALETTE = sns.color_palette("hls", 8).as_hex()
except Exception as e:
    print(f"⚠️ Seaborn not available ({e}), using basic palette.")
    COLOR_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]


# -------------------------------------------------------------------
# SMALL UTILITIES
# -------------------------------------------------------------------

def safe_para(text: str) -> str:
    """
    Escape characters that could be mistaken for HTML tags in ReportLab Paragraphs.
    We *always* run external text through this before putting into <Paragraph>.
    """
    if not isinstance(text, str):
        return ""
    return html.escape(text, quote=False)


def abbreviated_journal(journal: str) -> str:
    """
    Heuristic journal abbreviation for filenames/labels.
    """
    if not journal:
        return "unknown_journal"

    j = journal.strip()

    # If it's already short, just replace spaces
    if len(j) <= 25:
        return j.replace(" ", "_")

    parts = j.split()
    caps = [p for p in parts if p and p[0].isupper()]

    if len(caps) > 1:
        abbr = "".join(w[0] for w in caps if w[0].isalpha())
        if 3 <= len(abbr) <= 10:
            return abbr

    return j[:25].strip().replace(" ", "_")


def safe_filename(name: str) -> str:
    """
    Make a filesystem-safe filename stub.
    """
    if not isinstance(name, str):
        name = str(name)

    bad_chars = r'\/:*?"<>|'
    for ch in bad_chars:
        name = name.replace(ch, "_")

    name = name.strip()
    if not name:
        name = "unnamed"
    return name


def get_json(url, params=None, max_retries=5, pause=1.0):
    """
    Wrapper around requests.get with retries and basic rate-limit handling.
    Returns JSON or None.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait_time = pause * (attempt + 1)
                print(f"⏱️ Rate-limited by OpenAlex. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ HTTP {resp.status_code} for {resp.url}")
                break
        except requests.RequestException as e:
            print(f"⚠️ Error fetching {url}: {e}. Retry {attempt+1}/{max_retries}")
            time.sleep(pause * (attempt + 1))

    print(f"❌ Failed to get data from {url}")
    return None


# -------------------------------------------------------------------
# OPENALEX AUTHOR LOOKUP
# -------------------------------------------------------------------

def find_author_by_orcid(orcid: str):
    if not orcid:
        return None
    url = f"{BASE_URL}/authors"
    params = {"filter": f"orcid:{orcid}"}
    data = get_json(url, params=params)
    if not data or "results" not in data or not data["results"]:
        return None
    return data["results"][0]


def find_authors_by_name(name: str, max_results=5):
    url = f"{BASE_URL}/authors"
    params = {
        "search": name,
        "per-page": max_results,
        "sort": "cited_by_count:desc"
    }
    data = get_json(url, params=params)
    if not data or "results" not in data:
        return []
    return data["results"]


def pick_best_author(name: str, orcid: str = None):
    """
    Hybrid selection:
    - If ORCID is given, try that first.
    - Otherwise: search by name, auto-pick the most cited candidate.
    """
    if orcid:
        print(f"🔍 Searching OpenAlex by ORCID {orcid} ...")
        author = find_author_by_orcid(orcid)
        if author:
            print(f"✅ Found author (ORCID): {author.get('display_name')}")
            return author
        print("⚠️ No author found for given ORCID, falling back to name search...")

    print(f"🔍 Searching OpenAlex by name '{name}' ...")
    candidates = find_authors_by_name(name)
    if not candidates:
        print("❌ No authors found by name.")
        return None

    print("\nPossible matches (top results):")
    for i, a in enumerate(candidates):
        display_name = a.get("display_name", "Unknown")
        orcid_val = a.get("orcid", "n/a")
        works_count = a.get("works_count", 0)
        cited_by = a.get("cited_by_count", 0)
        print(f"  [{i}] {display_name} (ORCID: {orcid_val}, works={works_count}, cited_by={cited_by})")

    best = candidates[0]
    print(f"\n➡️ Automatically selecting: {best.get('display_name')} (highest cited_by_count among results)")
    return best


# -------------------------------------------------------------------
# WORKS + CITING WORKS
# -------------------------------------------------------------------

def fetch_works_for_author(author_id: str, per_page=200):
    works = []
    page = 1
    print("\n📚 Fetching works for author...")
    while True:
        params = {
            "filter": f"authorships.author.id:{author_id}",
            "per-page": per_page,
            "page": page,
            "sort": "publication_year:asc"
        }
        url = f"{BASE_URL}/works"
        data = get_json(url, params=params)
        if not data or "results" not in data:
            break
        these = data["results"]
        if not these:
            break
        works.extend(these)
        print(f"  Page {page}: {len(these)} works (total so far: {len(works)})")
        if len(these) < per_page:
            break
        page += 1
    return works


def get_primary_venue_short(work: dict) -> str:
    host_venue = work.get("primary_location", {}).get("source", {}) or {}
    journal = host_venue.get("display_name") or work.get("host_venue", {}).get("display_name") or ""
    return abbreviated_journal(journal)


def extract_work_metadata(w: dict):
    title = w.get("title", "Untitled")
    year = w.get("publication_year")
    cited_by_count = w.get("cited_by_count", 0)
    doi = w.get("doi", "")
    doi_link = ""
    if doi:
        doi_link = f"https://doi.org/{doi.split('doi.org/')[-1]}"

    journal_short = get_primary_venue_short(w)
    openalex_id = w.get("id", "")

    return {
        "title": title,
        "year": year,
        "cited_by_count": cited_by_count,
        "doi": doi,
        "doi_link": doi_link,
        "journal_short": journal_short,
        "openalex_id": openalex_id,
    }


def fetch_citing_works(openalex_id: str, per_page=200):
    citing = []
    page = 1
    short_id = openalex_id.split("/")[-1]
    print(f"\n🔎 Analyzing citing works for: {short_id}")
    while True:
        url = f"{BASE_URL}/works"
        params = {
            "filter": f"cites:{openalex_id}",
            "per-page": per_page,
            "page": page,
            "sort": "publication_year:asc"
        }
        data = get_json(url, params=params)
        if not data or "results" not in data:
            break
        these = data["results"]
        if not these:
            break
        citing.extend(these)
        print(f"    citing page {page}: {len(these)} works (total so far: {len(citing)})")
        if len(these) < per_page:
            break
        page += 1
    return citing


# -------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------

def analyze_citations_over_time(citing_works):
    if not citing_works:
        return pd.DataFrame(columns=["year", "count"])

    rows = []
    for cw in citing_works:
        y = cw.get("publication_year")
        if y is None:
            continue
        rows.append({"year": int(y), "count": 1})

    if not rows:
        return pd.DataFrame(columns=["year", "count"])

    df = pd.DataFrame(rows)
    out = df.groupby("year")["count"].sum().reset_index()
    return out.sort_values("year")


def analyze_citations_by_country(citing_works):
    rows = []
    for cw in citing_works:
        auths = cw.get("authorships", [])
        if not auths:
            continue
        inst_country = None
        for a in auths:
            insts = a.get("institutions", [])
            if not insts:
                continue
            inst = insts[0]
            inst_country = inst.get("country_code") or inst.get("country")
            if inst_country:
                break
        if not inst_country:
            inst_country = "Unknown"
        rows.append({"country": inst_country, "count": 1})

    if not rows:
        return pd.DataFrame(columns=["country", "count"])

    df = pd.DataFrame(rows)
    out = df.groupby("country")["count"].sum().reset_index()
    return out.sort_values("count", ascending=False)


# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------

from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------

def plot_citations_per_year(df_year: pd.DataFrame, title_stub: str, outdir: str = OUTDIR):
    """
    Plot citations per year with a dark bar style and a dynamic polynomial trend line.
    - Degree 1 (linear) if 3–4 data points
    - Degree 2 if ≥5 data points
    - No trend line if <3 data points
    Y-axis always starts at 0.
    """
    if df_year is None or df_year.empty:
        return None

    df_year = df_year.sort_values("year")
    years = df_year["year"].values.astype(float)
    counts = df_year["count"].values.astype(float)

    if len(years) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)

    # Bars
    ax.bar(years, counts, width=0.7, color="#444444", alpha=0.8)

    # Dynamic polynomial trend
    n = len(years)
    if n >= 3:
        if n >= 5:
            deg = 2
        else:
            deg = 1
        try:
            p = np.polyfit(years, counts, deg)
            xfit = np.linspace(years.min(), years.max(), 200)
            yfit = np.polyval(p, xfit)
            # Clip any negative fit values
            yfit = np.clip(yfit, 0, None)
            ax.plot(xfit, yfit, linewidth=2,
                    label=f"Trend (poly deg {deg})",
                    color=COLOR_PALETTE[1])
        except Exception as e:
            print(f"⚠️ Could not fit polynomial trend: {e}")

    ax.set_xlabel("Year")
    ax.set_ylabel("Citations")
    ax.set_title(f"Citations per Year — {title_stub}")

    # X axis: tight around available years, integer ticks
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Y axis: start from 0, go a bit above max
    ymax = max(counts.max(), 1)
    ax.set_ylim(bottom=0, top=ymax * 1.1)

    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    fname = os.path.join(outdir, f"{safe_filename(title_stub)}_citations_per_year.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_citations_by_country(df_country: pd.DataFrame, title_stub: str, outdir: str = OUTDIR):
    """
    Plot a pie chart of citations by country.
    Groups smaller countries into 'Other' if there are more than 10 entries.
    """
    if df_country is None or df_country.empty:
        return None

    # Work on a sorted copy
    df = df_country.copy().sort_values("count", ascending=False)

    # Group tail into "Other" if many countries
    if len(df) > 10:
        top = df.iloc[:9].copy()
        others_sum = df.iloc[9:]["count"].sum()
        other_row = pd.DataFrame([{"country": "Other", "count": others_sum}])
        df = pd.concat([top, other_row], ignore_index=True)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.pie(
        df["count"],
        labels=df["country"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 7},
    )
    ax.set_title(f"Citations by Country — {title_stub}", fontsize=10)

    fname = os.path.join(
        outdir,
        f"{safe_filename(title_stub)}_citations_by_country_pie.png"
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname

# -------------------------------------------------------------------
# PDF REPORT
# -------------------------------------------------------------------

from reportlab.platypus import Paragraph, Spacer, Image, SimpleDocTemplate, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

def create_pdf_report(author_name, orcid, summary_df, agg_year_plot, agg_country_plot, total_citations):
    """
    Create a styled PDF report with:
    - Total citation count on first page
    - Each DOI on its own page
    - Full citation in the heading with clickable DOI link
    """
    clean_orcid = re.sub(r'[^0-9A-Za-z]', '', orcid) if orcid else "noORCID"
    clean_name = safe_filename(author_name.replace(" ", "_"))
    pdf_name = f"Report_Citations_{clean_orcid}_{clean_name}.pdf"
    pdf_path = os.path.join(OUTDIR, pdf_name)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    link_style = ParagraphStyle('link', textColor=colors.HexColor("#1a73e8"), underline=True)

    # ---------- FIRST PAGE ----------
    elements = [
        Paragraph("<b>Author Citation Report</b>", styles['Title']),
        Spacer(1, 10),
        Paragraph(f"<b>Author:</b> {safe_para(author_name)}", styles['Normal']),
        Paragraph(f"<b>ORCID:</b> {safe_para(orcid if orcid else 'n/a')}", styles['Normal']),
        Paragraph(f"<b>Total citations (all works in this report):</b> {int(total_citations)}",
                  styles['Normal']),
        Paragraph(f"<b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  styles['Normal']),
        Spacer(1, 20),
    ]

    # Aggregate plots (still on first page)
    if agg_year_plot and os.path.exists(agg_year_plot):
        elements.append(Paragraph("<b>Total Citations per Year</b>", styles['Heading2']))
        elements.append(Image(agg_year_plot, width=500, height=220))
        elements.append(Spacer(1, 12))

    if agg_country_plot and os.path.exists(agg_country_plot):
        elements.append(Paragraph("<b>Total Citations by Country</b>", styles['Heading2']))
        elements.append(Image(agg_country_plot, width=400, height=400))
        elements.append(Spacer(1, 20))

    # ---------- PER-PUBLICATION SECTIONS ----------
    if summary_df is not None and not summary_df.empty:
        for idx, row in summary_df.iterrows():
            # Each DOI starts on a new page (except maybe the first if you want)
            elements.append(PageBreak())

            title = str(row.get("title") or "Untitled")
            journal_short = row.get("journal_short", "")
            year = row.get("year", "")
            doi = row.get("doi", "")
            year_plot = row.get("year_plot")
            country_plot = row.get("country_plot")

            # Clean title: remove HTML tags (e.g. <i>in vivo</i>) and escape
            title_no_tags = re.sub(r"<[^>]+>", "", title)
            title_safe = safe_para(title_no_tags)

            # Build citation string: Title — Journal — Year
            parts = [title_safe]
            if journal_short:
                parts.append(safe_para(str(journal_short)))
            if year is not None and str(year) not in ("", "nan"):
                if isinstance(year, (int, np.integer)):
                    year_str = str(year)
                elif isinstance(year, (float, np.floating)):
                    year_str = str(int(year))
                else:
                    year_str = str(year)
                parts.append(safe_para(year_str))

            citation_text = " — ".join(parts)

            # DOI URL + link text
            doi_url = ""
            if isinstance(doi, str) and doi.strip():
                d = doi.strip()
                if d.lower().startswith("http"):
                    doi_url = d
                else:
                    d = d.replace("doi:", "").strip()
                    doi_url = f"https://doi.org/{d}"

            if doi_url:
                header_html = (
                    f"<b>Publication:</b> {citation_text} "
                    f"(<a href='{safe_para(doi_url)}'>DOI link</a>)"
                )
            else:
                header_html = f"<b>Publication:</b> {citation_text}"

            # Full, un-truncated citation heading
            elements.append(Paragraph(header_html, styles['Heading3']))
            elements.append(Spacer(1, 6))

            # Per-DOI plots
            if isinstance(year_plot, str) and os.path.exists(year_plot):
                elements.append(Image(year_plot, width=450, height=200))
            if isinstance(country_plot, str) and os.path.exists(country_plot):
                elements.append(Image(country_plot, width=350, height=350))
            elements.append(Spacer(1, 10))

            # Citing works (top 15)
            label = row.get("label") or "UnknownLabel"
            base = os.path.join(OUTDIR, safe_filename(label))
            detailed_csv = f"{base}_citations_detailed.csv"

            if os.path.exists(detailed_csv):
                try:
                    df_cit = pd.read_csv(detailed_csv).head(15)
                except Exception:
                    continue
                for _, c in df_cit.iterrows():
                    yr = c.get("year", "")
                    t_raw = str(c.get("title") or "")
                    t_no_tags = re.sub(r"<[^>]+>", "", t_raw)
                    t_safe = safe_para(t_no_tags)
                    link = c.get("doi_link")
                    if isinstance(link, str) and link:
                        elements.append(
                            Paragraph(f"{yr}: <a href='{link}'>{t_safe}</a>", link_style)
                        )
                    else:
                        elements.append(Paragraph(f"{yr}: {t_safe}", styles['Normal']))
                elements.append(Spacer(1, 10))
    else:
        elements.append(Paragraph("No citation data available.", styles['Normal']))

    doc.build(elements)
    print(f"\n📄 PDF report created: {pdf_path}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    import argparse

    print("🚀 OpenAlex citation collector starting...\n")

    parser = argparse.ArgumentParser(
        description="Collect citation data from OpenAlex for a given author (by name or ORCID)."
    )
    parser.add_argument("--name", type=str, help="Author display name")
    parser.add_argument("--orcid", type=str, help="Author ORCID")
    parser.add_argument("--min_citations", type=int, default=5,
                        help="Minimum cited_by_count of a work to analyze in detail")
    args = parser.parse_args()

    name = args.name
    orcid = args.orcid

    # If nothing provided on command line: ask interactively
    if not name and not orcid:
        print("No --name or --orcid given. Let's enter them interactively.")
        name = input("Author name (leave empty if you want to use ORCID only): ").strip()
        if not name:
            orcid = input("ORCID (leave empty to abort): ").strip()
        if not name and not orcid:
            print("❌ No name or ORCID provided. Exiting.")
            sys.exit(1)

    # Pick author
    author = pick_best_author(name or "", orcid)
    if not author:
        print("❌ No suitable author found. Exiting.")
        sys.exit(1)

    author_name = author.get("display_name", "Unknown Author")
    author_orcid = author.get("orcid") or orcid

    print(f"\n=== Selected author: {author_name} (ORCID: {author_orcid or 'n/a'}) ===")

    # Fetch works
    author_id = author.get("id")
    works = fetch_works_for_author(author_id)
    if not works:
        print("❌ No works found for this author.")
        sys.exit(1)

    # Flatten metadata
    meta_rows = [extract_work_metadata(w) for w in works]
    df_works = pd.DataFrame(meta_rows)

    # Save summary of works
    summary_csv_path = os.path.join(OUTDIR, f"works_summary_{safe_filename(author_name)}.csv")
    df_works.to_csv(summary_csv_path, index=False)
    print(f"\n💾 Saved works summary to {summary_csv_path}")

    # Filter works by citation threshold
    df_sel = df_works[df_works["cited_by_count"] >= args.min_citations].copy()
    if df_sel.empty:
        print(f"⚠️ No works meet the minimum citation threshold ({args.min_citations}).")
        sys.exit(0)

    agg_year_all = []
    agg_country_all = []
    detailed_summary_rows = []

    for _, row in df_sel.iterrows():
        title = row["title"]
        openalex_id = row["openalex_id"]
        print(f"\n=== Processing work: {title} ===")

        citing_works = fetch_citing_works(openalex_id)
        if not citing_works:
            print("  No citing works found.")
            continue

        df_year = analyze_citations_over_time(citing_works)
        df_country = analyze_citations_by_country(citing_works)

        if not df_year.empty:
            df_year["title_label"] = title
            agg_year_all.append(df_year)
        if not df_country.empty:
            df_country["title_label"] = title
            agg_country_all.append(df_country)

        # Detailed CSV per work
        detailed_rows = []
        for cw in citing_works:
            cyear = cw.get("publication_year")
            ctitle = cw.get("title", "Untitled")
            cdoi = cw.get("doi", "")
            cdoi_link = ""
            if cdoi:
                cdoi_link = f"https://doi.org/{cdoi.split('doi.org/')[-1]}"

            auths = cw.get("authorships", [])
            inst_country = None
            if auths:
                for a in auths:
                    insts = a.get("institutions", [])
                    if not insts:
                        continue
                    inst = insts[0]
                    inst_country = inst.get("country_code") or inst.get("country")
                    if inst_country:
                        break
            if not inst_country:
                inst_country = "Unknown"

            detailed_rows.append(
                {
                    "year": cyear,
                    "title": ctitle,
                    "doi": cdoi,
                    "doi_link": cdoi_link,
                    "country": inst_country,
                }
            )

        df_det = pd.DataFrame(detailed_rows) if detailed_rows else \
            pd.DataFrame(columns=["year", "title", "doi", "doi_link", "country"])

        label = f"{abbreviated_journal(row['journal_short'])}_{row['year']}"
        base = os.path.join(OUTDIR, safe_filename(label))
        det_csv_path = f"{base}_citations_detailed.csv"
        df_det.to_csv(det_csv_path, index=False)
        print(f"  💾 Saved detailed citing-works CSV: {det_csv_path}")

        # Plots per work
        yp = plot_citations_per_year(df_year, label)
        cp = plot_citations_by_country(df_country, label)

        detailed_summary_rows.append(
            {
                "label": label,
                "title": title,
                "cited_by_count": row["cited_by_count"],
                "doi": row.get("doi", ""),
                "year": row.get("year", ""),
                "journal_short": row.get("journal_short", ""),
                "year_plot": yp,
                "country_plot": cp,
            }
        )


    if not detailed_summary_rows:
        print("⚠️ No detailed citation data to report (no citing works). Exiting.")
        sys.exit(0)

        df_summary = pd.DataFrame(detailed_summary_rows)

    # === Aggregate across all selected works ===
    agg_label = f"{safe_filename(author_name)}_AGGREGATED"

    if agg_year_all:
        df_agg_year = pd.concat(agg_year_all, ignore_index=True)
        df_agg_year_all = df_agg_year.groupby("year")["count"].sum().reset_index()
        total_citations = int(df_agg_year_all["count"].sum())
        agg_year_plot = plot_citations_per_year(df_agg_year_all, agg_label)
    else:
        agg_year_plot = None
        total_citations = 0

    if agg_country_all:
        df_agg_country = pd.concat(agg_country_all, ignore_index=True)
        df_agg_country_all = df_agg_country.groupby("country")["count"].sum().reset_index()
        agg_country_plot = plot_citations_by_country(df_agg_country_all, agg_label)
    else:
        agg_country_plot = None

    # === Create final PDF report ===
    create_pdf_report(
        author_name,
        author_orcid,
        df_summary,
        agg_year_plot,
        agg_country_plot,
        total_citations,
    )

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
