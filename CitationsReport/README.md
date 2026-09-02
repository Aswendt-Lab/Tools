# CitationsReport

These scripts query OpenAlex for an author's works and citing works, then produce citation tables and visual reports. The numbered files are development versions; `collect_citations_openalex_ORCIDname_v7.py` is the newest and should normally be used.

The scripts accept author information interactively or through constants/options defined in the selected version. Review its `--help` and configuration section before running:

```bash
python collect_citations_openalex_ORCIDname_v7.py --help
```

Typical dependencies include `requests`, `pandas`, `matplotlib`, and PDF/report libraries imported by the selected version. `scimagojr 2025.csv` supplies journal-ranking metadata. Generated files go under `openalex_citations_output/` and are intentionally excluded from Git.

OpenAlex is a network service; respect its current API guidance and identify requests with an email address when the script offers that option.
