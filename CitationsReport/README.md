# CitationsReport

`collect_citations_openalex_ORCIDname.py` queries OpenAlex for an author's works and citing works, then produces citation tables and visual reports. Earlier numbered development copies are retained in Git history rather than the working tree.

The scripts accept author information interactively or through constants/options defined in the selected version. Review its `--help` and configuration section before running:

```bash
python collect_citations_openalex_ORCIDname.py --help
```

Typical dependencies include `requests`, `pandas`, `matplotlib`, and PDF/report libraries imported by the selected version. `scimagojr 2025.csv` supplies journal-ranking metadata. Generated files go under `openalex_citations_output/` and are intentionally excluded from Git.

OpenAlex is a network service; respect its current API guidance and identify requests with an email address when the script offers that option.
