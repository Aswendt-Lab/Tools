# Assign Speaker for Lab Meetings

`PresentationLottery.py` assigns one person and either “Project Presentation” or “Journal Club” to every date. It avoids assigning the same person in two consecutive entries.

Run the script from this folder after editing `people.txt` and `dates.txt` (one value per line):

```bash
python PresentationLottery.py
```

It writes `assignments.txt` and appends assignments to `history.json`. Re-running with the same dates creates additional history entries; archive or clear the history deliberately when starting a new schedule.
