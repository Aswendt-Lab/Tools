""""
Created on 17.12.2025

@authors: Markus Aswendt, ChatGPT
Department of Neurology
University Hospital Frankfurt
Theodor-Stern-Kai 7
D-60590 Frankfurt am Main

"""

import random
import json
import os

# -----------------------------
# File loading helpers
# -----------------------------
def load_people(file_path):
    """Load names from a text file (one name per line)."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_dates(file_path):
    """Load meeting dates from text file."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_history(history_file):
    """Load assignment history from JSON file."""
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return []

def save_history(history, history_file):
    """Save assignment history to JSON file."""
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

# -----------------------------
# Assignment logic
# -----------------------------
def assign_for_date(date, people, history):
    """
    Assign one person for a given date.
    Rules:
      - no person twice in consecutive weeks
      - only one task (Journal Club OR Project Presentation)
    """
    last_person = history[-1]["person"] if history else None
    eligible = [p for p in people if p != last_person]

    if not eligible:
        raise ValueError("No eligible people — everyone blocked by rules!")

    person = random.choice(eligible)
    task = random.choice(["Project Presentation", "Journal Club"])

    assignment = {
        "date": date,
        "person": person,
        "task": task
    }

    history.append(assignment)
    return assignment

# -----------------------------
# MAIN PROGRAM
# -----------------------------
if __name__ == "__main__":
    PEOPLE_FILE = "people.txt"
    DATES_FILE = "dates.txt"
    HISTORY_FILE = "history.json"
    OUTPUT_FILE = "assignments.txt"

    people = load_people(PEOPLE_FILE)
    dates = load_dates(DATES_FILE)
    history = load_history(HISTORY_FILE)

    assignments = []

    for date in dates:
        assignment = assign_for_date(date, people, history)
        assignments.append(assignment)
        print(f"{date}: {assignment['person']} → {assignment['task']}")

    # Save updated history
    save_history(history, HISTORY_FILE)

    # Save results to output.txt
    with open(OUTPUT_FILE, "w") as f:
        for a in assignments:
            f.write(f"{a['date']}: {a['person']} → {a['task']}\n")

    print("\nAssignments saved to assignments.txt")
