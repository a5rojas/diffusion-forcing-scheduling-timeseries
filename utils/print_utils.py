from colorama import Fore
import os
import csv

def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"


def append_row_to_csv(path, row_dict):
    """Append a row to a CSV file, creating the header on first write."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)