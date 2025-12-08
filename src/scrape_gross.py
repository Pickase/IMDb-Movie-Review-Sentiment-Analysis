import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import urllib.parse

# ================================
# CLEAN MOVIE TITLE FOR URL
# ================================
def normalize_title(title):
    t = title.strip()
    t = t.replace(" ", "_")
    t = urllib.parse.quote(t)
    return t

# ================================
# SCRAPE GROSS FROM WIKIPEDIA
# ================================
def scrape_gross(title):
    """
    Returns total worldwide gross for a movie title from Wikipedia.
    Tries multiple URL patterns.
    """
    title_clean = normalize_title(title)

    candidate_urls = [
        f"https://en.wikipedia.org/wiki/{title_clean}",
        f"https://en.wikipedia.org/wiki/{title_clean}_(film)",
        f"https://en.wikipedia.org/wiki/{title_clean}_film",
    ]

    for url in candidate_urls:
        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # find infobox finance row
            info = soup.find("table", {"class": "infobox vevent"})
            if not info:
                continue

            rows = info.find_all("tr")

            for row in rows:
                header = row.find("th")
                if header and "box office" in header.text.lower():
                    value = row.find("td").text.strip()

                    # Remove citations like [1], [2]
                    value = re.sub(r"\[.*?\]", "", value)
                    return value

                if header and "gross" in header.text.lower():
                    value = row.find("td").text.strip()
                    value = re.sub(r"\[.*?\]", "", value)
                    return value

        except Exception:
            continue

    return "N/A"


# ================================
# MAIN FUNCTION
# ================================
def generate_movie_gross_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required_cols = ["id", "title", "year", "rating"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in your dataset")

    df["grossing"] = None

    print("Scraping gross data…\n")

    for idx, row in df.iterrows():
        title = row["title"]

        print(f"Scraping: {title} ... ", end="")
        gross = scrape_gross(title)
        df.at[idx, "grossing"] = gross
        print(f"{gross}")

        time.sleep(1.2)  # avoid Wikipedia rate-limiting

    df.to_csv(output_csv, index=False)
    print(f"\nDONE! Saved enriched file as: {output_csv}")


if __name__ == "__main__":
    # Update file paths as needed
    input_path = "data/bollywood_movies.csv"     # your original dataset
    output_path = "data/bollywood_movies_with_gross.csv"

    generate_movie_gross_csv(input_path, output_path)
