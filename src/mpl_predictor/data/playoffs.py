import re
from datetime import date
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from mpl_predictor.data.season18 import SCHEDULE_URL, _parse_indonesian_date


def parse_playoff_schedule_html(
    html: str, observed_at: date, config: dict[str, Any]
) -> pd.DataFrame:
    """Read official playoff dates/results; bracket routing and BO remain configured."""
    root = BeautifulSoup(html, "html.parser").select_one("#playoffs")
    if root is None:
        raise ValueError("Official schedule has no playoff section.")
    matches = config["playoffs"]["bracket"]
    records = []
    scheduled_date = None
    for element in root.select(".match"):
        if "date" in element.get("class", []):
            scheduled_date = _parse_indonesian_date(element.get_text(" ", strip=True))
            continue
        if "position-relative" not in element.get("class", []):
            continue
        if scheduled_date is None or len(records) >= len(matches):
            raise ValueError("Official playoff schedule does not match the configured bracket.")
        match = matches[len(records)]
        modal = element.select_one("[id^='modal2-']")
        time = re.search(r"\b(\d{2}:\d{2})\b", element.get_text(" ", strip=True))
        if modal is None or time is None:
            raise ValueError("Official playoff match is missing its identifier or time.")
        teams = [element.select_one(f".team{side} .name") for side in (1, 2)]
        if any(team is None for team in teams):
            raise ValueError("Official playoff match is missing team fields.")
        names = [team.get_text(" ", strip=True).upper() for team in teams]
        names = [name if name and name != "TBD" else None for name in names]
        scores = [item.get_text(" ", strip=True) for item in element.select(".score")]
        if len(scores) != 2 or any(value.isdigit() for value in scores) != all(
            value.isdigit() for value in scores
        ):
            raise ValueError("Official playoff match has incomplete score fields.")
        completed = all(value.isdigit() for value in scores)
        if completed and (scheduled_date > observed_at or None in names):
            raise ValueError("Official playoff result has a future date or unresolved teams.")
        score_a, score_b = (int(value) for value in scores) if completed else (None, None)
        records.append(
            {
                "season": int(config["season"]),
                "stage": "playoffs",
                "match_id": match["match_id"],
                "official_match_id": int(modal["id"].removeprefix("modal2-")),
                "round": match["round"],
                "scheduled_at": pd.Timestamp(
                    f"{scheduled_date} {time.group(1)}", tz="Asia/Jakarta"
                ),
                "best_of": int(match["best_of"]),
                "team_a_id": names[0],
                "team_b_id": names[1],
                "team_a_score": score_a,
                "team_b_score": score_b,
                "winner_team_id": (names[0] if score_a > score_b else names[1])
                if completed
                else None,
                "status": "completed" if completed else "scheduled",
                "observed_at": observed_at.isoformat(),
                "source_url": SCHEDULE_URL,
                "routing_basis": "configured_bracket_order",
                "best_of_basis": "simulation_config",
            }
        )
    result = pd.DataFrame.from_records(records)
    if len(result) != len(matches) or result["official_match_id"].duplicated().any():
        raise ValueError("Official playoff match count/identifiers do not match the bracket.")
    for column in ("team_a_score", "team_b_score"):
        result[column] = result[column].astype("Int64")
    return result
