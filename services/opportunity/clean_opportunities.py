from datetime import datetime, date
from typing import Optional

from db.db_conn import SessionLocal
from db.models.opportunity import Opportunity


DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
]


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def delete_closed_opportunities() -> int:
    today = date.today()
    deleted = 0

    with SessionLocal() as sess:
        query = sess.query(Opportunity).filter(
            Opportunity.close_date.isnot(None),
            Opportunity.forecasted_close_date.isnot(None),
        )

        for opp in query:
            close_date = parse_date(opp.close_date)
            forecast_date = parse_date(opp.forecasted_close_date)

            if close_date is None or forecast_date is None:
                continue

            if close_date <= today and forecast_date <= today:
                sess.delete(opp)
                deleted += 1

        if deleted:
            sess.commit()

    return deleted


if __name__ == "__main__":
    count = delete_closed_opportunities()
    print(f"Deleted {count} closed opportunity rows.")