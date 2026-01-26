from datetime import datetime

def format_date_iso(dt: datetime) -> str:
    return dt.isoformat()

def format_currency(amount: float) -> str:
    return f"${amount:.2f}"
