"""Deriva variables temporales del modelo a partir de una fecha calendario."""
from datetime import datetime

BASE_YEAR = 2011
TRAIN_YEAR_MIN = 2011
TRAIN_YEAR_MAX = 2012
MAX_YR = 3.0
MIN_YR = -2.0
YR_GROWTH_RATE = 0.25


def month_to_season(month):
    """1=invierno, 2=primavera, 3=verano, 4=otoño (según dataset)."""
    if month in (12, 1, 2, 3):
        return 1
    if month in (4, 5, 6):
        return 2
    if month in (7, 8, 9):
        return 3
    return 4


def effective_yr(year):
    """Convierte el año calendario a yr del modelo."""
    offset = year - BASE_YEAR
    if TRAIN_YEAR_MIN <= year <= TRAIN_YEAR_MAX:
        return float(offset)
    if offset < 0:
        return MIN_YR
    extra = offset - 1
    return min(1.0 + extra * YR_GROWTH_RATE, MAX_YR)


def parse_prediction_date(date_str, holiday=0):
    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    year = dt.year
    month = dt.month
    day = dt.day
    weekday = dt.weekday()
    # Python weekday: 0=lunes ... 6=domingo → dataset usa 0=domingo
    weekday = (weekday + 1) % 7
    is_holiday = int(holiday) == 1
    is_workingday = 1 <= weekday <= 5 and not is_holiday
    yr = effective_yr(year)

    return {
        "dia": day,
        "mnth": month,
        "anio": year,
        "yr": yr,
        "yr_natural": year - BASE_YEAR,
        "weekday": weekday,
        "season": month_to_season(month),
        "holiday": 1 if is_holiday else 0,
        "workingday": 1 if is_workingday else 0,
    }


def extrapolation_warning(year, yr, yr_natural):
    if TRAIN_YEAR_MIN <= year <= TRAIN_YEAR_MAX:
        return None
    if yr_natural == yr:
        return (
            f"Predicción extrapolada: el modelo se entrenó con datos de "
            f"{TRAIN_YEAR_MIN}-{TRAIN_YEAR_MAX}."
        )
    return (
        f"Predicción extrapolada: el modelo se entrenó con datos de "
        f"{TRAIN_YEAR_MIN}-{TRAIN_YEAR_MAX}. Para {year} se usa yr={yr:.2f} "
        f"(tendencia amortiguada; offset natural={yr_natural})."
    )
