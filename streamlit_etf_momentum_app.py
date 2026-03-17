from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="ETF Momentum Dashboard", layout="wide")

DEFAULT_ETFS = ["VWCE.DE", "EIMI.L", "AGGG.L"]
QUERY_PARAM_KEY = "etfs"

CHART_PERIOD_OPTIONS: Dict[str, int] = {
    "3 months": 90,
    "6 months": 180,
    "1 year": 365,
    "2 years": 730,
    "3 years": 1095,
    "5 years": 1825,
}

MOMENTUM_LOOKBACK_OPTIONS = {
    "1 month": 21,
    "3 months": 63,
    "6 months": 126,
    "12 months": 252,
}

SKIP_OPTIONS = {
    "0 days": 0,
    "5 trading days": 5,
    "21 trading days (~1 month)": 21,
}


@dataclass
class DownloadResult:
    prices: pd.DataFrame
    names: Dict[str, str]
    warnings: List[str]
    fetch_start: date
    chart_start: date


@st.cache_data(show_spinner=False)
def resolve_ticker_name(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not clean:
        return ""

    try:
        obj = yf.Ticker(clean)
        info = obj.info or {}
        for key in ("longName", "shortName", "displayName", "name"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        pass

    return clean


@st.cache_data(show_spinner=False)
def download_prices(tickers: Tuple[str, ...], chart_days: int, lookback_days: int, skip_days: int) -> DownloadResult:
    warnings: List[str] = []
    series_list = []
    names: Dict[str, str] = {}

    required_trading_days = max(chart_days, lookback_days + skip_days + 30)
    calendar_days = max(int(required_trading_days * 1.8), chart_days + 30)
    end_date = date.today()
    fetch_start = end_date - timedelta(days=calendar_days)
    chart_start = end_date - timedelta(days=chart_days)

    for ticker in tickers:
        clean = str(ticker).strip().upper()
        if not clean:
            continue

        names[clean] = resolve_ticker_name(clean)

        try:
            data = yf.download(
                clean,
                start=fetch_start,
                end=end_date + timedelta(days=1),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            warnings.append(f"Nie udało się pobrać danych dla {clean}: {exc}")
            continue

        if data.empty or "Close" not in data.columns:
            warnings.append(f"Brak danych dla {names[clean]} ({clean}).")
            continue

        close = data["Close"].copy()
        close.name = clean
        series_list.append(close)

    if not series_list:
        return DownloadResult(pd.DataFrame(), names, warnings, fetch_start, chart_start)

    prices = pd.concat(series_list, axis=1).dropna(how="all")
    prices = prices.dropna(axis=0, how="any")
    return DownloadResult(prices=prices, names=names, warnings=warnings, fetch_start=fetch_start, chart_start=chart_start)


def normalize_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    return prices.div(prices.iloc[0]).mul(base)


def calculate_chart_returns(prices: pd.DataFrame) -> pd.Series:
    return prices.iloc[-1] / prices.iloc[0] - 1


def calculate_momentum_scores(prices: pd.DataFrame, lookback_days: int, skip_days: int) -> pd.Series:
    needed_rows = lookback_days + skip_days + 1
    if len(prices) < needed_rows:
        raise ValueError(
            f"Za mało danych do obliczenia momentum. Potrzeba co najmniej {needed_rows} sesji, a pobrano {len(prices)}."
        )

    end_idx = len(prices) - 1 - skip_days
    start_idx = end_idx - lookback_days
    if start_idx < 0:
        raise ValueError("Za mało danych do obliczenia momentum.")

    recent_prices = prices.iloc[end_idx]
    past_prices = prices.iloc[start_idx]
    return (recent_prices / past_prices - 1).sort_values(ascending=False)


def build_summary_table(
    prices_chart: pd.DataFrame,
    prices_full: pd.DataFrame,
    names: Dict[str, str],
    momentum_scores: pd.Series,
    lookback_days: int,
    skip_days: int,
) -> pd.DataFrame:
    latest = prices_chart.iloc[-1]
    chart_return = calculate_chart_returns(prices_chart)

    momentum_end_idx = len(prices_full) - 1 - skip_days
    momentum_start_idx = momentum_end_idx - lookback_days
    momentum_start_date = prices_full.index[momentum_start_idx].date()
    momentum_end_date = prices_full.index[momentum_end_idx].date()

    rows = []
    ordered_tickers = momentum_scores.sort_values(ascending=False).index.tolist()
    for rank, ticker in enumerate(ordered_tickers, start=1):
        rows.append(
            {
                "Rank": rank,
                "Name": names.get(ticker, ticker),
                "Ticker": ticker,
                "Last Price": float(latest[ticker]),
                "Chart Return %": float(chart_return[ticker] * 100),
                "Momentum %": float(momentum_scores[ticker] * 100),
                "Momentum Window": f"{momentum_start_date} → {momentum_end_date}",
            }
        )

    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def persist_etf_list() -> None:
    cleaned = [clean_ticker(t) for t in st.session_state.etf_list if clean_ticker(t)]
    st.session_state.etf_list = cleaned
    st.query_params[QUERY_PARAM_KEY] = ",".join(cleaned)


def load_initial_etf_list() -> List[str]:
    raw = st.query_params.get(QUERY_PARAM_KEY, "")
    if raw:
        loaded = [clean_ticker(x) for x in str(raw).split(",") if clean_ticker(x)]
        if loaded:
            return loaded
    return DEFAULT_ETFS.copy()


def add_ticker_to_state(raw_ticker: str) -> bool:
    ticker = clean_ticker(raw_ticker)
    if not ticker:
        st.warning("Wpisz ticker ETF-a.")
        return False
    if ticker in st.session_state.etf_list:
        st.info(f"{ticker} już jest na liście.")
        return False
    st.session_state.etf_list.append(ticker)
    persist_etf_list()
    return True


def remove_ticker_from_state(ticker: str) -> None:
    st.session_state.etf_list = [t for t in st.session_state.etf_list if t != ticker]
    persist_etf_list()


def render_etf_manager(names: Dict[str, str]) -> None:
    st.subheader("Lista ETF-ów")

    add_col1, add_col2 = st.columns([5, 1])
    with add_col1:
        new_ticker = st.text_input(
            "Dodaj ETF po tickerze",
            value="",
            placeholder="Np. VWCE.DE, EIMI.L, AGGG.L, EUNL.DE",
            help="Wpisz ticker z Yahoo Finance i kliknij Dodaj.",
            key="new_ticker_input",
            label_visibility="collapsed",
        )
    with add_col2:
        if st.button("Dodaj", use_container_width=True):
            if add_ticker_to_state(new_ticker):
                st.session_state.new_ticker_input = ""
                st.rerun()

    if not st.session_state.etf_list:
        st.info("Lista ETF-ów jest pusta. Dodaj przynajmniej jeden ticker.")
        return

    for idx, ticker in enumerate(st.session_state.etf_list):
        row_cols = st.columns([2, 8, 2])
        row_cols[0].code(ticker)
        row_cols[1].write(names.get(ticker, resolve_ticker_name(ticker)))
        if row_cols[2].button("Usuń", key=f"remove_{ticker}_{idx}", use_container_width=True):
            remove_ticker_from_state(ticker)
            st.rerun()


def main() -> None:
    st.title("ETF Momentum Dashboard")
    st.caption(
        "Porównanie ETF-ów, wykres, tabela i automatyczny wybór najlepszego ETF-a na podstawie momentum. "
        "Wpisujesz tylko ticker z Yahoo Finance, a aplikacja sama spróbuje pobrać pełną nazwę funduszu."
    )

    if "etf_list" not in st.session_state:
        st.session_state.etf_list = load_initial_etf_list()
        persist_etf_list()

    with st.sidebar:
        st.header("Ustawienia")
        chart_period_label = st.selectbox("Zakres wykresu", list(CHART_PERIOD_OPTIONS.keys()), index=2)
        chart_days = CHART_PERIOD_OPTIONS[chart_period_label]

        lookback_label = st.selectbox("Lookback momentum", list(MOMENTUM_LOOKBACK_OPTIONS.keys()), index=3)
        lookback_days = MOMENTUM_LOOKBACK_OPTIONS[lookback_label]

        skip_label = st.selectbox("Pomijaj ostatni okres", list(SKIP_OPTIONS.keys()), index=2)
        skip_days = SKIP_OPTIONS[skip_label]

        base_value = st.number_input("Wartość startowa do normalizacji wykresu", min_value=1.0, value=100.0, step=1.0)
        st.markdown("---")
        if st.button("Przywróć domyślne ETF-y", use_container_width=True):
            st.session_state.etf_list = DEFAULT_ETFS.copy()
            persist_etf_list()
            st.rerun()
        st.markdown(
            "**Wskazówka:** wpisuj tickery zgodne z Yahoo Finance, np. `VWCE.DE`, `EIMI.L`, `AGGG.L`, `AGGU.L`."
        )

    tickers = tuple(clean_ticker(t) for t in st.session_state.etf_list if clean_ticker(t))
    if not tickers:
        render_etf_manager({})
        return

    with st.spinner("Pobieram dane i nazwy ETF-ów..."):
        result = download_prices(tickers, chart_days=chart_days, lookback_days=lookback_days, skip_days=skip_days)

    render_etf_manager(result.names)

    if not st.session_state.etf_list:
        return

    for warning in result.warnings:
        st.warning(warning)

    prices_full = result.prices
    if prices_full.empty:
        st.error("Nie udało się pobrać danych dla żadnego ETF-a.")
        return

    prices_chart = prices_full.loc[prices_full.index.date >= result.chart_start].copy()
    if prices_chart.empty:
        st.error("Brak danych w wybranym zakresie wykresu.")
        return

    if len(prices_chart) < 2:
        st.error("Za mało danych do narysowania wykresu.")
        return

    try:
        momentum_scores = calculate_momentum_scores(prices_full, lookback_days=lookback_days, skip_days=skip_days)
    except ValueError as exc:
        st.error(str(exc))
        return

    normalized = normalize_prices(prices_chart, base=base_value)
    summary = build_summary_table(
        prices_chart=prices_chart,
        prices_full=prices_full,
        names=result.names,
        momentum_scores=momentum_scores,
        lookback_days=lookback_days,
        skip_days=skip_days,
    ).sort_values("Rank", ascending=True).reset_index(drop=True)

    best_row = summary.iloc[0]
    required_trading_days = lookback_days + skip_days + 30
    st.info(
        f"Dane do wykresu: **{chart_period_label}**.  \n"
        f"Momentum liczone na podstawie **{lookback_label}**, z pominięciem **{skip_label}**.  \n"
        f"Aplikacja automatycznie pobiera szerszy zakres historii, żeby uniknąć błędów z brakiem danych (minimum ok. **{required_trading_days} sesji**)."
    )

    st.subheader("Najlepszy ETF")
    st.write(f"**{best_row['Name']}** (`{best_row['Ticker']}`)")

    st.subheader("Wykres")
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in normalized.columns:
        label = f"{result.names.get(ticker, ticker)} [{ticker}]"
        if ticker == best_row["Ticker"]:
            label += " ← BEST"
        ax.plot(normalized.index, normalized[ticker], label=label)
    ax.set_title(f"ETF performance ({chart_period_label}) — normalized to {base_value:.0f}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Wartość znormalizowana")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Tabela podsumowująca")
    display_summary = summary.copy()
    display_summary["Last Price"] = display_summary["Last Price"].map(lambda x: f"{x:.2f}")
    display_summary["Chart Return %"] = display_summary["Chart Return %"].map(lambda x: f"{x:.2f}%")
    display_summary["Momentum %"] = display_summary["Momentum %"].map(lambda x: f"{x:.2f}%")
    st.table(display_summary)

    st.download_button(
        "Pobierz tabelę jako CSV",
        data=to_csv_bytes(summary),
        file_name="etf_momentum_summary.csv",
        mime="text/csv",
    )

    with st.expander("Jak deployować na Streamlit Community Cloud"):
        st.markdown(
            """
1. Wrzuć ten plik do repozytorium na GitHubie jako `streamlit_etf_momentum_app.py`.
2. Dodaj plik `requirements.txt`.
3. Wejdź do Streamlit Community Cloud i połącz repozytorium.
4. Jako **Main file path** ustaw `streamlit_etf_momentum_app.py`.
5. Zdeployuj aplikację.
            """
        )


if __name__ == "__main__":
    main()
