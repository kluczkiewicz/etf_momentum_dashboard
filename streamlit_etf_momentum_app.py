from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="ETF Momentum Dashboard", layout="wide")


DEFAULT_TICKERS = ["VWCE.DE", "EIMI.L", "AGGG.L", "EUNL.DE"]

FALLBACK_NAMES: Dict[str, str] = {
    "VWCE.DE": "Vanguard FTSE All-World UCITS ETF USD Accumulation",
    "VWRA.L": "Vanguard FTSE All-World UCITS ETF USD Accumulation",
    "VWRL.L": "Vanguard FTSE All-World UCITS ETF USD Distributing",
    "EIMI.L": "iShares Core MSCI EM IMI UCITS ETF USD (Acc)",
    "AGGG.L": "iShares Core Global Aggregate Bond UCITS ETF USD (Dist)",
    "AGGU.L": "iShares Core Global Aggregate Bond UCITS ETF GBP Hedged (Acc)",
    "EUNL.DE": "iShares Core MSCI World UCITS ETF USD (Acc)",
    "IWDA.AS": "iShares Core MSCI World UCITS ETF USD (Acc)",
    "SXR8.DE": "iShares Core S&P 500 UCITS ETF USD (Acc)",
    "SPYL.DE": "SPDR S&P 500 UCITS ETF",
    "EQQQ.L": "Invesco EQQQ NASDAQ-100 UCITS ETF",
    "CNDX.L": "iShares NASDAQ 100 UCITS ETF USD (Acc)",
    "VUAA.DE": "Vanguard S&P 500 UCITS ETF USD Accumulation",
    "VUSA.DE": "Vanguard S&P 500 UCITS ETF USD Distributing",
    "IS3N.DE": "iShares Core MSCI EM IMI UCITS ETF USD (Acc)",
    "IMEA.AS": "iShares Core MSCI Europe UCITS ETF EUR (Acc)",
    "EXSA.DE": "iShares STOXX Europe 600 UCITS ETF (DE)",
    "IUSN.DE": "iShares MSCI World Small Cap UCITS ETF",
    "WSML.L": "iShares MSCI World Small Cap UCITS ETF",
    "SGLN.L": "iShares Physical Gold ETC",
    "4GLD.DE": "Xetra-Gold",
    "EMIM.L": "iShares MSCI Emerging Markets IMI UCITS ETF (Acc)",
}

ETF_CATALOG: List[Tuple[str, str]] = [
    ("VWCE.DE", FALLBACK_NAMES["VWCE.DE"]),
    ("VWRA.L", FALLBACK_NAMES["VWRA.L"]),
    ("VWRL.L", FALLBACK_NAMES["VWRL.L"]),
    ("EUNL.DE", FALLBACK_NAMES["EUNL.DE"]),
    ("IWDA.AS", FALLBACK_NAMES["IWDA.AS"]),
    ("EIMI.L", FALLBACK_NAMES["EIMI.L"]),
    ("IS3N.DE", FALLBACK_NAMES["IS3N.DE"]),
    ("EMIM.L", FALLBACK_NAMES["EMIM.L"]),
    ("AGGG.L", FALLBACK_NAMES["AGGG.L"]),
    ("AGGU.L", FALLBACK_NAMES["AGGU.L"]),
    ("SXR8.DE", FALLBACK_NAMES["SXR8.DE"]),
    ("SPYL.DE", FALLBACK_NAMES["SPYL.DE"]),
    ("VUAA.DE", FALLBACK_NAMES["VUAA.DE"]),
    ("VUSA.DE", FALLBACK_NAMES["VUSA.DE"]),
    ("EQQQ.L", FALLBACK_NAMES["EQQQ.L"]),
    ("CNDX.L", FALLBACK_NAMES["CNDX.L"]),
    ("IMEA.AS", FALLBACK_NAMES["IMEA.AS"]),
    ("EXSA.DE", FALLBACK_NAMES["EXSA.DE"]),
    ("IUSN.DE", FALLBACK_NAMES["IUSN.DE"]),
    ("WSML.L", FALLBACK_NAMES["WSML.L"]),
    ("SGLN.L", FALLBACK_NAMES["SGLN.L"]),
    ("4GLD.DE", FALLBACK_NAMES["4GLD.DE"]),
]

PERIOD_OPTIONS = {
    "1 month": 30,
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


@st.cache_data(show_spinner=False)
def resolve_name(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if ticker in FALLBACK_NAMES:
        return FALLBACK_NAMES[ticker]
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.get_info() or {}
        except Exception:
            info = getattr(t, "info", {}) or {}
        for key in ("longName", "shortName", "displayName", "name", "fundFamily"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        pass
    return FALLBACK_NAMES.get(ticker, ticker)


def calc_download_days(chart_days: int, required_trading_days: int) -> int:
    trading_to_calendar = int(required_trading_days * 1.7)
    return max(chart_days, trading_to_calendar) + 120


@st.cache_data(show_spinner=False)
def download_prices(tickers: Tuple[str, ...], chart_days: int, required_days: int) -> DownloadResult:
    warnings: List[str] = []
    names: Dict[str, str] = {}
    series_list = []

    period_days = calc_download_days(chart_days, required_days)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=period_days)

    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            warnings.append(f"Nie udało się pobrać danych dla {ticker}: {exc}")
            continue

        if data.empty or "Close" not in data.columns:
            warnings.append(f"Brak danych dla {ticker}.")
            continue

        names[ticker] = resolve_name(ticker)
        close = data["Close"].copy()
        close.name = ticker
        series_list.append(close)

    if not series_list:
        return DownloadResult(pd.DataFrame(), names, warnings)

    prices = pd.concat(series_list, axis=1).dropna(how="all")
    prices = prices.dropna(axis=0, how="any")
    return DownloadResult(prices=prices, names=names, warnings=warnings)


def normalize_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    return prices.div(prices.iloc[0]).mul(base)


def calculate_momentum_scores(prices: pd.DataFrame, lookback_days: int, skip_days: int) -> Tuple[pd.Series, str]:
    needed = lookback_days + skip_days + 1
    if len(prices) < needed:
        raise ValueError(f"Za mało danych do obliczenia momentum. Potrzeba co najmniej {needed} sesji.")
    end_idx = -1 - skip_days if skip_days > 0 else -1
    start_idx = end_idx - lookback_days
    past_prices = prices.iloc[start_idx]
    recent_prices = prices.iloc[end_idx]
    momentum = recent_prices / past_prices - 1
    window = f"{prices.index[start_idx].date()} → {prices.index[end_idx].date()}"
    return momentum.sort_values(ascending=False), window


def build_summary_table(prices: pd.DataFrame, momentum_scores: pd.Series, names: Dict[str, str], momentum_window: str) -> pd.DataFrame:
    latest = prices.iloc[-1]
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    summary = pd.DataFrame({
        "Ticker": prices.columns,
        "Name": [names.get(t, t) for t in prices.columns],
        "Last Price": latest.values,
        "Chart Return %": (total_return * 100).values,
        "Momentum %": (momentum_scores.reindex(prices.columns) * 100).values,
        "Momentum Window": momentum_window,
    })
    summary = summary.sort_values(["Momentum %", "Name"], ascending=[False, True]).reset_index(drop=True)
    summary.insert(0, "Rank", range(1, len(summary) + 1))
    return summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def parse_csv_param(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def sync_query_params() -> None:
    st.query_params["tickers"] = ",".join(st.session_state.tickers)
    st.query_params["added"] = ",".join(st.session_state.recent_added)
    st.query_params["removed"] = ",".join(st.session_state.recent_removed)


def init_state() -> None:
    if "tickers" not in st.session_state:
        query_tickers = parse_csv_param(st.query_params.get("tickers"))
        st.session_state.tickers = query_tickers or DEFAULT_TICKERS.copy()
    if "recent_added" not in st.session_state:
        st.session_state.recent_added = parse_csv_param(st.query_params.get("added"))[:3]
    if "recent_removed" not in st.session_state:
        st.session_state.recent_removed = parse_csv_param(st.query_params.get("removed"))[:3]
    sync_query_params()


def add_ticker(ticker: str) -> None:
    clean = ticker.strip().upper()
    if not clean:
        return
    if clean not in st.session_state.tickers:
        st.session_state.tickers.append(clean)
        st.session_state.recent_added = ([clean] + [t for t in st.session_state.recent_added if t != clean])[:3]
        sync_query_params()
    st.rerun()


def remove_ticker(ticker: str) -> None:
    st.session_state.tickers = [t for t in st.session_state.tickers if t != ticker]
    st.session_state.recent_removed = ([ticker] + [t for t in st.session_state.recent_removed if t != ticker])[:3]
    sync_query_params()
    st.rerun()


def render_recent_removed_horizontal() -> None:
    removed = st.session_state.recent_removed[:3]
    if not removed:
        st.caption("Brak ostatnio usuniętych.")
        return
    cols = st.columns(len(removed))
    for col, ticker in zip(cols, removed):
        with col:
            if st.button(f"↩ {ticker}", key=f"restore_{ticker}", use_container_width=True):
                add_ticker(ticker)


def render_etf_manager(names_map: Dict[str, str]) -> None:
    st.subheader("Lista ETF-ów")
    manage_tab, search_tab = st.tabs(["Moja lista", "Wyszukaj i dodaj"])

    with manage_tab:
        top_left, top_right = st.columns([1.2, 1.8])
        with top_left:
            with st.form("add_ticker_form", clear_on_submit=True):
                c1, c2 = st.columns([2.2, 1])
                ticker_input = c1.text_input(
                    "Ticker ETF",
                    label_visibility="collapsed",
                    placeholder="Np. VWCE.DE",
                )
                submitted = c2.form_submit_button("Dodaj", use_container_width=True)
                if submitted:
                    add_ticker(ticker_input)
        with top_right:
            st.caption("Ostatnio usunięte")
            render_recent_removed_horizontal()

        st.markdown("#### Aktualna lista")
        if st.session_state.tickers:
            for ticker in st.session_state.tickers:
                name = names_map.get(ticker, FALLBACK_NAMES.get(ticker, ticker))
                c1, c2, c3 = st.columns([5.5, 1.2, 1.1], vertical_alignment="center")
                c1.markdown(f"**{name}**")
                c2.caption(ticker)
                if c3.button("Usuń", key=f"remove_{ticker}", use_container_width=True):
                    remove_ticker(ticker)
                st.divider()
        else:
            st.info("Lista ETF-ów jest pusta. Dodaj pierwszy ticker.")

    with search_tab:
        options = [ticker for ticker, _ in ETF_CATALOG]
        selected = st.selectbox(
            "Znajdź ETF",
            options=options,
            index=None,
            placeholder="Zacznij wpisywać np. world, nasdaq, bond, VWCE...",
            format_func=lambda t: f"{t} — {FALLBACK_NAMES.get(t, t)}",
        )
        c1, c2 = st.columns([5, 1])
        if selected:
            c1.markdown(f"**{FALLBACK_NAMES.get(selected, selected)}**")
            c1.caption(selected)
        if c2.button("Dodaj do listy", key="search_add_selected", use_container_width=True, disabled=selected is None):
            add_ticker(selected)


def render_best_etf(best_row: pd.Series) -> None:
    st.subheader("Najlepszy ETF")
    st.markdown(f"### {best_row['Name']}")
    st.caption(f"Ticker: {best_row['Ticker']}")


def main() -> None:
    init_state()

    st.title("ETF Momentum Dashboard")
    st.caption("Porównanie ETF-ów, wykres, tabela i automatyczny wybór najlepszego ETF-a na podstawie momentum.")

    with st.sidebar:
        st.header("Ustawienia")
        chart_period_label = st.selectbox("Zakres wykresu", list(PERIOD_OPTIONS.keys()), index=3)
        chart_days = PERIOD_OPTIONS[chart_period_label]
        lookback_label = st.selectbox("Lookback momentum", list(MOMENTUM_LOOKBACK_OPTIONS.keys()), index=3)
        lookback_days = MOMENTUM_LOOKBACK_OPTIONS[lookback_label]
        skip_label = st.selectbox("Pomijaj ostatni okres", list(SKIP_OPTIONS.keys()), index=2)
        skip_days = SKIP_OPTIONS[skip_label]
        base_value = st.number_input("Wartość startowa do normalizacji wykresu", min_value=1.0, value=100.0, step=1.0)
        st.markdown("---")
        st.markdown("**Wskazówka:** tickery muszą być zgodne z Yahoo Finance, np. `VWCE.DE`, `EIMI.L`, `AGGG.L`, `EUNL.DE`.")

    tickers = tuple(dict.fromkeys([t.strip().upper() for t in st.session_state.tickers if t.strip()]))
    st.session_state.tickers = list(tickers)
    sync_query_params()

    if not tickers:
        render_etf_manager({})
        st.warning("Dodaj przynajmniej jeden ETF.")
        return

    required_days = lookback_days + skip_days + 1
    with st.spinner("Pobieram dane i nazwy ETF-ów..."):
        result = download_prices(tickers, chart_days=chart_days, required_days=required_days)

    names_map = {ticker: result.names.get(ticker, FALLBACK_NAMES.get(ticker, ticker)) for ticker in tickers}
    render_etf_manager(names_map)

    for warning in result.warnings:
        st.warning(warning)

    prices = result.prices
    if prices.empty:
        st.error("Nie udało się pobrać danych dla żadnego ETF-a.")
        return

    try:
        momentum_scores, momentum_window = calculate_momentum_scores(prices, lookback_days=lookback_days, skip_days=skip_days)
    except ValueError as exc:
        st.error(str(exc))
        return

    chart_calendar_days = PERIOD_OPTIONS[chart_period_label]
    chart_start = prices.index.max() - pd.Timedelta(days=chart_calendar_days)
    chart_prices = prices.loc[prices.index >= chart_start].copy()
    if len(chart_prices) < 20:
        chart_prices = prices.copy()

    normalized = normalize_prices(chart_prices, base=base_value)
    summary = build_summary_table(chart_prices, momentum_scores, names_map, momentum_window)
    best_row = summary.iloc[0]

    render_best_etf(best_row)

    st.subheader("Wykres")
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in normalized.columns:
        label = names_map.get(ticker, ticker)
        ax.plot(normalized.index, normalized[ticker], label=f"{label} [{ticker}]")
    ax.set_title(f"ETF performance ({chart_period_label}) — normalized to {base_value:.0f}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Wartość znormalizowana")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Tabela podsumowująca")
    display_summary = summary.copy()
    display_summary["Last Price"] = display_summary["Last Price"].map(lambda x: f"{x:.2f}")
    display_summary["Chart Return %"] = display_summary["Chart Return %"].map(lambda x: f"{x:.2f}%")
    display_summary["Momentum %"] = display_summary["Momentum %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    st.download_button(
        "Pobierz tabelę jako CSV",
        data=to_csv_bytes(summary),
        file_name="etf_momentum_summary.csv",
        mime="text/csv",
    )

    st.info(
        f"Dane do wykresu: {chart_period_label}. Momentum liczone na podstawie {lookback_label.lower()}, "
        f"z pominięciem {skip_label.lower()}. Aplikacja pobiera dłuższą historię niż sam wykres, "
        f"żeby uniknąć błędów z brakiem danych."
    )


if __name__ == "__main__":
    main()
