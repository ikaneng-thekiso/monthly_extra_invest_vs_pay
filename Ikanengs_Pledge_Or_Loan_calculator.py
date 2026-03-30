

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="Borrow vs Sell Portfolio Calculator",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(99,102,241,0.18), transparent 28%),
            radial-gradient(circle at top left, rgba(16,185,129,0.14), transparent 24%),
            linear-gradient(180deg, #0b1020 0%, #11182d 100%);
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    .hero {
        padding: 1.3rem 1.45rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(37,99,235,0.36), rgba(16,185,129,0.22));
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 18px 40px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.05rem;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: rgba(255,255,255,0.88);
        font-size: 1rem;
    }
    .glass {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1rem 1rem 1.05rem 1rem;
        box-shadow: 0 16px 34px rgba(0,0,0,0.20);
    }
    .pill {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        margin-right: 0.35rem;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 0.86rem;
    }
    .kpi {
        background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        min-height: 126px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.16);
    }
    .kpi-label {
        color: rgba(255,255,255,0.72);
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.12;
        margin-bottom: 0.25rem;
    }
    .kpi-note {
        color: rgba(255,255,255,0.78);
        font-size: 0.92rem;
    }
    .note {
        color: rgba(255,255,255,0.76);
        font-size: 0.93rem;
        margin-top: 0.8rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>�� Borrow vs Sell Portfolio Calculator</h1>
        <p>Compare the ending net worth after meeting the same cash need today.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<span class='pill'>One main table</span><span class='pill'>Base cost included</span><span class='pill'>Clear sale tax wording</span><span class='pill'>Net gain difference</span>",
    unsafe_allow_html=True,
)

sns.set_theme(style="whitegrid", context="notebook")


@dataclass
class Lot:
    date: str
    units: float
    price: float


def fmt_currency(x: float) -> str:
    return f"R{x:,.2f}"


def fmt_money_or_nil(x: Optional[float], nil_for_zero: bool = False) -> str:
    if x is None or pd.isna(x):
        return ""
    value = float(x)
    if nil_for_zero and abs(value) < 1e-9:
        return "Nil"
    return fmt_currency(value)


def render_kpi(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_lots_from_lump_sum(current_value: float, total_base_cost: float, price_per_unit: float = 100.0) -> List[Lot]:
    units = current_value / price_per_unit if price_per_unit > 0 else 0.0
    avg_cost = total_base_cost / units if units > 0 else 0.0
    return [Lot(date="Initial holding", units=units, price=avg_cost)]


def lots_from_transactions(df: pd.DataFrame) -> List[Lot]:
    clean = df.copy()
    clean["date"] = clean["date"].astype(str)
    clean["amount_invested"] = pd.to_numeric(clean["amount_invested"], errors="coerce").fillna(0.0)
    clean["price_per_unit"] = pd.to_numeric(clean["price_per_unit"], errors="coerce").fillna(0.0)
    clean = clean[(clean["amount_invested"] > 0) & (clean["price_per_unit"] > 0)]

    lots: List[Lot] = []
    for _, row in clean.iterrows():
        units = row["amount_invested"] / row["price_per_unit"]
        lots.append(Lot(date=str(row["date"]), units=float(units), price=float(row["price_per_unit"])))
    return lots


def portfolio_snapshot(lots: List[Lot], current_price: float) -> Dict[str, float]:
    total_units = sum(l.units for l in lots)
    total_cost = sum(l.units * l.price for l in lots)
    current_value = total_units * current_price
    unrealised_gain = current_value - total_cost
    avg_cost = total_cost / total_units if total_units else 0.0
    return {
        "total_units": total_units,
        "total_cost": total_cost,
        "current_value": current_value,
        "unrealised_gain": unrealised_gain,
        "avg_cost": avg_cost,
    }


def sell_using_weighted_average(lots: List[Lot], gross_sale_value: float, current_price: float) -> Dict[str, float]:
    snap = portfolio_snapshot(lots, current_price)
    total_units = snap["total_units"]
    if current_price <= 0 or total_units <= 0:
        return {"units_sold": 0.0, "proceeds": 0.0, "base_cost_sold": 0.0, "capital_gain": 0.0}

    units_sold = min(gross_sale_value / current_price, total_units)
    proceeds = units_sold * current_price
    base_cost_sold = units_sold * snap["avg_cost"]
    capital_gain = proceeds - base_cost_sold
    return {
        "units_sold": units_sold,
        "proceeds": proceeds,
        "base_cost_sold": base_cost_sold,
        "capital_gain": capital_gain,
    }


def sell_using_fifo(lots: List[Lot], gross_sale_value: float, current_price: float) -> Dict[str, float]:
    remaining_units_to_sell = gross_sale_value / current_price if current_price > 0 else 0.0
    proceeds = 0.0
    base_cost_sold = 0.0
    units_sold = 0.0

    for lot in lots:
        if remaining_units_to_sell <= 0:
            break
        take = min(lot.units, remaining_units_to_sell)
        units_sold += take
        proceeds += take * current_price
        base_cost_sold += take * lot.price
        remaining_units_to_sell -= take

    capital_gain = proceeds - base_cost_sold
    return {
        "units_sold": units_sold,
        "proceeds": proceeds,
        "base_cost_sold": base_cost_sold,
        "capital_gain": capital_gain,
    }


def cgt_tax(capital_gain: float, annual_exclusion: float, inclusion_rate: float, marginal_tax_rate: float) -> Dict[str, float]:
    gain = max(capital_gain, 0.0)
    taxable_gain_after_exclusion = max(gain - annual_exclusion, 0.0)
    taxable_income_addition = taxable_gain_after_exclusion * inclusion_rate
    tax = taxable_income_addition * marginal_tax_rate
    effective_rate_on_gain = (tax / gain) if gain > 0 else 0.0
    return {
        "annual_exclusion_used": min(gain, annual_exclusion),
        "taxable_gain_after_exclusion": taxable_gain_after_exclusion,
        "taxable_income_addition": taxable_income_addition,
        "tax": tax,
        "effective_rate_on_gain": effective_rate_on_gain,
    }


def calculate_sale_for_gross_value(
    lots: List[Lot],
    gross_sale_value: float,
    current_price: float,
    annual_exclusion: float,
    inclusion_rate: float,
    marginal_tax_rate: float,
    method: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    sale_fn = sell_using_weighted_average if method == "Weighted average" else sell_using_fifo
    sale = sale_fn(lots, gross_sale_value, current_price)
    tax = cgt_tax(sale["capital_gain"], annual_exclusion, inclusion_rate, marginal_tax_rate)
    sale["net_cash_after_tax"] = sale["proceeds"] - tax["tax"]
    return sale, tax


def solve_sale_for_net_cash(
    lots: List[Lot],
    target_net_cash: float,
    current_price: float,
    annual_exclusion: float,
    inclusion_rate: float,
    marginal_tax_rate: float,
    method: str,
) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    snap = portfolio_snapshot(lots, current_price)
    max_gross_sale = snap["current_value"]

    sale_full, tax_full = calculate_sale_for_gross_value(
        lots=lots,
        gross_sale_value=max_gross_sale,
        current_price=current_price,
        annual_exclusion=annual_exclusion,
        inclusion_rate=inclusion_rate,
        marginal_tax_rate=marginal_tax_rate,
        method=method,
    )

    if sale_full["net_cash_after_tax"] + 1e-9 < target_net_cash:
        return sale_full, tax_full, False

    low = 0.0
    high = max_gross_sale
    for _ in range(80):
        mid = (low + high) / 2
        sale_mid, tax_mid = calculate_sale_for_gross_value(
            lots=lots,
            gross_sale_value=mid,
            current_price=current_price,
            annual_exclusion=annual_exclusion,
            inclusion_rate=inclusion_rate,
            marginal_tax_rate=marginal_tax_rate,
            method=method,
        )
        if sale_mid["net_cash_after_tax"] >= target_net_cash:
            high = mid
        else:
            low = mid

    sale, tax = calculate_sale_for_gross_value(
        lots=lots,
        gross_sale_value=high,
        current_price=current_price,
        annual_exclusion=annual_exclusion,
        inclusion_rate=inclusion_rate,
        marginal_tax_rate=marginal_tax_rate,
        method=method,
    )
    return sale, tax, True


def build_display_table(
    years: int,
    snap: Dict[str, float],
    cash_needed: float,
    sale: Dict[str, float],
    tax: Dict[str, float],
    untouched_future_value: float,
    sell_remaining_today: float,
    sell_future_value: float,
    borrow_future_loan_balance: float,
    borrow_ending_net_worth: float,
    sell_ending_net_worth: float,
    borrow_drag: float,
    sell_drag: float,
    net_gain_difference: float,
) -> pd.DataFrame:
    rows = [
        {
            "Metric": "Current portfolio value",
            "Sell to raise cash": fmt_currency(snap["current_value"]),
            "Borrow against portfolio": fmt_currency(snap["current_value"]),
        },
        {
            "Metric": f"Future value if untouched after {years} year(s)",
            "Sell to raise cash": fmt_currency(untouched_future_value),
            "Borrow against portfolio": fmt_currency(untouched_future_value),
        },
        {
            "Metric": "Gross assets sold today / loan raised today",
            "Sell to raise cash": fmt_currency(sale["proceeds"]),
            "Borrow against portfolio": fmt_currency(cash_needed),
        },
        {
            "Metric": "Base cost of units sold",
            "Sell to raise cash": fmt_currency(sale["base_cost_sold"]),
            "Borrow against portfolio": "Nil",
        },
        {
            "Metric": "Capital gain on units sold",
            "Sell to raise cash": fmt_money_or_nil(sale["capital_gain"], nil_for_zero=True),
            "Borrow against portfolio": "Nil",
        },
        {
            "Metric": "Annual exclusion used",
            "Sell to raise cash": fmt_money_or_nil(tax["annual_exclusion_used"], nil_for_zero=True),
            "Borrow against portfolio": "Nil",
        },
        {
            "Metric": "Estimated tax on sale today",
            "Sell to raise cash": fmt_money_or_nil(tax["tax"], nil_for_zero=True),
            "Borrow against portfolio": "Nil",
        },
        {
            "Metric": "Net cash available today",
            "Sell to raise cash": fmt_currency(sale["net_cash_after_tax"]),
            "Borrow against portfolio": fmt_currency(cash_needed),
        },
        {
            "Metric": "Portfolio remaining today",
            "Sell to raise cash": fmt_currency(sell_remaining_today),
            "Borrow against portfolio": fmt_currency(snap["current_value"]),
        },
        {
            "Metric": f"Future value of portfolio after {years} year(s)",
            "Sell to raise cash": fmt_currency(sell_future_value),
            "Borrow against portfolio": fmt_currency(untouched_future_value),
        },
        {
            "Metric": f"Future loan balance after {years} year(s)",
            "Sell to raise cash": "Nil",
            "Borrow against portfolio": fmt_currency(borrow_future_loan_balance),
        },
        {
            "Metric": f"Key result: Ending net worth after {years} year(s)",
            "Sell to raise cash": fmt_currency(sell_ending_net_worth),
            "Borrow against portfolio": fmt_currency(borrow_ending_net_worth),
        },
        {
            "Metric": "Net worth drag versus untouched portfolio",
            "Sell to raise cash": fmt_currency(sell_drag),
            "Borrow against portfolio": fmt_currency(borrow_drag),
        },
        {
            "Metric": f"Key result: Net gain difference after {years} year(s)",
            "Sell to raise cash": "Reference",
            "Borrow against portfolio": fmt_money_or_nil(net_gain_difference, nil_for_zero=True),
        },
    ]
    return pd.DataFrame(rows)


def build_chart_df(
    years: int,
    untouched_future_value: float,
    sell_ending_net_worth: float,
    borrow_ending_net_worth: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Scenario": f"Untouched after {years} year(s)", "Amount": untouched_future_value},
            {"Scenario": f"Sell after {years} year(s)", "Amount": sell_ending_net_worth},
            {"Scenario": f"Borrow after {years} year(s)", "Amount": borrow_ending_net_worth},
        ]
    )


def render_chart(chart_df: pd.DataFrame, years: int) -> None:
    palette = {
        chart_df.loc[0, "Scenario"]: "#64748b",
        chart_df.loc[1, "Scenario"]: "#f59e0b",
        chart_df.loc[2, "Scenario"]: "#22c55e",
    }

    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfdff")

    sns.barplot(data=chart_df, x="Amount", y="Scenario", palette=palette, orient="h", ax=ax)
    ax.set_title(f"Ending net worth comparison after {years} year(s)", fontsize=17, fontweight="bold", pad=14)
    ax.set_xlabel("Rand value")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.18)

    max_amount = max(chart_df["Amount"].max(), 1.0)
    for patch, (_, row) in zip(ax.patches, chart_df.iterrows()):
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(width + max_amount * 0.015, y, fmt_currency(float(row["Amount"])), va="center", ha="left", fontsize=10.5)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main() -> None:
    with st.sidebar:
        st.header("Inputs")
        mode = st.radio(
            "Portfolio setup",
            ["Simple lump sum portfolio", "Recurring contributions or mixed buys"],
            index=0,
        )

        st.subheader("Cash need")
        cash_needed = st.number_input("Cash you need now", min_value=0.0, value=15000.0, step=1000.0)

        st.subheader("Market and borrowing")
        expected_return = st.slider("Expected annual portfolio return %", min_value=-20.0, max_value=30.0, value=15.0, step=0.5) / 100
        loan_rate = st.slider("Loan interest rate %", min_value=0.0, max_value=35.0, value=13.25, step=0.25) / 100
        years = st.slider("Time horizon in years", min_value=1, max_value=10, value=1, step=1)

        st.subheader("Tax assumptions")
        annual_exclusion = st.number_input("Annual capital gains exclusion", min_value=0.0, value=50000.0, step=5000.0)
        inclusion_rate = st.slider("Capital gains inclusion rate %", min_value=0.0, max_value=100.0, value=40.0, step=1.0) / 100
        marginal_tax_rate = st.slider("Marginal tax rate %", min_value=0.0, max_value=45.0, value=45.0, step=1.0) / 100
        method = st.selectbox("Cost basis method", ["Weighted average", "FIFO"])

    if mode == "Simple lump sum portfolio":
        st.markdown("### Portfolio")
        c1, c2, c3 = st.columns(3)
        with c1:
            current_value = st.number_input("Current portfolio value", min_value=0.0, value=57500.0, step=1000.0)
        with c2:
            total_base_cost = st.number_input("Total amount originally invested", min_value=0.0, value=50000.0, step=1000.0)
        with c3:
            current_price = st.number_input("Reference unit price", min_value=0.01, value=100.0, step=1.0)
        lots = build_lots_from_lump_sum(current_value, total_base_cost, current_price)
    else:
        st.markdown("### Portfolio transaction table")
        st.caption("Enter your buy transactions. Tax is calculated only on the portion sold.")
        current_price = st.number_input("Current unit price", min_value=0.01, value=125.0, step=1.0)

        default_df = pd.DataFrame(
            [
                {"date": "2024-01-31", "amount_invested": 100000.0, "price_per_unit": 100.0},
                {"date": "2024-02-29", "amount_invested": 10000.0, "price_per_unit": 102.0},
                {"date": "2024-03-31", "amount_invested": 10000.0, "price_per_unit": 104.0},
                {"date": "2024-04-30", "amount_invested": 10000.0, "price_per_unit": 106.0},
                {"date": "2024-05-31", "amount_invested": 10000.0, "price_per_unit": 108.0},
            ]
        )

        edited = st.data_editor(
            default_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "date": st.column_config.TextColumn("Date"),
                "amount_invested": st.column_config.NumberColumn("Amount invested", format="R %.2f", min_value=0.0),
                "price_per_unit": st.column_config.NumberColumn("Buy price per unit", format="R %.2f", min_value=0.01),
            },
            key="tx_editor",
        )
        lots = lots_from_transactions(edited)

    snap = portfolio_snapshot(lots, current_price)
    growth_factor = (1 + expected_return) ** years
    loan_factor = (1 + loan_rate) ** years

    sale, tax, can_fund_target = solve_sale_for_net_cash(
        lots=lots,
        target_net_cash=cash_needed,
        current_price=current_price,
        annual_exclusion=annual_exclusion,
        inclusion_rate=inclusion_rate,
        marginal_tax_rate=marginal_tax_rate,
        method=method,
    )

    untouched_future_value = snap["current_value"] * growth_factor
    sell_remaining_today = max(snap["current_value"] - sale["proceeds"], 0.0)
    sell_future_value = sell_remaining_today * growth_factor
    sell_ending_net_worth = sell_future_value

    borrow_future_loan_balance = cash_needed * loan_factor
    borrow_ending_net_worth = untouched_future_value - borrow_future_loan_balance

    sell_drag = untouched_future_value - sell_ending_net_worth
    borrow_drag = untouched_future_value - borrow_ending_net_worth
    net_gain_difference = borrow_ending_net_worth - sell_ending_net_worth

    if not can_fund_target:
        st.warning("Your portfolio cannot raise the full requested net cash after the estimated tax at the current price.")

    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Current portfolio value", fmt_currency(snap["current_value"]), f"Unrealised gain: {fmt_currency(snap['unrealised_gain'])}")
    with k2:
        render_kpi("Base cost of units sold", fmt_currency(sale["base_cost_sold"]), "This is the original cost attached to the units being sold")
    with k3:
        render_kpi("Estimated tax on sale today", fmt_money_or_nil(tax["tax"], nil_for_zero=True), "This tax is created only by the gain in the units sold now")
    with k4:
        render_kpi(f"Net gain difference after {years} year(s)", fmt_money_or_nil(net_gain_difference, nil_for_zero=True), "Borrow minus sell. Positive means borrowing ends ahead")

    display_table = build_display_table(
        years=years,
        snap=snap,
        cash_needed=cash_needed,
        sale=sale,
        tax=tax,
        untouched_future_value=untouched_future_value,
        sell_remaining_today=sell_remaining_today,
        sell_future_value=sell_future_value,
        borrow_future_loan_balance=borrow_future_loan_balance,
        borrow_ending_net_worth=borrow_ending_net_worth,
        sell_ending_net_worth=sell_ending_net_worth,
        borrow_drag=borrow_drag,
        sell_drag=sell_drag,
        net_gain_difference=net_gain_difference,
    )

    chart_df = build_chart_df(
        years=years,
        untouched_future_value=untouched_future_value,
        sell_ending_net_worth=sell_ending_net_worth,
        borrow_ending_net_worth=borrow_ending_net_worth,
    )

    st.write("")
    left, right = st.columns([1.3, 0.9], gap="large")

    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Net worth comparison table")
        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Sell to raise cash": st.column_config.TextColumn("Sell to raise cash", width="small"),
                "Borrow against portfolio": st.column_config.TextColumn("Borrow against portfolio", width="small"),
            },
        )
        st.markdown(
            f"<div class='note'>Estimated tax on sale today means the tax triggered by selling now. It is based only on the gain in the units sold, not on the whole portfolio. In this case, gain = gross sale {fmt_currency(sale['proceeds'])} less base cost {fmt_currency(sale['base_cost_sold'])} = {fmt_currency(sale['capital_gain'])}.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Visual comparison")
        render_chart(chart_df, years)
        st.markdown(
            f"<div class='note'>The sell path grows only the portfolio left after the sale. The borrow path keeps the full portfolio invested, then subtracts the future loan balance of {fmt_currency(borrow_future_loan_balance)}.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


