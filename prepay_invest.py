import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Invest vs Home Loan", layout="wide")


def monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    if months <= 0:
        return 0.0
    r = annual_rate / 100 / 12
    if abs(r) < 1e-12:
        return principal / months
    return principal * r / (1 - (1 + r) ** (-months))


def remaining_balance(principal: float, annual_rate: float, payment: float, months_paid: int) -> float:
    r = annual_rate / 100 / 12
    if months_paid <= 0:
        return principal
    if abs(r) < 1e-12:
        return max(principal - payment * months_paid, 0.0)
    balance = principal * (1 + r) ** months_paid - payment * (((1 + r) ** months_paid - 1) / r)
    return max(balance, 0.0)


def payoff_months(principal: float, annual_rate: float, payment: float, max_months: int = 1000 * 12) -> int | None:
    if principal <= 0:
        return 0
    r = annual_rate / 100 / 12
    balance = principal
    for month in range(1, max_months + 1):
        interest = balance * r
        if payment <= interest and balance > 0:
            return None
        amount_due = balance + interest
        actual_payment = min(payment, amount_due)
        principal_paid = max(actual_payment - interest, 0.0)
        balance = max(balance - principal_paid, 0.0)
        if balance <= 1e-10:
            return month
    return None


def simulate_strategy(
    principal: float,
    annual_loan_rate: float,
    annual_invest_return: float,
    base_payment: float,
    extra_monthly: float,
    horizon_months: int,
    strategy: str,
    schedule_month_start: int,
) -> tuple[pd.DataFrame, float, int | None]:
    loan_r = annual_loan_rate / 100 / 12
    invest_r = annual_invest_return / 100 / 12
    total_budget = base_payment + extra_monthly

    balance = principal
    investment_value = 0.0
    total_interest = 0.0
    payoff_month_from_now = None
    rows: list[dict[str, float | int]] = []

    for month_from_now in range(1, horizon_months + 1):
        start_balance = balance
        interest = start_balance * loan_r
        amount_due = start_balance + interest

        if strategy == "invest":
            target_loan_payment = base_payment
        elif strategy == "prepay":
            target_loan_payment = total_budget
        else:
            raise ValueError("strategy must be 'invest' or 'prepay'")

        actual_loan_payment = min(target_loan_payment, amount_due)
        principal_paid = max(actual_loan_payment - interest, 0.0)
        balance = max(start_balance - principal_paid, 0.0)

        if balance <= 1e-10:
            balance = 0.0
            if payoff_month_from_now is None:
                payoff_month_from_now = month_from_now

        investment_contribution = max(total_budget - actual_loan_payment, 0.0)
        investment_value = investment_value * (1 + invest_r) + investment_contribution
        net_worth = investment_value - balance
        total_interest += interest

        rows.append(
            {
                "MonthFromNow": month_from_now,
                "ScheduleMonth": schedule_month_start + month_from_now,
                "LoanPayment": actual_loan_payment,
                "Interest": interest,
                "PrincipalPaid": principal_paid,
                "LoanBalance": balance,
                "InvestmentContribution": investment_contribution,
                "InvestmentValue": investment_value,
                "NetWorth": net_worth,
            }
        )

    return pd.DataFrame(rows), total_interest, payoff_month_from_now


def deflate_series(series: pd.Series, annual_inflation: float, months: pd.Series) -> pd.Series:
    monthly_inflation = annual_inflation / 100 / 12
    if abs(monthly_inflation) < 1e-12:
        return series
    factor = (1 + monthly_inflation) ** months
    return series / factor


st.title("Invest vs Home Loan Calculator")
st.caption("Compare investing extra cash versus paying extra into your bond, using the bond stage you are at right now.")

st.sidebar.header("Inputs")
loan_stage = st.sidebar.radio(
    "Where are you on the bond now?",
    ["Starting today", "Already part way through"],
    index=0,
)

home_price = st.sidebar.number_input("Home price", min_value=100000, value=1500000, step=50000)
deposit = st.sidebar.number_input("Deposit", min_value=0, value=150000, step=10000)
loan_rate = st.sidebar.number_input("Current home loan interest rate (%)", min_value=0.0, value=11.75, step=0.1)
loan_term_years = st.sidebar.number_input("Original loan term (years)", min_value=5, value=20, step=1)
extra_monthly = st.sidebar.number_input("Extra monthly cash available", min_value=0, value=3000, step=100)
invest_return = st.sidebar.number_input("Expected investment return (%)", min_value=0.0, value=12.0, step=0.1)
inflation = st.sidebar.number_input("Inflation (%)", min_value=0.0, value=5.0, step=0.1)
show_real = st.sidebar.checkbox("Show inflation adjusted values", value=False)

original_loan_amount = max(home_price - deposit, 0)
original_term_months = int(loan_term_years * 12)
original_scheduled_payment = monthly_payment(original_loan_amount, loan_rate, original_term_months)

if original_loan_amount == 0:
    st.success("No loan is needed because the deposit covers the full home price.")
    st.stop()

current_month = 0
loan_amount = original_loan_amount
remaining_months_original = original_term_months
scheduled_balance_now = original_loan_amount
payment_mode = "Keep original scheduled payment"

if loan_stage == "Already part way through":
    current_month = int(st.sidebar.number_input("Months already paid", min_value=0, value=36, step=1))
    current_month = min(current_month, max(original_term_months - 1, 0))
    remaining_months_original = max(original_term_months - current_month, 1)
    scheduled_balance_now = remaining_balance(
        original_loan_amount,
        loan_rate,
        original_scheduled_payment,
        current_month,
    )
    loan_amount = float(
        st.sidebar.number_input(
            "Current bond balance",
            min_value=0,
            value=int(round(scheduled_balance_now, -4)) if scheduled_balance_now > 0 else 0,
            step=10000,
            help="Use the actual current balance from your lender if it differs from the scheduled balance.",
        )
    )
    payment_mode = st.sidebar.radio(
        "How should the current required bond payment be handled?",
        [
            "Keep original scheduled payment",
            "Enter actual current payment",
            "Recalculate to finish by original term",
        ],
        index=0,
        help="This matters if your current balance is not exactly on the original schedule, or if your bank changed the instalment.",
    )

if loan_stage == "Starting today":
    base_payment = original_scheduled_payment
    current_payment_note = "Using the standard scheduled payment from the original loan inputs."
else:
    if payment_mode == "Keep original scheduled payment":
        base_payment = original_scheduled_payment
        current_payment_note = "Using the original scheduled monthly payment."
    elif payment_mode == "Enter actual current payment":
        base_payment = float(
            st.sidebar.number_input(
                "Actual current required monthly payment",
                min_value=0.0,
                value=float(round(original_scheduled_payment, 2)),
                step=100.0,
            )
        )
        current_payment_note = "Using the payment amount you entered."
    else:
        base_payment = monthly_payment(loan_amount, loan_rate, remaining_months_original)
        current_payment_note = "Using a recast payment that finishes exactly at the end of the original term."

if loan_amount <= 0:
    st.success("The current bond balance is already zero.")
    st.stop()

loan_r_monthly = loan_rate / 100 / 12
first_month_interest = loan_amount * loan_r_monthly
if loan_rate > 0 and base_payment <= first_month_interest:
    st.error("The current monthly payment is too low to reduce the bond at this interest rate.")
    st.stop()

payoff_invest_guess = payoff_months(loan_amount, loan_rate, base_payment)
payoff_prepay_guess = payoff_months(loan_amount, loan_rate, base_payment + extra_monthly)

if payoff_invest_guess is None:
    st.error("With the current payment, the loan will not amortize. Increase the payment or lower the rate.")
    st.stop()

if payoff_prepay_guess is None:
    st.error("Even the payment plus extra does not amortize the loan. Increase the payment or lower the rate.")
    st.stop()

auto_horizon = max(remaining_months_original, payoff_invest_guess, payoff_prepay_guess)
horizon_mode = st.sidebar.radio(
    "Comparison horizon",
    [
        "Until both strategies have no bond",
        "End of original term",
        "Custom months",
    ],
    index=0,
)

if horizon_mode == "Until both strategies have no bond":
    horizon_months = auto_horizon
elif horizon_mode == "End of original term":
    horizon_months = remaining_months_original
else:
    horizon_months = int(
        st.sidebar.number_input(
            "Months to compare from today",
            min_value=1,
            value=int(auto_horizon),
            step=12,
        )
    )

invest_df, interest_invest_nominal, payoff_invest = simulate_strategy(
    principal=loan_amount,
    annual_loan_rate=loan_rate,
    annual_invest_return=invest_return,
    base_payment=base_payment,
    extra_monthly=extra_monthly,
    horizon_months=horizon_months,
    strategy="invest",
    schedule_month_start=current_month,
)

prepay_df, interest_prepay_nominal, payoff_prepay = simulate_strategy(
    principal=loan_amount,
    annual_loan_rate=loan_rate,
    annual_invest_return=invest_return,
    base_payment=base_payment,
    extra_monthly=extra_monthly,
    horizon_months=horizon_months,
    strategy="prepay",
    schedule_month_start=current_month,
)

compare_nominal = pd.DataFrame({
    "MonthFromNow": invest_df["MonthFromNow"],
    "ScheduleMonth": invest_df["ScheduleMonth"],
    "LoanBalance_Invest": invest_df["LoanBalance"],
    "LoanBalance_Prepay": prepay_df["LoanBalance"],
    "Investment_Invest": invest_df["InvestmentValue"],
    "Investment_Prepay": prepay_df["InvestmentValue"],
    "NetWorth_Invest": invest_df["NetWorth"],
    "NetWorth_Prepay": prepay_df["NetWorth"],
})
compare_nominal["Advantage_Invest"] = compare_nominal["NetWorth_Invest"] - compare_nominal["NetWorth_Prepay"]

money_cols = [
    "LoanBalance_Invest",
    "LoanBalance_Prepay",
    "Investment_Invest",
    "Investment_Prepay",
    "NetWorth_Invest",
    "NetWorth_Prepay",
    "Advantage_Invest",
]

compare_display = compare_nominal.copy()
value_label = "Nominal values"
if show_real:
    for col in money_cols:
        compare_display[col] = deflate_series(compare_display[col], inflation, compare_display["MonthFromNow"])
    interest_invest_display = deflate_series(invest_df["Interest"], inflation, invest_df["MonthFromNow"]).sum()
    interest_prepay_display = deflate_series(prepay_df["Interest"], inflation, prepay_df["MonthFromNow"]).sum()
    value_label = "Inflation adjusted values in today's rand"
else:
    interest_invest_display = interest_invest_nominal
    interest_prepay_display = interest_prepay_nominal

interest_saved_display = interest_invest_display - interest_prepay_display
months_saved = (payoff_invest or horizon_months) - (payoff_prepay or horizon_months)
final_advantage = float(compare_display["Advantage_Invest"].iloc[-1])
final_invest_if_investing = float(compare_display["Investment_Invest"].iloc[-1])
final_invest_if_prepaying = float(compare_display["Investment_Prepay"].iloc[-1])
final_net_worth_invest = float(compare_display["NetWorth_Invest"].iloc[-1])
final_net_worth_prepay = float(compare_display["NetWorth_Prepay"].iloc[-1])
current_progress = current_month / original_term_months if original_term_months else 0

next_interest = loan_amount * loan_r_monthly
next_principal_normal = max(min(base_payment, loan_amount + next_interest) - next_interest, 0.0)
next_principal_prepay = max(min(base_payment + extra_monthly, loan_amount + next_interest) - next_interest, 0.0)

winner_text = "Invest the extra cash" if final_advantage > 0 else "Pay extra into the bond"
winning_amount = abs(final_advantage)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current monthly bond payment", f"R {base_payment:,.0f}")
col2.metric("Next payment interest portion", f"R {next_interest:,.0f}")
col3.metric("Bond paid off earlier if you prepay", f"{months_saved} months")
col4.metric("Projected winner at chosen horizon", f"{winner_text} by R {winning_amount:,.0f}")

stage_col1, stage_col2, stage_col3 = st.columns(3)
stage_col1.metric("Original loan amount", f"R {original_loan_amount:,.0f}")
stage_col2.metric("Current balance used", f"R {loan_amount:,.0f}")
stage_col3.metric("Position in original term", f"{current_progress:.0%}")

with st.expander("Why the bond stage matters"):
    st.markdown(
        f"""
        You are currently at schedule month **{current_month}** out of **{original_term_months}**.

        1. Your scheduled balance at this point would be about **R {scheduled_balance_now:,.0f}**.
        2. The actual balance used in the model is **R {loan_amount:,.0f}**.
        3. The model uses a current required payment of **R {base_payment:,.0f}**.
        4. On your very next payment, about **R {next_interest:,.0f}** goes to interest and about **R {next_principal_normal:,.0f}** goes to principal under the invest strategy.
        5. If you put the extra **R {extra_monthly:,.0f}** into the bond instead, that entire extra amount goes straight to principal immediately, so next month's interest starts from a lower balance.

        Earlier in the loan, the balance is larger and more future interest still remains, so prepaying has more time to snowball.
        Later in the loan, less interest remains, so the case for investing becomes relatively stronger if the expected after tax return is higher than the bond rate.
        """
    )

st.caption(current_payment_note)
if loan_stage == "Already part way through":
    st.caption(
        f"Comparison starts from today, after {current_month} payments have already happened. "
        f"The remaining months to the original contract end are {remaining_months_original}."
    )
else:
    st.caption(f"Comparison starts at month 0 of a {original_term_months} month bond.")

st.subheader("What the numbers suggest")
if final_advantage > 0:
    st.success(
        "Under these assumptions, investing the extra cash gives the higher projected net worth at the chosen horizon."
    )
elif final_advantage < 0:
    st.info(
        "Under these assumptions, paying extra into the bond and then investing the freed cash gives the higher projected net worth at the chosen horizon."
    )
else:
    st.info("Under these assumptions, both choices end at the same projected net worth.")

st.caption(value_label)

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(compare_display["MonthFromNow"], compare_display["LoanBalance_Invest"], label="Loan balance if you invest extra")
ax1.plot(compare_display["MonthFromNow"], compare_display["LoanBalance_Prepay"], label="Loan balance if you prepay")
ax1.set_title("Loan balance from your current point on the bond")
ax1.set_xlabel("Months from today")
ax1.set_ylabel("Rand")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(compare_display["MonthFromNow"], compare_display["Investment_Invest"], label="Investment value if you invest extra")
ax2.plot(compare_display["MonthFromNow"], compare_display["Investment_Prepay"], label="Investment value if you prepay first")
ax2.set_title("Investment value over time")
ax2.set_xlabel("Months from today")
ax2.set_ylabel("Rand")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(compare_display["MonthFromNow"], compare_display["Advantage_Invest"], label="Net worth advantage of investing")
ax3.axhline(0, linewidth=1)
ax3.set_title("Net worth advantage from your current amortization point")
ax3.set_xlabel("Months from today")
ax3.set_ylabel("Rand")
ax3.legend()
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

with st.expander("See comparison table"):
    st.dataframe(compare_display.round(2), use_container_width=True)

with st.expander("Key totals"):
    summary = pd.DataFrame(
        {
            "Metric": [
                "Original loan amount",
                "Current balance used",
                "Schedule month today",
                "Remaining months to original contract end",
                "Current required monthly payment",
                "Chosen comparison horizon in months",
                "Total interest if you invest extra",
                "Total interest if you prepay",
                "Interest saved by prepaying",
                "Payoff month from today if you invest extra",
                "Payoff month from today if you prepay",
                "Ending investment value if you invest extra",
                "Ending investment value if you prepay first",
                "Ending net worth if you invest extra",
                "Ending net worth if you prepay first",
                "Ending net worth advantage of investing",
            ],
            "Value": [
                original_loan_amount,
                loan_amount,
                current_month,
                remaining_months_original,
                base_payment,
                horizon_months,
                interest_invest_display,
                interest_prepay_display,
                interest_saved_display,
                payoff_invest if payoff_invest is not None else np.nan,
                payoff_prepay if payoff_prepay is not None else np.nan,
                final_invest_if_investing,
                final_invest_if_prepaying,
                final_net_worth_invest,
                final_net_worth_prepay,
                final_advantage,
            ],
        }
    )
    st.dataframe(summary, use_container_width=True)

with st.expander("How the calculations work"):
    st.markdown(
        r"""
        1. The standard scheduled monthly bond payment is:

        \[
        P = L \times \frac{r}{1 - (1 + r)^{-n}}
        \]

        where:

        * \(L\) is the loan balance at the point the payment is set
        * \(r\) is the monthly loan rate
        * \(n\) is the number of months over which that payment is meant to amortize the loan

        2. For the next payment from where you are today:

        \[
        \text{Interest}_{t} = \text{Balance}_{t-1} \times r
        \]

        \[
        \text{PrincipalPaid}_{t} = \text{Payment}_{t} - \text{Interest}_{t}
        \]

        \[
        \text{Balance}_{t} = \max(\text{Balance}_{t-1} - \text{PrincipalPaid}_{t}, 0)
        \]

        3. The invest strategy does this each month:

        * pay the required bond instalment
        * invest the extra monthly cash
        * once the bond is gone, invest the full freed monthly cash flow

        4. The prepay strategy does this each month:

        * pay the required instalment plus the extra cash into the bond
        * once the bond is gone, invest the full freed monthly cash flow

        5. Investment growth uses monthly compounding:

        \[
        \text{Investment}_{t} = \text{Investment}_{t-1} \times (1 + i) + \text{Contribution}_{t}
        \]

        where \(i\) is the monthly investment return.

        6. The fair comparison is net worth, not just the loan balance difference:

        \[
        \text{NetWorth}_{t} = \text{InvestmentValue}_{t} - \text{LoanBalance}_{t}
        \]

        7. The final chart shows:

        \[
        \text{AdvantageOfInvesting}_{t} = \text{NetWorth}_{invest,t} - \text{NetWorth}_{prepay,t}
        \]

        Positive means investing is ahead.
        Negative means prepaying is ahead.
        """
    )

st.caption(
    "Planning tool only. It ignores tax, fees, rate changes, offset accounts, access bond features, and investment volatility."
)
