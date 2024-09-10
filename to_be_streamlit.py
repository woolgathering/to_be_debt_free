import streamlit as st
import numpy as np
from plotly import graph_objects as go
from to_be_debt_free_or_not_to_be import (
    MonteCarloSim,
    GeometricBrownianMotion,
    HestonModel,
    TDistributionModel,
    JumpDiffusionModel,
    AutocorrelationModel,
    CompoundInterestModel,
)  # Import your classes
from to_be_strategic import (
    BuyAndHoldStrategy,
    HedgeFundieStrategy,
)  # Import your strategy class


# Function to compute APR based on total interest, fees, loan principal, and loan term in months
def compute_apr_from_interest(total_interest, loan_principal, loan_term_months, fees=0):
    apr = ((fees + total_interest) / loan_principal) * (12 / loan_term_months)
    return apr


# Function to compute total interest based on APR, fees, loan principal, and loan term in months
def compute_total_interest_from_apr(apr, loan_principal, loan_term_months, fees=0):
    total_interest = apr * (loan_principal * loan_term_months / 12) - fees
    return total_interest

# Function to compute APR (accounting for compound interest) from total interest, fees, loan principal, and loan term in months
def compute_apr_from_interest_compound(total_interest, loan_principal, loan_term_months, fees=0):
    total_loan = loan_principal + total_interest + fees
    # Compute monthly payment from total loan amount
    monthly_payment = total_loan / loan_term_months

    # Use the formula to solve for the monthly interest rate
    def pmt(rate, nper, pv):
        return (rate * pv) / (1 - (1 + rate) ** -nper)

    # Binary search to find the monthly interest rate that gives the correct payment
    low, high = 0, 1  # reasonable range for monthly rate
    for _ in range(100):  # iteratively narrow the search range
        r = (low + high) / 2
        computed_payment = pmt(r, loan_term_months, loan_principal)
        if computed_payment < monthly_payment:
            low = r
        else:
            high = r

    apr = r * 12  # Convert monthly rate to APR
    return apr


# Function to compute total interest (accounting for compound interest) based on APR, fees, loan principal, and loan term in months
def compute_total_interest_from_apr_compound(apr, loan_principal, loan_term_months, fees=0):
    monthly_rate = apr / 12  # Convert APR to monthly rate
    # Use the formula to compute the monthly payment
    monthly_payment = (monthly_rate * loan_principal) / (
        1 - (1 + monthly_rate) ** -loan_term_months
    )
    # Compute the total amount paid over the loan term
    total_paid = monthly_payment * loan_term_months
    # Compute total interest as the difference between total paid and loan principal
    total_interest = total_paid - loan_principal + fees
    return total_interest


# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Title of the app
st.title("Monte Carlo Simulation for Debt and Investment Decisions")

with st.expander("About this Dashboard"):
    st.markdown(
        """
    ## Motivation:
    This dashboard is designed to help users model the opportunity cost tradeoff between repaying a debt immediately or investing the funds to potentially earn a higher return over time. By simulating different financial models and investment strategies, users can explore the impact of various scenarios on their financial health and decision-making.

    ### Why use this tool?
    - **Evaluate investment scenarios**: Simulate different financial models to understand the growth or decline of an investment portfolio over time in relation to paying off a loan.
    - **Understand risk**: Visualize the uncertainty in outcomes by observing price paths and volatility.
    - **Plan debt repayment**: Assess how investments might perform relative to loan repayments and explore potential strategies for better financial outcomes.

    ### How to use it:
    1. **Input your financial details**: Fill in the principal, loan terms, and interest rates or APR.
    2. **Select a financial model**: Choose from Geometric Brownian Motion, Jump Diffusion, Heston Model, etc., and adjust the relevant parameters.
    3. **Visualize the results**: Review the Monte Carlo simulation results, including the price paths, terminal distribution, and volatility (if applicable).
    4. **Interpret the results**: Use the simulation results to make informed decisions about investments, loan repayments, and risk management.

    ---

    ### Disclaimer:
    This tool is for educational purposes only and does **not** constitute financial advice. It is based on simplified models and assumptions that may not reflect real-world conditions. Always consult with a financial advisor before making investment or debt management decisions.

    **Important Notes:**
    - The simulations **do not account for inflation** or taxes.
    - The tool assumes **constant parameters** in the loan, such as interest rates, throughout the simulation. That is, the loan rate is not variable and the loan principal is not added to or subtracted from.
    - Although founded in financial research, the results are purely hypothetical and intended to give an idea of potential outcomes under different scenarios rather than precise predictions.
    """
    )

# Sidebar for input parameters
st.sidebar.header("Simulation Settings")

# Input fields for loan details
principal = st.sidebar.number_input("Initial Investment (Principal)", value=10000)
loan_principal = st.sidebar.number_input("Loan Principal", value=10000)
loan_term_months = st.sidebar.number_input("Loan Term (Months)", value=120)

# The user provides either the total interest or the APR
compound_interest = st.sidebar.checkbox("Compound Interest?")
total_interest_provided = st.sidebar.checkbox("Provide Total Interest?")
pct = 2.154
amount = 2154.00

if total_interest_provided:
    total_interest_paid = st.sidebar.number_input(
        "Total Interest Paid on Loan",
        value=amount,
        format="%.2f",
        help="Total interest paid over the loan term",
    )
    amount = total_interest_paid
    if compound_interest:
        apr = compute_apr_from_interest_compound(
            total_interest_paid, loan_principal, loan_term_months
        )
    else:
        apr = compute_apr_from_interest(
            total_interest_paid, loan_principal, loan_term_months
        )
    st.sidebar.write(f"Computed APR: {apr*100:.2f}%")
else:
    apr = st.sidebar.number_input(
        "Loan APR %",
        value=pct,
        format="%.2f",
        help="Annual Percentage Rate in percentage",
    )
    pct = apr
    apr *= 0.01  # Convert to decimal
    if compound_interest:
        total_interest_paid = compute_total_interest_from_apr_compound(
            apr, loan_principal, loan_term_months
        )
    else:
        total_interest_paid = compute_total_interest_from_apr(
            apr, loan_principal, loan_term_months
        )
    st.sidebar.write(f"Computed Total Interest: {total_interest_paid:.2f}")

import streamlit as st
from to_be_debt_free_or_not_to_be import (
    GeometricBrownianMotion,
    HestonModel,
    JumpDiffusionModel,
    TDistributionModel,
    AutocorrelationModel,
)

# Create two columns for stock and bond model inputs
col1, col2 = st.columns(2)

# Stock model controls in the first column
with col1:
    st.header("Stock Model")
    stock_model = st.selectbox(
        "Select Stock Model",
        [
            "Geometric Brownian Motion",
            "Heston Model",
            "Jump Diffusion Model",
            "T-Distribution Model",
            "Autocorrelation Model",
        ],
    )

    if stock_model == "Geometric Brownian Motion":
        mu_stock = st.number_input("Expected Annual Return (mu) [Stock]", value=0.07)
        sigma_stock = st.number_input("Volatility (sigma) [Stock]", value=0.15)
        stock_model_instance = GeometricBrownianMotion(mu=mu_stock, sigma=sigma_stock)

    elif stock_model == "Heston Model":
        mu_stock = st.number_input("Expected Annual Return (mu) [Stock]", value=0.07)
        theta_stock = st.number_input("Long-term Variance (theta) [Stock]", value=0.04)
        kappa_stock = st.number_input(
            "Speed of Mean Reversion (kappa) [Stock]", value=2.0
        )
        sigma_v_stock = st.number_input(
            "Volatility of Volatility (sigma_v) [Stock]", value=0.3
        )
        rho_stock = st.number_input(
            "Correlation between Price and Volatility (rho) [Stock]", value=-0.7
        )
        stock_model_instance = HestonModel(
            mu=mu_stock,
            theta=theta_stock,
            kappa=kappa_stock,
            sigma_v=sigma_v_stock,
            rho=rho_stock,
        )

    elif stock_model == "Jump Diffusion Model":
        mu_stock = st.number_input("Expected Annual Return (mu) [Stock]", value=0.07)
        lambda_jump_stock = st.number_input(
            "Jump Frequency (lambda) [Stock]", value=0.1
        )
        jump_size_mean_stock = st.number_input("Jump Size Mean [Stock]", value=0.0)
        jump_size_std_stock = st.number_input("Jump Size Std Dev [Stock]", value=0.02)
        stock_model_instance = JumpDiffusionModel(
            lambda_jump=lambda_jump_stock,
            jump_size_mean=jump_size_mean_stock,
            jump_size_std=jump_size_std_stock,
            mu=mu_stock,
        )

    elif stock_model == "T-Distribution Model":
        mu_stock = st.number_input("Expected Annual Return (mu) [Stock]", value=0.07)
        sigma_stock = st.number_input("Volatility (sigma) [Stock]", value=0.15)
        degrees_of_freedom_stock = st.number_input(
            "Degrees of Freedom [Stock]", min_value=3, value=3
        )
        stock_model_instance = TDistributionModel(
            degrees_of_freedom=degrees_of_freedom_stock, mu=mu_stock, sigma=sigma_stock
        )

    elif stock_model == "Autocorrelation Model":
        mu_stock = st.number_input("Expected Annual Return (mu) [Stock]", value=0.07)
        alpha_stock = st.number_input(
            "Autocorrelation Coefficient (alpha) [Stock]", value=0.5
        )
        stock_model_instance = AutocorrelationModel(
            alpha=alpha_stock, mu=mu_stock, scale=0.1
        )

# Bond model controls in the second column
with col2:
    st.header("Bond Model")
    bond_model = st.selectbox(
        "Select Bond Model",
        [
            "Compound Interest Model",
            "Geometric Brownian Motion",
            "Heston Model",
            "Jump Diffusion Model",
            "T-Distribution Model",
            "Autocorrelation Model",
        ],
    )

    if bond_model == "Compound Interest Model":
        mu_bond = st.number_input("Expected Annual Return (mu) [Bond]", value=0.03)
        bond_model_instance = CompoundInterestModel(mu=mu_bond)

    elif bond_model == "Geometric Brownian Motion":
        mu_bond = st.number_input("Expected Annual Return (mu) [Bond]", value=0.03)
        sigma_bond = st.number_input("Volatility (sigma) [Bond]", value=0.05)
        bond_model_instance = GeometricBrownianMotion(mu=mu_bond, sigma=sigma_bond)

    elif bond_model == "Heston Model":
        mu_bond = st.number_input("Expected Annual Return (mu) [Bond]", value=0.03)
        theta_bond = st.number_input("Long-term Variance (theta) [Bond]", value=0.02)
        kappa_bond = st.number_input(
            "Speed of Mean Reversion (kappa) [Bond]", value=1.0
        )
        sigma_v_bond = st.number_input(
            "Volatility of Volatility (sigma_v) [Bond]", value=0.2
        )
        rho_bond = st.number_input(
            "Correlation between Price and Volatility (rho) [Bond]", value=-0.5
        )
        bond_model_instance = HestonModel(
            mu=mu_bond,
            theta=theta_bond,
            kappa=kappa_bond,
            sigma_v=sigma_v_bond,
            rho=rho_bond,
        )

    elif bond_model == "Jump Diffusion Model":
        mu_bond = st.number_input("Expected Annual Return (mu) [Bond]", value=0.03)
        lambda_jump_bond = st.number_input("Jump Frequency (lambda) [Bond]", value=0.1)
        jump_size_mean_bond = st.number_input("Jump Size Mean [Bond]", value=0.0)
        jump_size_std_bond = st.number_input("Jump Size Std Dev [Bond]", value=0.02)
        bond_model_instance = JumpDiffusionModel(
            lambda_jump=lambda_jump_bond,
            jump_size_mean=jump_size_mean_bond,
            jump_size_std=jump_size_std_bond,
            mu=mu_bond,
        )

    elif bond_model == "T-Distribution Model":
        mu_bond = st.number_input("Expected Annual Return (mu) [Bond]", value=0.03)
        sigma_bond = st.number_input("Volatility (sigma) [Bond]", value=0.05)
        degrees_of_freedom_bond = st.number_input(
            "Degrees of Freedom [Bond]", min_value=3, value=3
        )
        bond_model_instance = TDistributionModel(
            degrees_of_freedom=degrees_of_freedom_bond, mu=mu_bond, sigma=sigma_bond
        )

    elif bond_model == "Autocorrelation Model":
        mu_bond = st.number_input("Expected Annual Return (mu) [Bond]", value=0.03)
        alpha_bond = st.number_input(
            "Autocorrelation Coefficient (alpha) [Bond]", value=0.5
        )
        bond_model_instance = AutocorrelationModel(
            alpha=alpha_bond, mu=mu_bond, scale=0.1
        )

# Number of simulations and time steps
num_simulations = st.sidebar.slider(
    "Number of Simulations", min_value=1000, max_value=100000, value=10000
)
N = loan_term_months  # Set time steps based on loan term

strategy_name = st.sidebar.selectbox("Select Strategy", ["Buy and Hold", "HedgeFundie"])

if strategy_name == "Buy and Hold":
    stock_allocation = st.sidebar.slider(
        "Stock Allocation (%)", min_value=0, max_value=100, value=60
    )
    rebalance_frequency = st.sidebar.slider(
        "Rebalance Frequency (Months)",
        min_value=0,
        max_value=60,
        value=0,
        help="0 means no rebalancing",
    )
    bond_allocation = 100 - stock_allocation
    strategy = BuyAndHoldStrategy(
        stock_allocation=stock_allocation / 100,
        bond_allocation=bond_allocation / 100,
        rebalance_frequency=rebalance_frequency,
    )

elif strategy_name == "HedgeFundie":
    stock_allocation = st.sidebar.slider(
        "Leveraged Stock Allocation (%)", min_value=0, max_value=100, value=55
    )
    bond_allocation = 100 - stock_allocation
    leverage_stock = st.sidebar.slider(
        "Leverage Factor (Stock)", min_value=1, max_value=5, value=3
    )
    leverage_bond = st.sidebar.slider(
        "Leverage Factor (Bond)", min_value=1, max_value=5, value=3
    )
    rebalance_frequency = st.sidebar.slider(
        "Rebalance Frequency (Months)",
        min_value=0,
        max_value=60,
        value=3,
        help="0 means no rebalancing",
    )
    strategy = HedgeFundieStrategy(
        stock_allocation=stock_allocation / 100,
        bond_allocation=bond_allocation / 100,
        leverage_stock=leverage_stock,
        leverage_bond=leverage_bond,
        rebalance_frequency=rebalance_frequency,
    )

# allow drawdowns or additions
st.sidebar.header("Monthly Drawdown/Contribution")
enable_drawdown = st.sidebar.checkbox("Enable Drawdown/Contribution")
if enable_drawdown:
    contribution_type = st.sidebar.selectbox(
        "Drawdown/Contribution Type", ["Fixed Amount", "Percentage of Portfolio"]
    )
    draw_freq = st.sidebar.number_input(
        "Drawdown/Contribution Frequency (Months)", min_value=0, max_value=60, value=0
    )

    if contribution_type == "Fixed Amount":
        drawdown_amount = st.sidebar.number_input("Drawdown Amount ($)", value=100)
        is_percentage = False
    else:
        drawdown_amount = st.sidebar.number_input(
            "Drawdown Percentage (%)", value=2.0, format="%.2f"
        )
        is_percentage = True
else:
    drawdown_amount = 0
    draw_freq = 0
    is_percentage = False

# Create Monte Carlo simulation instance
mc_simulator = MonteCarloSim(
    stock_model=stock_model_instance,
    bond_model=bond_model_instance,
    principal=principal,
    loan_principal=loan_principal,
    total_interest_paid=total_interest_paid,
    loan_term_months=loan_term_months,
    apr=apr,
    num_simulations=num_simulations,
)

# Run the simulation for both stock and bond price series
stock_prices, bond_prices = mc_simulator.simulate_paths()

# Choose a strategy
# strategy = BuyAndHoldStrategy(stock_allocation=1, bond_allocation=0)

# Apply the strategy and get the strategy paths and instantaneous return
paths, instantaneous_return = mc_simulator.apply_strategy(
    strategy,
    stock_prices,
    bond_prices,
    drawdown_amount=drawdown_amount,
    drawdown_frequency=draw_freq,
    is_percentage=is_percentage,
)

# Plot results using Streamlit
st.header("Simulation Results")

# Show the primary results plot
results_fig = mc_simulator.plot_results(
    paths, instantaneous_return, show_distribution=True
)
st.plotly_chart(results_fig)

# Option to show price series
if st.checkbox("Show Price Series", value=True):
    num_series = st.slider(
        "Number of Price Series to Display", min_value=1, max_value=20, value=5
    )
    price_series_fig = mc_simulator.plot_price_series(num_series=num_series)
    st.plotly_chart(price_series_fig)

# Option to show terminal distribution
if st.checkbox("Show Terminal Distribution", value=True):
    terminal_dist_fig = mc_simulator.plot_terminal_distribution(show_principal=True)
    st.plotly_chart(terminal_dist_fig)

# Option to show volatility series (if Heston Model)
if st.checkbox("Show Volatility Series") and isinstance(model, HestonModel):
    num_vol_series = st.slider(
        "Number of Volatility Series to Display", min_value=1, max_value=20, value=5
    )
    stock_vol_series_fig = mc_simulator.plot_stock_volatility(num_series=num_vol_series)
    bond_vol_series_fig = mc_simulator.plot_bond_volatility(num_series=num_vol_series)
    st.plotly_chart(stock_vol_series_fig)
    st.plotly_chart(bond_vol_series_fig)

# Option to show utility comparison
if st.checkbox("Show Utility Comparison"):
    terminal_values = mc_simulator.price_series[-1, :]  # Get terminal values
    utility_fig = mc_simulator.plot_utility_comparison(terminal_values)
    st.plotly_chart(utility_fig)
