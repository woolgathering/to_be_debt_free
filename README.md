# Debt Repayment vs. Investment Opportunity Dashboard

## Motivation:

This dashboard was designed to help me model the opportunity cost tradeoff between repaying a debt immediately or investing the funds to potentially earn a higher return over time and figured others could find it helpful. By simulating different financial models and investment strategies, users can explore the impact of various scenarios on their financial health and decision-making.

### Why use this tool?
- **Evaluate investment scenarios**: Simulate different financial models to understand the growth or decline of an investment portfolio over time in relation to paying off a loan.
- **Understand risk**: Visualize the uncertainty in outcomes by observing price paths and volatility.
- **Plan debt repayment**: Assess how investments might perform relative to loan repayments and explore potential strategies for better financial outcomes.

## How to use it:

1. **Input your financial details**: Fill in the principal, loan terms, and interest rates or APR.
2. **Select a financial model**: Choose from Geometric Brownian Motion, Jump Diffusion, Heston Model, etc., and adjust the relevant parameters.
3. **Visualize the results**: Review the Monte Carlo simulation results, including the price paths, terminal distribution, and volatility (if applicable).
4. **Interpret the results**: Use the simulation results to make informed decisions about investments, loan repayments, and risk management.

## Features

The dashboard provides the following features:
- **Simulation of various financial models**: Choose from models like Geometric Brownian Motion, Jump Diffusion, and Heston model to model investment behavior.
- **Monte Carlo simulations**: View potential outcomes for investments and loans with the aid of Monte Carlo simulations.
- **Visualization of price paths and risk metrics**: Visualize simulated price paths, terminal distributions, and risk metrics like Value at Risk (VaR) and Conditional Value at Risk (CVaR).
- **Comparison of investment vs. debt repayment**: See how investment returns could compare to debt repayment strategies over time.

## Requirements

Make sure the following Python libraries are installed to run the dashboard:

- `numpy`
- `plotly`
- `streamlit`

## Running the Dashboard

1.	Clone the repository or download the files.
2.	Run the dashboard using Streamlit:
    ```bash
    streamlit run to_be_streamlit.py
    ```

3.	The dashboard will open in your default web browser, where you can start inputting your financial details and viewing the results.

## Disclaimer

This tool is for educational purposes only and does not constitute financial advice. It is based on simplified models and assumptions that may not reflect real-world conditions. Always consult with a financial advisor before making investment or debt management decisions.

### Important Notes:

- The simulations do not account for inflation or taxes.
- The tool assumes constant parameters in the loan, such as interest rates, throughout the simulation. That is, the loan rate is not variable and the loan principal is not added to or subtracted from.
- Although founded in financial research, the results are purely hypothetical and intended to give an idea of potential outcomes under different scenarios rather than precise predictions.

## Authors
- Jacob Sundstrom