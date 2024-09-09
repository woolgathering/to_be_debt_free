import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp


def symlog_transform(values, lin_thresh=1e3):
    """
    Apply a sym-log transformation to handle both positive and negative values.
    :param values: Array of values to transform
    :param lin_thresh: Threshold for switching between linear and log scales
    :return: Transformed values
    """
    return np.sign(values) * np.log1p(np.abs(values / lin_thresh)) * lin_thresh


class CompoundInterestModel:
    def __init__(self, rate_of_return=0.05, **kwargs):
        """
        Compound Interest model for price paths.
        :param rate_of_return: Constant rate of return (interest rate per period)
        """
        self.rate_of_return = rate_of_return  # Constant rate of return

    def generate(self, num_simulations, N, dt, initial_price=1, **kwargs):
        """
        Generate price paths using Compound Interest.
        :param num_simulations: Number of simulation paths
        :param N: Number of time steps
        :param dt: Time increment
        :param initial_price: The initial price of the asset
        :return: A 2D array of simulated paths with shape (N+1, num_simulations)
        """
        # Initialize an array for price paths
        paths = np.zeros((N + 1, num_simulations))
        paths[0] = initial_price  # Set the initial price at t=0

        # Calculate compound interest for each time step
        for t in range(1, N + 1):
            paths[t] = (
                paths[0] * (1 + self.rate_of_return * dt) ** t
            )  # Compound interest formula

        return paths


class GeometricBrownianMotion:
    def __init__(self, mu=0.07, sigma=0.15, **kwargs):
        """
        Geometric Brownian Motion model for price paths.
        :param mu: Expected return (drift)
        :param sigma: Volatility (standard deviation of returns)
        """
        self.mu = mu  # Drift or expected return
        self.sigma = sigma  # Volatility

    def generate(self, num_simulations, N, dt, initial_price=1, **kwargs):
        """
        Generate price paths using Geometric Brownian Motion.
        :param num_simulations: Number of simulation paths
        :param N: Number of time steps
        :param dt: Time increment
        :param initial_price: The initial price of the asset
        :return: A 2D array of simulated paths with shape (N+1, num_simulations)
        """
        # Initialize an array for price paths
        paths = np.zeros((N + 1, num_simulations))
        paths[0] = initial_price  # Set the initial price at t=0

        # Generate standard normal random variables for each time step and simulation
        Z = np.random.standard_normal((N, num_simulations))

        # Generate the price paths
        for t in range(1, N + 1):
            # Update paths using the GBM formula
            paths[t] = paths[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt
                + self.sigma * np.sqrt(dt) * Z[t - 1]
            )

        return paths


class TDistributionModel:
    def __init__(self, degrees_of_freedom=3, mu=0.07, sigma=0.15, **kwargs):
        """
        T-distribution model for price paths with scaling.
        :param degrees_of_freedom: Degrees of freedom for the t-distribution
        :param mu: Expected return (drift)
        :param sigma: Volatility (standard deviation of returns)
        """
        self.degrees_of_freedom = degrees_of_freedom
        self.mu = mu
        self.sigma = sigma

    def generate(self, num_simulations, N, dt, initial_price=1, **kwargs):
        """
        Generate t-distributed price paths over N time steps.
        :param num_simulations: Number of simulation paths
        :param N: Number of time steps
        :param dt: Time increment (used if needed for scaling)
        :param initial_price: The initial price of the asset
        :return: A 2D array of simulated paths with shape (N+1, num_simulations)
        """
        # Generate t-distributed random variables with shape (N, num_simulations)
        Z = np.random.standard_t(self.degrees_of_freedom, (N, num_simulations))

        # Scale the t-distribution values to match the volatility
        Z = Z * (
            self.sigma
            / np.sqrt(self.degrees_of_freedom / (self.degrees_of_freedom - 2))
        )

        # Initialize an array for price paths, starting with the initial price
        paths = np.zeros((N + 1, num_simulations))
        paths[0] = initial_price  # Set the initial price

        # Generate paths using a geometric Brownian motion-like approach, but with t-distributed increments
        for t in range(1, N + 1):
            paths[t] = paths[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + np.sqrt(dt) * Z[t - 1]
            )

        return paths


class JumpDiffusionModel:
    def __init__(
        self,
        lambda_jump=0.1,
        jump_size_mean=0,
        jump_size_std=0.02,
        mu=0.07,
        sigma=0.15,
        **kwargs,
    ):
        """
        Jump diffusion model with both diffusion (GBM) and jump components.

        :param lambda_jump: Jump intensity (average rate of jumps per time step)
        :param jump_size_mean: Mean of the jump size (for the normal distribution)
        :param jump_size_std: Standard deviation of the jump size
        :param mu: Expected annual return (drift)
        :param sigma: Volatility (diffusion component)
        """
        self.lambda_jump = lambda_jump  # Jump intensity
        self.jump_size_mean = jump_size_mean  # Mean of jump sizes
        self.jump_size_std = jump_size_std  # Std dev of jump sizes
        self.mu = mu  # Drift term (expected return)
        self.sigma = sigma  # Volatility term

    def generate(self, num_simulations, N, dt, initial_price=1):
        """
        Generate price paths using the jump diffusion model over N time steps.

        :param num_simulations: Number of simulation paths
        :param N: Number of time steps
        :param dt: Time increment (in years)
        :param initial_price: Initial asset price
        :return: A 2D array of simulated paths with shape (N+1, num_simulations)
        """
        # Initialize arrays for storing price paths
        paths = np.zeros((N + 1, num_simulations))
        paths[0] = initial_price  # Set initial price

        # Standard Brownian motion component
        Z = np.random.normal(0, 1, (N, num_simulations))

        # Poisson process for jumps
        jumps = np.random.poisson(self.lambda_jump * dt, (N, num_simulations))

        # Generate jump sizes
        jump_magnitudes = np.random.normal(
            self.jump_size_mean, self.jump_size_std, (N, num_simulations)
        )

        # Loop over each time step to compute the path
        for t in range(1, N + 1):
            # Continuous diffusion component (GBM)
            paths[t] = paths[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt
                + self.sigma * np.sqrt(dt) * Z[t - 1]
            )

            # Add the jumps
            paths[t] += paths[t - 1] * (jumps[t - 1] * jump_magnitudes[t - 1])

        return paths


class AutocorrelationModel:
    def __init__(self, alpha=0.5, mu=0.10, scale=0.5, **kwargs):
        self.alpha = alpha
        self.scale = scale  # Scaling factor to avoid large jumps
        self.mu = mu  # Increased drift to counteract downward trend

    def generate(self, num_simulations, N, dt, initial_price=1):
        """
        Generate autocorrelated price paths over N time steps using an AR(1) model.
        :param num_simulations: Number of simulation paths
        :param N: Number of time steps
        :param dt: Time increment (used if needed for scaling)
        :param initial_price: The initial price (starting value)
        :return: A 2D array of simulated paths with shape (N+1, num_simulations)
        """
        Z = np.zeros((N, num_simulations))

        # Initialize the first return value
        Z[0] = np.random.standard_normal(num_simulations) * self.scale

        # Generate autocorrelated returns using AR(1) model
        for t in range(1, N):
            Z[t] = (
                self.alpha * Z[t - 1]
                + np.random.standard_normal(num_simulations) * self.scale
            )

        # Initialize an array for price paths, starting with the initial price
        paths = np.zeros((N + 1, num_simulations))
        paths[0] = initial_price  # Set initial price

        # Generate paths using the autocorrelated returns
        for t in range(1, N + 1):
            # Apply positive drift and reduced noise bias
            paths[t] = paths[t - 1] * np.exp((self.mu) * dt + Z[t - 1] * np.sqrt(dt))

        return paths


class HestonModel:
    def __init__(self, mu=0.07, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5):
        """
        Heston model for stochastic volatility.
        :param mu: Expected return (drift)
        :param v0: Initial variance (volatility squared)
        :param kappa: Rate of mean reversion for volatility
        :param theta: Long-term variance level
        :param sigma_v: Volatility of volatility
        :param rho: Correlation between price and volatility
        """
        self.mu = mu
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def generate(self, num_simulations, N, dt, **kwargs):
        """
        Generate price paths using the Heston model.
        :param num_simulations: Number of simulation paths
        :param N: Number of time steps
        :param dt: Time increment
        :return: Simulated price paths and volatility paths
        """
        # Initialize arrays for prices and variances
        S = np.zeros((N + 1, num_simulations))
        v = np.zeros((N + 1, num_simulations))
        S[0] = 1  # Start with price 1 (we'll multiply by the principal later)
        v[0] = self.v0  # Start with initial variance

        # Correlated random numbers for price and variance
        Z1 = np.random.standard_normal((N, num_simulations))
        Z2 = np.random.standard_normal((N, num_simulations))

        # Apply correlation to the Wiener processes
        W_S = Z1
        W_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        for t in range(1, N + 1):
            # Update variance (volatility squared) using the Heston model equation
            v[t] = (
                v[t - 1]
                + self.kappa * (self.theta - v[t - 1]) * dt
                + self.sigma_v * np.sqrt(v[t - 1]) * np.sqrt(dt) * W_v[t - 1]
            )
            v[t] = np.maximum(v[t], 0)  # Variance can't be negative

            # Update price using the generated variance
            S[t] = S[t - 1] * np.exp(
                (self.mu - 0.5 * v[t - 1]) * dt
                + np.sqrt(v[t - 1]) * np.sqrt(dt) * W_S[t - 1]
            )

        return S, v  # Return both price paths and volatility paths


class MonteCarloSim:
    def __init__(
        self,
        # price_model,
        stock_model,
        bond_model,
        principal,
        loan_principal,
        total_interest_paid,
        loan_term_months,
        apr,
        mu=0.07,
        sigma=0.15,
        num_simulations=10000,
        N=None,
    ):
        # self.price_model = price_model  # Price model for generating random values
        self.stock_model = stock_model
        self.bond_model = bond_model
        self.principal = principal  # Initial investment
        self.loan_principal = loan_principal  # Loan principal
        self.total_interest_paid = (
            total_interest_paid  # Total interest paid on the loan
        )
        self.loan_term_months = loan_term_months  # Loan term in months
        self.apr = apr  # Average APR
        self.mu = mu  # Expected annual return (drift)
        self.sigma = sigma  # Volatility (standard deviation of returns)
        self.N = (
            N if N is not None else loan_term_months
        )  # Number of time steps (defaults to monthly)
        self.dt = (
            self.loan_term_months / 12
        ) / self.N  # Time increment (in years per step)
        self.num_simulations = num_simulations  # Number of simulation paths

        self.price_series = None

    def amortization_schedule(self):
        """
        Compute the amortization schedule to track cumulative interest paid over time.
        """
        monthly_interest_rate = self.apr / 12
        remaining_balance = self.loan_principal
        cumulative_interest = np.zeros(
            self.N + 1
        )  # Array to hold cumulative interest values

        # Monthly payment calculation based on APR
        monthly_payment = (self.loan_principal * monthly_interest_rate) / (
            1 - (1 + monthly_interest_rate) ** -self.loan_term_months
        )

        for t in range(1, self.N + 1):
            interest_payment = remaining_balance * monthly_interest_rate
            principal_payment = monthly_payment - interest_payment
            remaining_balance -= principal_payment
            cumulative_interest[t] = cumulative_interest[t - 1] + interest_payment

        return cumulative_interest

    def simulate_paths(self):
        """
        Simulate the price paths for stocks and bonds using the selected price model.
        """
        # Initialize arrays for stock and bond paths, normalized around 1
        stock_paths = np.zeros((self.N + 1, self.num_simulations))
        bond_paths = np.zeros((self.N + 1, self.num_simulations))

        # Generate stock and bond price paths (normalized around 1)
        stock_results = self.stock_model.generate(
            self.num_simulations, N=self.N, dt=self.dt, initial_price=1
        )  # Stock model
        bond_results = self.bond_model.generate(
            self.num_simulations, N=self.N, dt=self.dt, initial_price=1
        )  # Bond model

        if isinstance(stock_results, tuple):
            # For models like Heston that return both price and volatility
            stock_paths, stock_volatilities = stock_results
            self.stock_volatility_series = (
                stock_volatilities  # Save volatility series for inspection
            )
        else:
            # For other models that return only price paths
            stock_paths = stock_results

        if isinstance(bond_results, tuple):
            # For models like Heston that return both price and volatility
            bond_paths, bond_volatilities = bond_results
            self.bond_volatility_series = (
                bond_volatilities  # Save volatility series for inspection
            )
        else:
            # For other models that return only price paths
            bond_paths = bond_results

        # Store stock and bond paths for later inspection
        self.stock_paths = stock_paths
        self.bond_paths = bond_paths

        return stock_paths, bond_paths

    def apply_strategy(self, strategy, stock_prices, bond_prices, **kwargs):
        """
        Apply the given strategy to the simulated paths and return the strategy-dependent investment values and instantaneous returns.
        :param strategy: The strategy class to apply (e.g., BuyAndHoldStrategy)
        :param stock_prices: Price paths of stocks generated by the simulation
        :param bond_prices: Price paths of bonds generated by the simulation
        :param kwargs: Additional arguments required for the strategy
        :return: strategy_paths, instantaneous_return
        """
        # Apply the strategy to compute portfolio values (strategy-dependent paths)
        strategy_paths = strategy.apply(
            stock_prices, bond_prices, self.principal, **kwargs
        )

        # Compute cumulative interest using the amortization schedule
        cumulative_interest = self.amortization_schedule().reshape(-1, 1)

        # Compute instantaneous return: strategy_paths - cumulative interest
        instantaneous_return = strategy_paths - cumulative_interest
        self.price_series = (
            strategy_paths  # Update terminal values for utility comparison
        )

        return strategy_paths, instantaneous_return

    def calculate_var(self, confidence_level=0.95):
        """
        Calculate the Value-at-Risk (VaR) at the given confidence level.
        :param confidence_level: Confidence level for VaR (e.g., 0.95 for 95% VaR)
        :return: VaR value
        """
        # Get the terminal values (last row of price series)
        terminal_values = self.price_series[-1, :]

        # Calculate the VaR at the specified confidence level
        var = np.percentile(terminal_values, (1 - confidence_level) * 100)
        return var

    def calculate_cvar(self, confidence_level=0.95):
        """
        Calculate the Conditional Value-at-Risk (CVaR) at the given confidence level.
        :param confidence_level: Confidence level for CVaR (e.g., 0.95 for 95% CVaR)
        :return: CVaR value
        """
        # Get the terminal values (last row of price series)
        terminal_values = self.price_series[-1, :]

        # Calculate the VaR threshold
        var = self.calculate_var(confidence_level)

        # Calculate the average of the values below the VaR threshold (CVaR)
        cvar = terminal_values[terminal_values <= var].mean()
        return cvar

    def quadratic_utility(self, terminal_values, max_loss_tolerance=0.10):
        """
        Calculate quadratic utility for the given terminal values, using max loss tolerance as a natural parameter.
        :param terminal_values: Array of terminal values (wealth)
        :param max_loss_tolerance: Maximum tolerable loss as a percentage (default 10%)
        :return: Quadratic utility values
        """
        # Translate max loss tolerance to alpha
        alpha = 2 / max_loss_tolerance
        return terminal_values - (alpha / 2) * (terminal_values**2)

    def logarithmic_utility(self, terminal_values, wealth_sensitivity=1):
        """
        Calculate logarithmic utility for the given terminal values, with an adjustable sensitivity factor.
        :param terminal_values: Array of terminal values (wealth)
        :param wealth_sensitivity: Sensitivity factor to adjust the perception of wealth changes (default 1)
        :return: Logarithmic utility values (log(W))
        """
        # Ensure no negative or zero values in terminal values (log is undefined for <= 0)
        return wealth_sensitivity * np.log(np.maximum(terminal_values, 1e-10))

    def expected_utility(self, utility_values):
        """
        Calculate the expected utility (average utility).
        :param utility_values: Array of utility values for each simulation
        :return: The average utility value
        """
        return np.mean(utility_values)

    def plot_results(
        self, paths, instantaneous_return, show_distribution=False, bins=50
    ):
        """
        Plot the simulated investment paths and instantaneous return using Plotly,
        with an optional flipped terminal distribution plot on the right.
        """
        # Generate the time axis (in months)
        time = np.linspace(0, self.loan_term_months, self.N + 1)

        # Calculate median and percentile bounds for both paths and instantaneous return
        path_median = np.median(paths, axis=1)
        path_5th = np.percentile(paths, 5, axis=1)
        path_95th = np.percentile(paths, 95, axis=1)

        inst_return_median = np.median(instantaneous_return, axis=1)
        inst_return_5th = np.percentile(instantaneous_return, 5, axis=1)
        inst_return_95th = np.percentile(instantaneous_return, 95, axis=1)

        # Create hover text for each path and return (show value and value minus principal)
        path_median_hover_text = [
            f"Value: ${v:.2f}<br>Less Principal: ${(v - self.loan_principal):.2f}"
            for v in path_median
        ]
        path_5th_hover_text = [
            f"5th Percentile: ${v:.2f}<br>Less Principal: ${(v - self.loan_principal):.2f}"
            for v in path_5th
        ]
        path_95th_hover_text = [
            f"95th Percentile: ${v:.2f}<br>Less Principal: ${(v - self.loan_principal):.2f}"
            for v in path_95th
        ]

        inst_return_median_hover_text = [
            f"Instant Return: ${r:.2f}<br>Less Principal: ${(r - self.loan_principal):.2f}"
            for r in inst_return_median
        ]
        inst_return_5th_hover_text = [
            f"5th Percentile Instant: ${r:.2f}<br>Less Principal: ${(r - self.loan_principal):.2f}"
            for r in inst_return_5th
        ]
        inst_return_95th_hover_text = [
            f"95th Percentile Instant: ${r:.2f}<br>Less Principal: ${(r - self.loan_principal):.2f}"
            for r in inst_return_95th
        ]

        # Create a subplot layout if showing distribution, otherwise a single figure
        if show_distribution:
            fig = sp.make_subplots(
                rows=1,
                cols=2,
                column_widths=[0.75, 0.25],
                shared_yaxes=True,
                horizontal_spacing=0.05,
                subplot_titles=("Monte Carlo Paths", "Terminal Distribution"),
            )
        else:
            fig = go.Figure()

        # Helper function to add traces with or without row/col based on show_distribution
        def add_trace_to_fig(trace, row=None, col=None):
            if show_distribution:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)

        # Add traces for the investment paths with hover info
        add_trace_to_fig(
            go.Scatter(
                x=time,
                y=path_median,
                mode="lines",
                name="Median Investment Path",
                hovertext=path_median_hover_text,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
        add_trace_to_fig(
            go.Scatter(
                x=time,
                y=path_5th,
                fill=None,
                mode="lines",
                name="5th Percentile",
                hovertext=path_5th_hover_text,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
        add_trace_to_fig(
            go.Scatter(
                x=time,
                y=path_95th,
                fill="tonexty",
                mode="lines",
                name="95th Percentile",
                hovertext=path_95th_hover_text,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

        # Add traces for instantaneous return with hover info
        add_trace_to_fig(
            go.Scatter(
                x=time,
                y=inst_return_median,
                mode="lines",
                name="Median Instantaneous Return",
                hovertext=inst_return_median_hover_text,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
        add_trace_to_fig(
            go.Scatter(
                x=time,
                y=inst_return_5th,
                fill=None,
                mode="lines",
                name="5th Percentile Instantaneous",
                hovertext=inst_return_5th_hover_text,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
        add_trace_to_fig(
            go.Scatter(
                x=time,
                y=inst_return_95th,
                fill="tonexty",
                mode="lines",
                name="95th Percentile Instantaneous",
                hovertext=inst_return_95th_hover_text,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

        # Add a flat repayment line equal to the loan principal
        loan_principal_hover_text = [
            f"Loan Principal: ${self.loan_principal:.2f}"
        ] * len(time)
        add_trace_to_fig(
            go.Scatter(
                x=time,
                y=[self.loan_principal] * len(time),
                mode="lines",
                name="Loan Principal",
                hovertext=loan_principal_hover_text,
                hoverinfo="text",
                line=dict(dash="dash", color="orange"),
            ),
            row=1,
            col=1,
        )

        # Add the terminal distribution plot if show_distribution is True
        if show_distribution:
            # Extract terminal values (last row of price series)
            terminal_values = self.price_series[-1, :]

            # Calculate the 5th and 95th percentiles
            terminal_5th = np.percentile(terminal_values, 5)
            terminal_95th = np.percentile(terminal_values, 95)

            # Filter values between the 5th and 95th percentiles
            terminal_values = terminal_values[
                (terminal_values >= terminal_5th) & (terminal_values <= terminal_95th)
            ]

            # Create a histogram for the terminal values, oriented to the right (horizontal)
            hist = np.histogram(terminal_values, bins=bins)
            bin_centers = 0.5 * (hist[1][:-1] + hist[1][1:])

            # Add the histogram as a bar plot
            fig.add_trace(
                go.Bar(
                    x=hist[0],
                    y=bin_centers,
                    orientation="h",
                    name="Terminal Distribution",
                    marker_color="gray",
                ),
                row=1,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title="Monte Carlo Simulation of Investment vs. Loan Interest",
            xaxis_title="Months",
            yaxis_title="Value ($)",
            legend_title="Paths",
            template="plotly_dark",
        )

        # Show the figure
        return fig

    def plot_price_series(self, num_series=10):
        """
        Plot the raw price series from the Monte Carlo simulations.
        """
        if self.price_series is None or self.price_series.size == 0:
            raise ValueError("No portfolio data available. Run simulate_paths() first.")

        fig = go.Figure()

        # Plot the first `num_series` price series
        for i in range(min(num_series, self.num_simulations)):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.N + 1),
                    y=self.price_series[:, i],
                    mode="lines",
                    name=f"Simulation {i+1}",
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Portfolio Value of {num_series} Simulations",
            xaxis_title="Time (Steps)",
            yaxis_title="Price",
            legend_title="Simulations",
            template="plotly_dark",
        )

        return fig

    def plot_terminal_distribution(self, bins=50, show_principal=True):
        """
        Plot the distribution of terminal outcomes from the Monte Carlo simulations,
        with an optional vertical line showing the initial principal and a cumulative distribution (hover shows percentage).
        """
        if self.price_series is None:
            raise ValueError(
                "No price series data available. Run simulate_paths() first."
            )

        # Extract terminal values (last row of price series)
        terminal_values = self.price_series[-1, :]

        # Create the histogram
        hist, bin_edges = np.histogram(terminal_values, bins=bins)

        # Calculate the cumulative distribution (still scale it to the max histogram height)
        cdf = np.cumsum(hist) / np.sum(hist)  # Keep cumulative between 0 and 1
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Create a plotly figure
        fig = go.Figure()

        # Add the histogram for the terminal values
        fig.add_trace(
            go.Bar(
                x=bin_centers, y=hist, name="Terminal Distribution", marker_color="gray"
            )
        )

        # Add the cumulative distribution as a line plot (hover shows percentage)
        cumulative_hover_text = [f"{v * 100:.2f}%" for v in cdf]
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=cdf * max(hist),
                mode="lines",
                name="Cumulative Distribution (%)",
                line=dict(color="blue", width=2, dash="dot"),
                hovertext=cumulative_hover_text,
                hoverinfo="text",
            )
        )

        # Optionally add a vertical line at the loan principal
        if show_principal:
            fig.add_shape(
                type="line",
                x0=self.loan_principal,
                x1=self.loan_principal,
                y0=0,
                y1=max(hist),
                line=dict(color="red", width=3, dash="dash"),
                name="Principal",
            )

        # Update layout
        fig.update_layout(
            title="Distribution and Cumulative Distribution of Terminal Outcomes",
            xaxis_title="Terminal Value",
            yaxis_title="Frequency",
            template="plotly_dark",
        )

        # Show the figure
        return fig

    def plot_stock_volatility(self, num_series=10):
        """
        Plot the volatility paths from the Heston model simulations.
        """
        if (
            self.stock_volatility_series is None
            or self.stock_volatility_series.size == 0
        ):
            raise ValueError(
                "No volatility series data available. Run simulate_paths() first."
            )

        fig = go.Figure()

        # Plot the first `num_series` volatility series
        for i in range(min(num_series, self.num_simulations)):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.N + 1),
                    y=self.stock_volatility_series[:, i],
                    mode="lines",
                    name=f"Simulation {i+1}",
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Volatility Series of Stocks of {num_series} Simulations (Heston Model)",
            xaxis_title="Time (Steps)",
            yaxis_title="Volatility (Variance)",
            legend_title="Simulations",
            template="plotly_dark",
        )

        return fig

    def plot_bond_volatility(self, num_series=10):
        """
        Plot the volatility paths from the Heston model simulations.
        """
        if self.bond_volatility_series is None or self.bond_volatility_series.size == 0:
            raise ValueError(
                "No volatility series data available. Run simulate_paths() first."
            )

        fig = go.Figure()

        # Plot the first `num_series` volatility series
        for i in range(min(num_series, self.num_simulations)):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.N + 1),
                    y=self.bond_volatility_series[:, i],
                    mode="lines",
                    name=f"Simulation {i+1}",
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Volatility Series of Bond of {num_series} Simulations (Heston Model)",
            xaxis_title="Time (Steps)",
            yaxis_title="Volatility (Variance)",
            legend_title="Simulations",
            template="plotly_dark",
        )

        return fig

    def plot_var_cvar(self, confidence_level=0.95, bins=50):
        """
        Plot the distribution of terminal outcomes with VaR and CVaR highlighted.
        """
        # Get the terminal values (last row of price series)
        terminal_values = self.price_series[-1, :]

        # Calculate VaR and CVaR
        var = self.calculate_var(confidence_level)
        cvar = self.calculate_cvar(confidence_level)

        # Create a histogram for the terminal values
        hist = np.histogram(terminal_values, bins=bins)
        bin_centers = 0.5 * (hist[1][:-1] + hist[1][1:])

        fig = go.Figure()

        # Plot the histogram
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist[0],
                name="Terminal Distribution",
                marker_color="gray",
            )
        )

        # Add a vertical line for VaR
        fig.add_shape(
            type="line",
            x0=var,
            x1=var,
            y0=0,
            y1=max(hist[0]),
            line=dict(color="red", width=3, dash="dash"),
            name="VaR",
        )

        # Add a vertical line for CVaR
        fig.add_shape(
            type="line",
            x0=cvar,
            x1=cvar,
            y0=0,
            y1=max(hist[0]),
            line=dict(color="blue", width=3, dash="dot"),
            name="CVaR",
        )

        # Update layout
        fig.update_layout(
            title=f"VaR and CVaR at {int(confidence_level * 100)}% Confidence Level",
            xaxis_title="Terminal Value",
            yaxis_title="Frequency",
            template="plotly_dark",
        )

        return fig

    def plot_utility_comparison(
        self, terminal_values, alpha=0.0001, use_symlog=True, lin_thresh=1e3
    ):
        """
        Plot the comparison of quadratic and logarithmic utilities for the terminal values.
        Option to toggle between symlog and linear scales.
        :param terminal_values: Array of terminal wealth values
        :param alpha: Risk aversion parameter for quadratic utility
        :param use_symlog: Whether to apply the symlog transformation (default: True)
        :param lin_thresh: Threshold at which to switch between linear and log behavior (used for symlog)
        """
        quadratic_utilities = self.quadratic_utility(terminal_values, alpha)
        logarithmic_utilities = self.logarithmic_utility(terminal_values)

        if use_symlog:
            # Apply symlog transformation for both utilities
            quadratic_utilities = symlog_transform(quadratic_utilities, lin_thresh)
            logarithmic_utilities = symlog_transform(logarithmic_utilities, lin_thresh)

        fig = go.Figure()

        # Plot quadratic utility
        fig.add_trace(
            go.Scatter(
                x=terminal_values,
                y=quadratic_utilities,
                mode="markers",
                name="Quadratic Utility",
                line=dict(color="blue"),
            )
        )
        # Plot logarithmic utility
        fig.add_trace(
            go.Scatter(
                x=terminal_values,
                y=logarithmic_utilities,
                mode="markers",
                name="Logarithmic Utility",
                line=dict(color="green"),
            )
        )

        # Update layout: Use linear or symlog scale based on the flag
        fig.update_layout(
            title="Quadratic vs Logarithmic Utility Comparison",
            xaxis_title="Terminal Wealth",
            yaxis_title="Utility",
            template="plotly_dark",
            yaxis=dict(
                type="log" if not use_symlog else None
            ),  # Log for linear scale, symlog otherwise
        )

        return fig


# principal = 35880.34  # Initial investment
# loan_principal = 35880.34  # Loan principal
# total_interest_paid = 7497  # Total interest paid on the loan
# loan_term_months = 120  # Loan term in months (10 years)
# apr = 0.02154  # Average APR

# # Create a Jump Diffusion model
# # jump_model = JumpDiffusionModel(
# #     lambda_jump=0.2, jump_size_mean=0.01, jump_size_std=0.03
# # )

# # Realistic parameters for equity markets
# mu = 0.07  # Expected annual return (7%)
# theta = 0.04  # Long-run variance (20% volatility)
# kappa = 2  # Speed of mean reversion
# sigma = 0.3  # Volatility of volatility
# rho = -0.7  # Correlation between price and variance

# # Create the Heston price model with these parameters
# # price_model = HestonModel(mu=mu, theta=theta, kappa=kappa, sigma_v=sigma, rho=rho)

# price_model = NormalModel(mean=mu, std_dev=sigma)

# # Create a MonteCarloSim instance with the jump model
# mc_simulator = MonteCarloSim(
#     principal=principal,
#     loan_principal=loan_principal,
#     total_interest_paid=total_interest_paid,
#     loan_term_months=loan_term_months,
#     apr=apr,
#     price_model=price_model,
# )

# # Simulate paths
# paths, instantaneous_return = mc_simulator.simulate_paths()

# # Plot the results
# mc_simulator.plot_results(paths, instantaneous_return).show()

# # Plot the price series
# mc_simulator.plot_price_series(50).show()

# # Plot the terminal distribution
# mc_simulator.plot_terminal_distribution().show()

# # Plot the volatility series (if available)
# if mc_simulator.volatility_series is not None:
#     mc_simulator.plot_volatility(num_series=10).show()

# # Calculate and print 95% VaR and CVaR
# var_95 = mc_simulator.calculate_var(confidence_level=0.95)
# cvar_95 = mc_simulator.calculate_cvar(confidence_level=0.95)

# # print(f"95% VaR: {var_95}")
# # print(f"95% CVaR: {cvar_95}")
# # Plot VaR and CVaR on the distribution of terminal values
# mc_simulator.plot_var_cvar(confidence_level=0.95).show()

# # Get terminal values (final wealth)
# terminal_values = mc_simulator.price_series[-1, :]

# # Calculate quadratic and logarithmic utility
# # quadratic_utilities = mc_simulator.quadratic_utility(terminal_values, alpha=0.1)
# # logarithmic_utilities = mc_simulator.logarithmic_utility(terminal_values)

# # # Calculate expected utility for both
# # expected_quadratic_utility = mc_simulator.expected_utility(quadratic_utilities)
# # expected_logarithmic_utility = mc_simulator.expected_utility(logarithmic_utilities)

# # print(f"Expected Quadratic Utility: {expected_quadratic_utility}")
# # print(f"Expected Logarithmic Utility: {expected_logarithmic_utility}")

# # mc_simulator.plot_utility_comparison(terminal_values)

# # Quadratic utility with a 5% max tolerable loss
# quadratic_utilities = mc_simulator.quadratic_utility(
#     terminal_values, max_loss_tolerance=0.5
# )

# # Logarithmic utility with a wealth sensitivity factor of 2
# logarithmic_utilities = mc_simulator.logarithmic_utility(
#     terminal_values, wealth_sensitivity=2
# )

# # Plot the utility comparison
# mc_simulator.plot_utility_comparison(
#     terminal_values,
# ).show()

