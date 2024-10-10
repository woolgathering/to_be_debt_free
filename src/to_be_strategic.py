import numpy as np


class Strategy:
    def __init__(self, stock_allocation=0.6, bond_allocation=0.4):
        """
        Initialize the Strategy class with stock and bond prices.
        :param stock_allocation: Initial percentage allocation to stocks.
        :param bond_allocation: Initial percentage allocation to bonds.
        """
        self.stock_allocation = stock_allocation
        self.bond_allocation = bond_allocation

    def apply(self, *args, **kwargs):
        """
        Apply the strategy. This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def rebalance(self, stock_prices, bond_prices, portfolio_values, time_index):
        """
        Rebalance the portfolio based on the given portfolio value at a specific time index.
        :param stock_prices: Price series for stocks.
        :param bond_prices: Price series for bonds.
        :param portfolio_value: Current portfolio value.
        :param time_index: The time index to rebalance the portfolio.
        :return: Updated stock and bond holdings after rebalancing.
        """
        # Recalculate stock and bond holdings based on the portfolio value and allocations
        stock_holdings = (
            self.stock_allocation * portfolio_values[time_index]
        ) / stock_prices[time_index]
        bond_holdings = (
            self.bond_allocation * portfolio_values[time_index]
        ) / bond_prices[time_index]

        return stock_holdings, bond_holdings

    @staticmethod
    def apply_leverage(stock_prices, bond_prices, leverage_stock, leverage_bond):
        """
        Apply leverage to the log returns of stock and bond price series and return the leveraged price paths.
        :param stock_prices: Price series for stocks.
        :param bond_prices: Price series for bonds.
        :param leverage_stock: Leverage factor for stocks.
        :param leverage_bond: Leverage factor for bonds.
        :return: Leveraged stock and bond price series.
        """
        # Calculate log returns for stocks and bonds
        stock_returns = np.diff(np.log(stock_prices), axis=0)
        bond_returns = np.diff(np.log(bond_prices), axis=0)

        # Apply leverage to the returns
        leveraged_stock_returns = leverage_stock * stock_returns
        leveraged_bond_returns = leverage_bond * bond_returns

        # Convert leveraged returns back to price series by cumulatively summing them
        leveraged_stock_prices = (
            np.exp(np.cumsum(leveraged_stock_returns, axis=0)) * stock_prices[0]
        )
        leveraged_bond_prices = (
            np.exp(np.cumsum(leveraged_bond_returns, axis=0)) * bond_prices[0]
        )

        # Add the initial price row (since np.diff removes one row)
        leveraged_stock_prices = np.vstack([stock_prices[0], leveraged_stock_prices])
        leveraged_bond_prices = np.vstack([bond_prices[0], leveraged_bond_prices])

        return leveraged_stock_prices, leveraged_bond_prices


class BuyAndHoldStrategy(Strategy):
    def __init__(
        self, stock_allocation=0.6, bond_allocation=0.4, rebalance_frequency=0
    ):
        """
        Initialize the buy and hold strategy with stock and bond allocation percentages and rebalancing.
        :param stock_allocation: Percentage of the portfolio allocated to stocks.
        :param bond_allocation: Percentage of the portfolio allocated to bonds.
        :param rebalance_frequency: Frequency of rebalancing in months (0 means no rebalancing).
        """
        self.stock_allocation = stock_allocation
        self.bond_allocation = bond_allocation
        self.rebalance_frequency = rebalance_frequency

    def apply(
        self,
        stock_prices,
        bond_prices,
        principal,
        drawdown_amount=0,
        drawdown_frequency=0,
        is_percentage=False,
    ):
        """
        Apply the buy and hold strategy with or without rebalancing and handle periodic drawdowns.
        :param stock_prices: Price series for stocks (shape: [time, simulations])
        :param bond_prices: Price series for bonds (shape: [time, simulations])
        :param principal: Initial portfolio value
        :param drawdown_amount: Amount to draw down from the portfolio (fixed dollar amount or percentage).
        :param drawdown_frequency: Frequency of drawdowns in months (0 means no drawdown).
        :param is_percentage: If True, drawdown is treated as a percentage of the portfolio value at the time.
        :return: Portfolio values over time.
        """
        # Initial holdings based on the fixed allocation (normalized around initial prices)
        stock_holdings = (self.stock_allocation * principal) / stock_prices[0]
        bond_holdings = (self.bond_allocation * principal) / bond_prices[0]

        # Initialize portfolio value array
        portfolio_values = np.zeros_like(stock_prices)
        portfolio_values[0] = principal  # Initial portfolio value at time 0

        # Loop through each time step and compute portfolio values
        for t in range(1, len(stock_prices)):
            # Update portfolio values at time t
            portfolio_values[t] = (
                stock_holdings * stock_prices[t] + bond_holdings * bond_prices[t]
            )

            # Perform rebalancing if it's time based on the rebalance frequency
            if self.rebalance_frequency > 0 and t % self.rebalance_frequency == 0:
                stock_holdings, bond_holdings = self.rebalance(
                    stock_prices, bond_prices, portfolio_values, t
                )
                # total_portfolio_value = portfolio_values[t]
                # stock_holdings = (
                #     self.stock_allocation * total_portfolio_value
                # ) / stock_prices[t]
                # bond_holdings = (
                #     self.bond_allocation * total_portfolio_value
                # ) / bond_prices[t]

            # Apply drawdown if it's time
            if drawdown_frequency > 0 and t % drawdown_frequency == 0:
                if is_percentage:
                    drawdown = portfolio_values[t] * (drawdown_amount / 100)
                else:
                    drawdown = drawdown_amount

                # Apply the drawdown proportionally to stock and bond holdings
                stock_drawdown = drawdown * (self.stock_allocation)
                bond_drawdown = drawdown * (self.bond_allocation)

                # Adjust holdings based on the drawdown
                stock_holdings = np.maximum(
                    stock_holdings - (stock_drawdown / stock_prices[t]), 0
                )
                bond_holdings = np.maximum(
                    bond_holdings - (bond_drawdown / bond_prices[t]), 0
                )

                # Recompute portfolio value after drawdown
                portfolio_values[t] = (
                    stock_holdings * stock_prices[t] + bond_holdings * bond_prices[t]
                )

        return portfolio_values


class HedgeFundieStrategy(Strategy):
    def __init__(
        self,
        stock_allocation=0.55,
        bond_allocation=0.45,
        leverage_stock=3,
        leverage_bond=3,
        rebalance_frequency=1,
    ):
        """
        Initialize the Hedgefundie strategy with stock and bond allocation percentages and leverage.
        :param stock_allocation: Percentage of the portfolio allocated to leveraged stocks.
        :param bond_allocation: Percentage of the portfolio allocated to leveraged bonds.
        :param leverage_stock: Leverage factor for the stock allocation (e.g., 3 for 3x leverage).
        :param leverage_bond: Leverage factor for the bond allocation (e.g., 3 for 3x leverage).
        :param rebalance_frequency: Frequency of rebalancing in months (0 means no rebalancing).
        """
        self.stock_allocation = stock_allocation
        self.bond_allocation = bond_allocation
        self.leverage_stock = leverage_stock
        self.leverage_bond = leverage_bond
        self.rebalance_frequency = rebalance_frequency

    def apply(
        self,
        stock_prices,
        bond_prices,
        principal,
        drawdown_amount=0,
        drawdown_frequency=0,
        is_percentage=False,
    ):
        """
        Apply the Hedgefundie strategy with leverage, rebalancing, and periodic drawdowns.
        :param stock_prices: Price series for stocks (shape: [time, simulations])
        :param bond_prices: Price series for bonds (shape: [time, simulations])
        :param principal: Initial portfolio value
        :param drawdown_amount: Amount to draw down from the portfolio (fixed dollar amount or percentage).
        :param drawdown_frequency: Frequency of drawdowns in months (0 means no drawdown).
        :param is_percentage: If True, drawdown is treated as a percentage of the portfolio value at the time.
        :return: Portfolio values over time.
        """
        # Apply leverage to the price series
        leveraged_stock_prices, leveraged_bond_prices = Strategy.apply_leverage(
            stock_prices, bond_prices, self.leverage_stock, self.leverage_bond
        )

        # Initial holdings based on the allocation and leverage
        stock_holdings = (self.stock_allocation * principal) / leveraged_stock_prices[0]
        bond_holdings = (self.bond_allocation * principal) / leveraged_bond_prices[0]

        # Initialize portfolio value array
        portfolio_values = np.zeros_like(leveraged_stock_prices)
        portfolio_values[0] = principal  # Initial portfolio value at time 0

        # Loop through each time step and compute portfolio values
        for t in range(1, len(leveraged_stock_prices)):
            # Calculate the portfolio value at time t
            portfolio_values[t] = (
                stock_holdings * leveraged_stock_prices[t]  # Stock growth
                + bond_holdings * leveraged_bond_prices[t]  # Bond growth
            )

            # Perform rebalancing if it's time based on the rebalance frequency
            if self.rebalance_frequency > 0 and t % self.rebalance_frequency == 0:
                stock_holdings, bond_holdings = self.rebalance(
                    leveraged_stock_prices, leveraged_bond_prices, portfolio_values, t
                )

            # Apply drawdown if it's time
            if drawdown_frequency > 0 and t % drawdown_frequency == 0:
                if is_percentage:
                    drawdown = portfolio_values[t] * (drawdown_amount / 100)
                else:
                    drawdown = drawdown_amount

                # Apply the drawdown proportionally to stock and bond holdings
                stock_drawdown = drawdown * (self.stock_allocation)
                bond_drawdown = drawdown * (self.bond_allocation)

                # Adjust holdings based on the drawdown
                stock_holdings = np.maximum(
                    stock_holdings - (stock_drawdown / leveraged_stock_prices[t]), 0
                )
                bond_holdings = np.maximum(
                    bond_holdings - (bond_drawdown / leveraged_bond_prices[t]), 0
                )

                # Recompute portfolio value after drawdown
                portfolio_values[t] = (
                    stock_holdings * leveraged_stock_prices[t]
                    + bond_holdings * leveraged_bond_prices[t]
                )

        return portfolio_values


class ShannonsDemonStrategy(Strategy):
    def __init__(
        self,
        stock_allocation=0.5,
        bond_allocation=0.5,
        rebalance_frequency=1,  # Rebalance every period
        volatility_window=20,  # Number of periods to calculate rolling volatility
        risk_free_rate=0.0,  # Risk-free rate (for bonds)
        volatility_estimation_method="rolling",  # 'rolling', 'ewma', or 'external'
        ewma_decay=0.94,  # Decay factor for EWMA
        external_volatility=None,  # User-provided volatility estimates
        initial_volatility=0.15,  # Initial volatility estimate (e.g., 15%)
        dynamic_rebalancing=False,  # Flag to switch between constant and dynamic rebalancing
        volatility_change_threshold=0.05,  # Threshold for dynamic rebalancing (e.g., 5%)
    ):
        # Constructor remains the same
        super().__init__(
            stock_allocation=stock_allocation,
            bond_allocation=bond_allocation,
        )
        self.rebalance_frequency = rebalance_frequency
        self.volatility_window = volatility_window
        self.risk_free_rate = risk_free_rate
        self.volatility_estimation_method = volatility_estimation_method
        self.ewma_decay = ewma_decay
        self.external_volatility = external_volatility
        self.initial_volatility = initial_volatility
        self.dynamic_rebalancing = dynamic_rebalancing
        self.volatility_change_threshold = volatility_change_threshold


    def calculate_optimal_stock_allocation(
        self, t, stock_returns, past_volatility, mean_return_estimate
    ):
        """
        Calculate the optimal allocation to the stock based on estimated volatility.
        Uses the Kelly Criterion adjusted for the risk-free rate.
        :param t: Current time index.
        :param stock_returns: Array of stock returns up to time t - 1.
        :param past_volatility: Previous volatility estimate (array).
        :param mean_return_estimate: Previous mean return estimate (array).
        :return: Optimal stock allocation, updated volatility, and updated mean return.
        """
        num_simulations = stock_returns.shape[1]

        if self.volatility_estimation_method == "rolling":
            # Use rolling window of past returns
            start_index = max(0, t - self.volatility_window)
            if t - start_index < self.volatility_window:
                # Not enough data; use initial volatility and mean return estimates
                variance = np.full(num_simulations, self.initial_volatility**2)
                mean_return = (
                    mean_return_estimate
                    if mean_return_estimate is not None
                    else np.zeros(num_simulations)
                )
            else:
                window_returns = stock_returns[start_index:t]
                mean_return = np.mean(window_returns, axis=0)
                variance = np.var(window_returns, axis=0)

        elif self.volatility_estimation_method == "ewma":
            # Update volatility estimate using EWMA
            squared_return = stock_returns[-1] ** 2  # Use the last observed return

            if past_volatility is None:
                # Initialize volatility estimate with initial volatility
                variance = np.full(num_simulations, self.initial_volatility**2)
            else:
                # EWMA update
                variance = (
                    self.ewma_decay * past_volatility
                    + (1 - self.ewma_decay) * squared_return
                )

            # Update mean return estimate using EWMA
            if mean_return_estimate is None:
                mean_return = np.full(num_simulations, 0.0)
            else:
                mean_return = (
                    self.ewma_decay * mean_return_estimate
                    + (1 - self.ewma_decay)
                    * stock_returns[-1]  # Use the last observed return
                )

        elif self.volatility_estimation_method == "external":
            if self.external_volatility is None:
                raise ValueError(
                    "External volatility estimates must be provided for 'external' method."
                )
            variance = self.external_volatility[t - 1] ** 2
            mean_return = (
                mean_return_estimate
                if mean_return_estimate is not None
                else np.zeros(num_simulations)
            )
        else:
            raise ValueError("Invalid volatility estimation method.")

        # Avoid division by zero
        variance = np.where(variance == 0, 1e-8, variance)

        # Calculate excess return over the risk-free rate per period
        try:
            excess_return = mean_return - (self.risk_free_rate / self.rebalance_frequency)
        except ZeroDivisionError:
            excess_return = mean_return - self.risk_free_rate

        # Calculate optimal allocation using Kelly Criterion
        optimal_allocation = excess_return / variance

        # Ensure allocations are within [0, 1]
        optimal_allocation = np.clip(optimal_allocation, 0, 1)

        return optimal_allocation, variance, mean_return

    def apply(
        self,
        stock_prices,
        bond_prices,
        principal,
        drawdown_amount=0,
        drawdown_frequency=0,
        is_percentage=False,
    ):
        # Apply method with corrections
        num_periods, num_simulations = stock_prices.shape

        # Calculate stock returns
        stock_returns = np.diff(np.log(stock_prices), axis=0)

        # Initialize allocations per simulation
        stock_allocations = np.full(num_simulations, self.stock_allocation)
        bond_allocations = np.full(num_simulations, self.bond_allocation)

        # Initialize holdings per simulation
        stock_holdings = (stock_allocations * principal) / stock_prices[0]
        bond_holdings = (bond_allocations * principal) / bond_prices[0]

        # Initialize portfolio values array
        portfolio_values = np.zeros_like(stock_prices)
        portfolio_values[0] = principal

        # Initialize past volatility and mean return estimates
        past_volatility = None  # Arrays of shape (num_simulations,)
        mean_return_estimate = None

        # Initialize last rebalanced volatility and allocations
        last_rebalance_volatility = np.full(num_simulations, self.initial_volatility**2)
        last_rebalance_allocations = stock_allocations.copy()

        for t in range(1, num_periods):
            # Update portfolio values at time t
            portfolio_values[t] = (
                stock_holdings * stock_prices[t] + bond_holdings * bond_prices[t]
            )

            # Update volatility and mean return estimates
            if t >= 2:  # Need at least 1 return observation
                optimal_stock_allocation, past_volatility, mean_return_estimate = (
                    self.calculate_optimal_stock_allocation(
                        t, stock_returns[: t - 1], past_volatility, mean_return_estimate
                    )
                )
            else:
                # Use initial estimates
                optimal_stock_allocation = stock_allocations
                past_volatility = np.full(num_simulations, self.initial_volatility**2)
                mean_return_estimate = np.full(num_simulations, 0.0)

            # Determine if rebalancing is needed
            if self.dynamic_rebalancing:
                # Compute relative change in volatility per simulation
                volatility_change = (
                    np.abs(past_volatility - last_rebalance_volatility)
                    / last_rebalance_volatility
                )
                # Determine which simulations need rebalancing
                rebalance_now = volatility_change >= self.volatility_change_threshold
            else:
                # Rebalance at fixed intervals
                rebalance_now = np.zeros(num_simulations, dtype=bool)
                if self.rebalance_frequency > 0 and t % self.rebalance_frequency == 0:
                    rebalance_now[:] = True

            if np.any(rebalance_now):
                # Update allocations for simulations needing rebalancing
                stock_allocations[rebalance_now] = optimal_stock_allocation[
                    rebalance_now
                ]
                bond_allocations[rebalance_now] = 1 - stock_allocations[rebalance_now]

                # Rebalance holdings for simulations needing rebalancing
                stock_holdings, bond_holdings = self.rebalance_per_simulation(
                    stock_prices[t],
                    bond_prices[t],
                    portfolio_values[t],
                    stock_allocations,
                    bond_allocations,
                    stock_holdings,
                    bond_holdings,
                    rebalance_now,
                )

                # Update last rebalanced volatility and allocations
                last_rebalance_volatility[rebalance_now] = past_volatility[
                    rebalance_now
                ]
                last_rebalance_allocations[rebalance_now] = stock_allocations[
                    rebalance_now
                ]

            # Apply drawdown if it's time
            if drawdown_frequency > 0 and t % drawdown_frequency == 0:
                if is_percentage:
                    drawdown = portfolio_values[t] * (drawdown_amount / 100)
                else:
                    drawdown = drawdown_amount

                # Apply the drawdown proportionally to stock and bond holdings
                stock_drawdown = drawdown * stock_allocations
                bond_drawdown = drawdown * bond_allocations

                # Adjust holdings based on the drawdown
                stock_holdings = np.maximum(
                    stock_holdings - (stock_drawdown / stock_prices[t]), 0
                )
                bond_holdings = np.maximum(
                    bond_holdings - (bond_drawdown / bond_prices[t]), 0
                )

                # Recompute portfolio value after drawdown
                portfolio_values[t] = (
                    stock_holdings * stock_prices[t] + bond_holdings * bond_prices[t]
                )

        return portfolio_values

    def rebalance_per_simulation(
        self,
        stock_price_t,
        bond_price_t,
        portfolio_value_t,
        stock_allocations,
        bond_allocations,
        stock_holdings,
        bond_holdings,
        rebalance_now,
    ):
        """
        Rebalance holdings per simulation based on current allocations and portfolio values.
        :param stock_price_t: Stock prices at time t (array).
        :param bond_price_t: Bond prices at time t (array).
        :param portfolio_value_t: Portfolio values at time t (array).
        :param stock_allocations: Current stock allocations (array).
        :param bond_allocations: Current bond allocations (array).
        :param stock_holdings: Current stock holdings (array).
        :param bond_holdings: Current bond holdings (array).
        :param rebalance_now: Boolean array indicating which simulations need rebalancing.
        :return: Updated stock holdings and bond holdings (arrays).
        """
        # Initialize holdings arrays
        new_stock_holdings = np.copy(stock_holdings)
        new_bond_holdings = np.copy(bond_holdings)

        # Rebalance holdings for simulations needing rebalancing
        new_stock_holdings[rebalance_now] = (
            stock_allocations[rebalance_now]
            * portfolio_value_t[rebalance_now]
            / stock_price_t[rebalance_now]
        )
        new_bond_holdings[rebalance_now] = (
            bond_allocations[rebalance_now]
            * portfolio_value_t[rebalance_now]
            / bond_price_t[rebalance_now]
        )

        return new_stock_holdings, new_bond_holdings
