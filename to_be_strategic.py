import numpy as np

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

    def rebalance(self, stock_prices, bond_prices, portfolio_value, time_index):
        """
        Rebalance the portfolio based on the given portfolio value at a specific time index.
        :param stock_prices: Price series for stocks.
        :param bond_prices: Price series for bonds.
        :param portfolio_value: Current portfolio value.
        :param time_index: The time index to rebalance the portfolio.
        :return: Updated stock and bond holdings after rebalancing.
        """
        # Recalculate stock and bond holdings based on the portfolio value and allocations
        stock_holdings = (self.stock_allocation * portfolio_value) / stock_prices[
            time_index
        ]
        bond_holdings = (self.bond_allocation * portfolio_value) / bond_prices[
            time_index
        ]

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
                total_portfolio_value = portfolio_values[t]
                stock_holdings = (
                    self.stock_allocation * total_portfolio_value
                ) / stock_prices[t]
                bond_holdings = (
                    self.bond_allocation * total_portfolio_value
                ) / bond_prices[t]

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
                total_portfolio_value = portfolio_values[t]
                stock_holdings = (
                    self.stock_allocation * total_portfolio_value
                ) / leveraged_stock_prices[t]
                bond_holdings = (
                    self.bond_allocation * total_portfolio_value
                ) / leveraged_bond_prices[t]

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
