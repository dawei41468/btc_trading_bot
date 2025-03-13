from config import POSITION_SIZE_PERCENT, STOP_LOSS_PERCENT, DAILY_LOSS_LIMIT

class RiskManager:
    def __init__(self, initial_capital):
        """Initialize with starting capital."""
        self.initial_capital = initial_capital
        self.daily_loss = 0

    def calculate_position_size(self, capital, price):
        """Calculate position size based on a percentage of capital."""
        max_position_value = capital * POSITION_SIZE_PERCENT
        return max_position_value / price

    def set_stop_loss(self, entry_price):
        """Set stop-loss price based on entry price."""
        return entry_price * (1 - STOP_LOSS_PERCENT)

    def check_daily_loss(self, current_portfolio_value):
        """Check if daily loss exceeds the limit."""
        loss = (self.initial_capital - current_portfolio_value) / self.initial_capital
        self.daily_loss = max(self.daily_loss, loss)
        return self.daily_loss <= DAILY_LOSS_LIMIT