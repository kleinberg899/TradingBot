class Portfolio:
    def __init__(self, starting_balance, list_of_stocks):

        self.balance = starting_balance
        self.list_of_stocks = list_of_stocks
        self.stock_dict = {entry: 0 for entry in self.list_of_stocks}

    def increase_stock_amount(self, stock, amount):

        current_amount = self.stock_dict[stock]

        if stock not in self.list_of_stocks:
            return False

        elif current_amount + amount < 0:
            return False

        else:
            self.stock_dict[stock] += amount
            return True

    def increase_balance(self, amount):

        if amount + self.balance < 0:
            return False
        else:
            self.balance += amount
            return True

    def get_portfolio(self):
        return self.stock_dict

    def get_balance(self):
        return self.balance

