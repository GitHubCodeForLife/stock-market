class PriceROC:
    # Documentations: https://www.investopedia.com/terms/p/pricerateofchange.asp#:~:text=The%20Price%20Rate%20of%20Change%20(ROC)%20is%20a%20momentum%2D,certain%20number%20of%20periods%20ago.
    # STEP to caculate the ROC of a price
    # 1. Choose n value often is 200
    # 2. Find the most recent period's closing price
    # 3. Find the n-periods ago's closing price
    # 4. Plug the prices from 2 and 3 into the formula
    # Formula: (Close - n-periods ago's closing price) / n-periods ago's closing price * 100
    # 5. As each period ends, calculate the new ROC value.

    @staticmethod
    def caculateROC(current, previous):
        return (current - previous) / previous * 100

    @staticmethod
    def caculateROC_list(close_price, n):
        roc_list = [0]
        for i in range(n, len(close_price)):
            roc_list.append(PriceROC.caculateROC(
                close_price[i], close_price[i - n]))
        return roc_list

    # @staticmethod
    def caculateROC_list(close_price):
        roc_list = [0]
        for i in range(1, len(close_price)):
            roc_list.append(PriceROC.caculateROC(
                close_price[i], close_price[i - 1]))
        return roc_list
