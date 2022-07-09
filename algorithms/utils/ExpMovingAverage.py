import numpy as np

class ExpMovingAverage:
    def calculateEMA(current_price, previous_ema):
        multiplier = 2 / 11
        return (current_price - previous_ema)*multiplier + previous_ema
    
    def calculateEMA_list(s, n=10):
        s = np.array(s)
        ema_arr = [0]*(n-1)
        j = n-1
        #get n sma first and calculate the next n period ema
        sma = sum(s[:n]) / n
        multiplier = 2 / float(1 + n)
        ema_arr.append(sma)
        #now calculate the rest of the values
        for i in s[n:]:
            cur_ema = (i - ema_arr[j]) * multiplier + ema_arr[j]
            j = j + 1
            ema_arr.append(cur_ema)
        return ema_arr