# Factor

This is a framework for factor mining and backtesting. The data supply is from anthor quantative tool repository named `quool`, see [quool](https://github.com/ppoak/quool) for more information.

## Usage

You can simple click download file in zip format on github, or you can use git command line if you want:

```
git clone git@github.com:ppoak/factor
```

## Factors

### Barra

all barra factors can be found in `contrib/barra.py`

1. `size`: the circulation value of the factor.

   $$
   size = \text{log}(price \times circulation)
   $$
2. `momentum`: the historical return on a stock in the past n days.

   $$
   momentum = \frac{price_{t} - price_{t-n}}{{price}_{t-n}}
   $$
3. `volatility`: the standard deviation of the daily returns of a stock over the past n days.

   $$
   volatility = \sqrt{\frac{1}{n} \sum_{i=1}^{n} return_{t-i}^2}
   $$
4. `ep`: the ratio of net profit to the total circulation value

   $$
   ep = \frac{net\_profit}{circulation}
   $$

## Change Log

version 0.1.0: add `report` module can help you backtesting any factor you want. Organized some computing modules named `barra`, `financial`, `highfreq`.

version 0.2.0: seperate task file out, and can perform backtest more flexible; reorganizing tools to factor core file.

version 0.2.1: fix problem in not returning information from backtest function

version 0.2.2: reorganize factor computing scripts to `contrib` directory. And any definitions added can be put there in form of function.

version 0.3.0: (current version) reorganize factor computing scripts to root. Add some data getting interfaces.
