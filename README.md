# Factor

This is a framework for factor mining and backtesting. The data supply is from anthor quantative tool repository named `quool`, see [quool](https://github.com/ppoak/quool) for more information.

## Usage

You can simple click download file in zip format on github, or you can use git command line if you want:

```
git clone git@github.com:ppoak/factor
```

## Change Log

version 0.1.0: add `report` module can help you backtesting any factor you want. Organized some computing modules named `barra`, `financial`, `highfreq`.

version 0.2.0: seperate task file out, and can perform backtest more flexible; reorganizing tools to factor core file.

version 0.2.1: fix problem in not returning information from backtest function

version 0.2.2: (current version) reorganize factor computing scripts to `contrib` directory. And any definitions added can be put there in form of function.
