import datetime
import dataforge as forge
from .factor import Momentum

quotesday = forge.AssetTable("/home/data/quotes-day")
df = quotesday.read("close, adjfactor", start="20230801")
# ohlcv = quotesday.read("open, high, low, close, volume", start="20200101", stop="20231031")

momentum = Momentum(df["close"] * df["adjfactor"])
factor = momentum.compute(span=20)
momentum.save("momentum_span20")
