import dataforge as forge
import factormonitor as fm
from .factor import Momentum

quotesday = forge.AssetTable("/home/data/quotes-day")
df = quotesday.read("close, adjfactor", start="20200101", stop="20231031")
momentum = Momentum(df["close"] * df["adjfactor"])
print(momentum.compute())