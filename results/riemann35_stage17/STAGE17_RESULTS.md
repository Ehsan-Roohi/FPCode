# Stage 17: tail identifiability audit

The LP result is an explicit inner witness range on a fixed compact support, not a claim that the discrete extrema are global moment bounds. The common support contains 1024 adaptive two-population Gauss--Hermite centers with the standardized coordinate box

`[-1.557, 6.390] x [-1.258, 2.950] x [-1.891, 1.166]`,

and maximum `|c|^2/theta = 53.113`. Even under this compact restriction, non-uniqueness is large enough to reverse the sign of the self-consistent source. On the unrestricted velocity domain the sixth-moment supremum is generally unbounded when only moments through degree four are fixed; a Lasserre/SOS hierarchy, rather than this grid LP, is the route to sharp global polynomial-moment bounds.

All relative tail ranges below use `|max-min| / |(max+min)/2|`.

| Family | tail | minimum | maximum | midpoint-relative range | frozen-coefficient source pair | self-consistent source pair | max retained residual |
|---|---|---:|---:|---:|---:|---:|---:|
| atomic compact | M500 | 338.26286 | 350.2465 | 3.48% | -123.300 to -22.463 | -127.521 to 23.996 | 1.537e-14 |
| atomic compact | M600 | 1802.1489 | 1968.5735 | 8.83% | -123.311 to -22.388 | -127.532 to 24.189 | 1.017e-15 |
| atomic compact | M420 | 217.44972 | 242.6702 | 10.96% | -118.713 to -23.911 | -121.952 to 20.526 | 6.284e-12 |
| Gaussian-mollified | M500 | 338.31116 | 350.07183 | 3.42% | -122.762 to -24.029 | -126.910 to 21.083 | 2.487e-14 |
| Gaussian-mollified | M600 | 1802.6373 | 1966.0867 | 8.67% | -122.807 to -23.995 | -126.960 to 21.161 | 8.334e-14 |
| Gaussian-mollified | M420 | 217.54249 | 242.34196 | 10.79% | -118.575 to -25.292 | -121.804 to 18.019 | 3.133e-15 |

For the atomic M600 witness pair, freezing `(C, gamma, beta)` at the generating-mixture values leaves a source span of 100.923; the self-consistent 9x9 solve increases it to 151.721. Coefficient feedback contributes 33.5% of the latter span and is responsible for changing the upper-witness source from negative to positive. Regularizing the coefficient solve may remove that sign reversal, but it cannot remove the already-large direct-tail span.

Replacing every atom by an isotropic Maxwellian kernel with variance/theta = 1.0e-03 and re-solving the LP still gives an M600 range of 8.67%. The result is therefore not an artifact of singular atomic distributions.

The generating two-Gaussian value `dM400/dt = -100.484` lies inside the frozen and self-consistent witness spans. A universally exact instantaneous 35-to-M5/M6 map does not exist without additional assumptions or inherited tail memory.
