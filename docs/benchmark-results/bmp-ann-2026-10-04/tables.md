# Complete measured comparison table

Negative latency change is faster. All values are microseconds.

Binary and replay medians/p95 pool per-query samples from the two runs
of each variant. Criterion values are the median of two run medians;
each run rotates queries across iterations. They are not request-tail
latency distributions, so p95 is omitted. ABBA order throughout.

## arm/binary-b

| Case                         | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| ---------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| 256 bits / probes 1 / k10    |     6.15 |      4.46 | -27.4% |         8.42 |          6.00 |
| 256 bits / probes 1 / k100   |    20.21 |     16.17 | -20.0% |        28.92 |         24.08 |
| 256 bits / probes 32 / k10   |   141.63 |     98.06 | -30.8% |       152.67 |        115.42 |
| 256 bits / probes 32 / k100  |   177.42 |    137.10 | -22.7% |       199.17 |        152.75 |
| 256 bits / probes 8 / k10    |    40.00 |     26.27 | -34.3% |        46.29 |         29.08 |
| 256 bits / probes 8 / k100   |    64.90 |     52.44 | -19.2% |        76.88 |         64.96 |
| 2560 bits / probes 1 / k10   |    22.79 |     19.33 | -15.2% |        29.58 |         22.62 |
| 2560 bits / probes 1 / k100  |    40.31 |     31.87 | -20.9% |        54.71 |         40.42 |
| 2560 bits / probes 32 / k10  |   718.58 |    671.29 |  -6.6% |       909.83 |        834.62 |
| 2560 bits / probes 32 / k100 |   747.77 |    707.88 |  -5.3% |       855.04 |        822.67 |
| 2560 bits / probes 8 / k10   |   181.02 |    158.81 | -12.3% |       211.79 |        180.83 |
| 2560 bits / probes 8 / k100  |   206.31 |    187.69 |  -9.0% |       281.71 |        214.50 |

## arm/binary-final

| Case                         | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| ---------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| 256 bits / probes 1 / k10    |     5.23 |      4.92 |  -6.0% |         8.33 |          8.29 |
| 256 bits / probes 1 / k100   |    20.65 |     19.27 |  -6.7% |        26.21 |         28.54 |
| 256 bits / probes 32 / k10   |   142.54 |    104.52 | -26.7% |       153.08 |        115.25 |
| 256 bits / probes 32 / k100  |   177.52 |    144.92 | -18.4% |       202.50 |        164.83 |
| 256 bits / probes 8 / k10    |    36.42 |     28.71 | -21.2% |        41.75 |         38.12 |
| 256 bits / probes 8 / k100   |    58.56 |     53.75 |  -8.2% |        86.00 |         71.12 |
| 2560 bits / probes 1 / k10   |    21.79 |     20.10 |  -7.7% |        25.92 |         25.38 |
| 2560 bits / probes 1 / k100  |    36.92 |     32.71 | -11.4% |        46.12 |         41.25 |
| 2560 bits / probes 32 / k10  |   721.44 |    699.19 |  -3.1% |       955.54 |        797.92 |
| 2560 bits / probes 32 / k100 |   754.73 |    725.77 |  -3.8% |       932.67 |        806.50 |
| 2560 bits / probes 8 / k10   |   180.12 |    162.83 |  -9.6% |       210.21 |        182.50 |
| 2560 bits / probes 8 / k100  |   206.90 |    195.04 |  -5.7% |       220.25 |        225.75 |

## arm/bmp-a

| Case                                    | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| --------------------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| wide_q60_default_gamma_top10/BMP/200000 |  5135.03 |   5379.79 |  +4.8% |            — |             — |
| wide_q60_exhaustive_top10/BMP/200000    |  5083.78 |   5417.68 |  +6.6% |            — |             — |
| wide_q60_exhaustive_top100/BMP/200000   |  5186.58 |   5290.90 |  +2.0% |            — |             — |
| wide_q60_gamma_quarter_top10/BMP/200000 |  1200.25 |   1351.34 | +12.6% |            — |             — |
| wide_q60_uniform_top10/BMP/200000       |  2210.63 |   3003.43 | +35.9% |            — |             — |
| wide_q60_uniform_top100/BMP/200000      |  2251.40 |   2980.61 | +32.4% |            — |             — |

## arm/bmp-b

| Case                                    | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| --------------------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| wide_q60_default_gamma_top10/BMP/200000 |  5113.31 |   5154.11 |  +0.8% |            — |             — |
| wide_q60_exhaustive_top10/BMP/200000    |  5122.13 |   5050.99 |  -1.4% |            — |             — |
| wide_q60_exhaustive_top100/BMP/200000   |  5038.60 |   5203.71 |  +3.3% |            — |             — |
| wide_q60_gamma_quarter_top10/BMP/200000 |  1222.77 |   1236.46 |  +1.1% |            — |             — |
| wide_q60_uniform_top10/BMP/200000       |  2254.09 |   2262.44 |  +0.4% |            — |             — |
| wide_q60_uniform_top100/BMP/200000      |  2270.26 |   2265.97 |  -0.2% |            — |             — |

## arm/bmp-c

| Case                                               | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| -------------------------------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| block256 / wide_q60_default_gamma_top10/BMP/100000 |  2491.20 |   2528.37 |  +1.5% |            — |             — |
| block256 / wide_q60_exhaustive_top10/BMP/100000    |  2397.66 |   2410.40 |  +0.5% |            — |             — |
| block256 / wide_q60_exhaustive_top100/BMP/100000   |  2419.18 |   2449.58 |  +1.3% |            — |             — |
| block256 / wide_q60_gamma_quarter_top10/BMP/100000 |   526.91 |    512.17 |  -2.8% |            — |             — |
| block256 / wide_q60_uniform_top10/BMP/100000       |   958.92 |    953.02 |  -0.6% |            — |             — |
| block256 / wide_q60_uniform_top100/BMP/100000      |   898.26 |    951.85 |  +6.0% |            — |             — |
| block32 / wide_q12_default_gamma_top10/BMP/100000  |  2516.58 |   2687.38 |  +6.8% |            — |             — |
| block32 / wide_q12_exhaustive_top10/BMP/100000     |  2736.77 |   2622.52 |  -4.2% |            — |             — |
| block32 / wide_q12_exhaustive_top100/BMP/100000    |  3158.34 |   2931.81 |  -7.2% |            — |             — |
| block32 / wide_q12_gamma_quarter_top10/BMP/100000  |   717.23 |    697.59 |  -2.7% |            — |             — |
| block32 / wide_q12_uniform_top10/BMP/100000        |   452.58 |    400.63 | -11.5% |            — |             — |
| block32 / wide_q12_uniform_top100/BMP/100000       |   609.20 |    570.77 |  -6.3% |            — |             — |
| block32 / wide_q2_default_gamma_top10/BMP/100000   |   249.09 |    237.72 |  -4.6% |            — |             — |
| block32 / wide_q2_exhaustive_top10/BMP/100000      |   246.83 |    240.45 |  -2.6% |            — |             — |
| block32 / wide_q2_exhaustive_top100/BMP/100000     |   339.63 |    328.90 |  -3.2% |            — |             — |
| block32 / wide_q2_gamma_quarter_top10/BMP/100000   |   105.87 |    103.23 |  -2.5% |            — |             — |
| block32 / wide_q2_uniform_top10/BMP/100000         |    30.94 |     31.65 |  +2.3% |            — |             — |
| block32 / wide_q2_uniform_top100/BMP/100000        |    62.70 |     60.37 |  -3.7% |            — |             — |
| block32 / wide_q60_default_gamma_top10/BMP/100000  |  6391.95 |   6465.49 |  +1.2% |            — |             — |
| block32 / wide_q60_exhaustive_top10/BMP/100000     |  6067.89 |   6152.66 |  +1.4% |            — |             — |
| block32 / wide_q60_exhaustive_top100/BMP/100000    |  6796.00 |   6454.29 |  -5.0% |            — |             — |
| block32 / wide_q60_gamma_quarter_top10/BMP/100000  |  1557.96 |   1606.79 |  +3.1% |            — |             — |
| block32 / wide_q60_uniform_top10/BMP/100000        |  2587.99 |   2756.56 |  +6.5% |            — |             — |
| block32 / wide_q60_uniform_top100/BMP/100000       |  3084.76 |   3110.22 |  +0.8% |            — |             — |

## arm/bmp-d

| Case                                               | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| -------------------------------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| block256 / wide_q60_default_gamma_top10/BMP/100000 |  2431.60 |   2662.79 |  +9.5% |            — |             — |
| block256 / wide_q60_exhaustive_top10/BMP/100000    |  2414.06 |   2621.91 |  +8.6% |            — |             — |
| block256 / wide_q60_exhaustive_top100/BMP/100000   |  2570.51 |   2681.72 |  +4.3% |            — |             — |
| block256 / wide_q60_gamma_quarter_top10/BMP/100000 |   531.85 |    551.90 |  +3.8% |            — |             — |
| block256 / wide_q60_uniform_top10/BMP/100000       |   880.39 |    983.43 | +11.7% |            — |             — |
| block256 / wide_q60_uniform_top100/BMP/100000      |   985.32 |    988.69 |  +0.3% |            — |             — |
| block32 / wide_q12_default_gamma_top10/BMP/100000  |  2690.17 |   2626.89 |  -2.4% |            — |             — |
| block32 / wide_q12_exhaustive_top10/BMP/100000     |  2634.36 |   2797.84 |  +6.2% |            — |             — |
| block32 / wide_q12_exhaustive_top100/BMP/100000    |  2995.76 |   3101.59 |  +3.5% |            — |             — |
| block32 / wide_q12_gamma_quarter_top10/BMP/100000  |   680.16 |    723.99 |  +6.4% |            — |             — |
| block32 / wide_q12_uniform_top10/BMP/100000        |   427.87 |    377.67 | -11.7% |            — |             — |
| block32 / wide_q12_uniform_top100/BMP/100000       |   578.74 |    588.44 |  +1.7% |            — |             — |
| block32 / wide_q2_default_gamma_top10/BMP/100000   |   256.97 |    239.95 |  -6.6% |            — |             — |
| block32 / wide_q2_exhaustive_top10/BMP/100000      |   258.17 |    237.79 |  -7.9% |            — |             — |
| block32 / wide_q2_exhaustive_top100/BMP/100000     |   332.93 |    330.61 |  -0.7% |            — |             — |
| block32 / wide_q2_gamma_quarter_top10/BMP/100000   |   106.31 |    102.34 |  -3.7% |            — |             — |
| block32 / wide_q2_uniform_top10/BMP/100000         |    32.54 |     30.89 |  -5.1% |            — |             — |
| block32 / wide_q2_uniform_top100/BMP/100000        |    67.58 |     61.37 |  -9.2% |            — |             — |
| block32 / wide_q60_default_gamma_top10/BMP/100000  |  6106.64 |   6388.94 |  +4.6% |            — |             — |
| block32 / wide_q60_exhaustive_top10/BMP/100000     |  6337.70 |   6437.45 |  +1.6% |            — |             — |
| block32 / wide_q60_exhaustive_top100/BMP/100000    |  6075.05 |   6292.78 |  +3.6% |            — |             — |
| block32 / wide_q60_gamma_quarter_top10/BMP/100000  |  1512.32 |   1535.08 |  +1.5% |            — |             — |
| block32 / wide_q60_uniform_top10/BMP/100000        |  2749.04 |   2805.52 |  +2.1% |            — |             — |
| block32 / wide_q60_uniform_top100/BMP/100000       |  3186.97 |   3104.88 |  -2.6% |            — |             — |

## arm/bmp-e

| Case                                               | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| -------------------------------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| block256 / wide_q60_default_gamma_top10/BMP/100000 |  2477.06 |   2457.22 |  -0.8% |            — |             — |
| block256 / wide_q60_exhaustive_top10/BMP/100000    |  2434.62 |   2532.33 |  +4.0% |            — |             — |
| block256 / wide_q60_exhaustive_top100/BMP/100000   |  2420.00 |   2423.02 |  +0.1% |            — |             — |
| block256 / wide_q60_gamma_quarter_top10/BMP/100000 |   518.79 |    500.57 |  -3.5% |            — |             — |
| block256 / wide_q60_uniform_top10/BMP/100000       |   960.08 |    982.47 |  +2.3% |            — |             — |
| block256 / wide_q60_uniform_top100/BMP/100000      |  1057.74 |    958.13 |  -9.4% |            — |             — |
| block32 / wide_q2_default_gamma_top10/BMP/100000   |   253.20 |    242.77 |  -4.1% |            — |             — |
| block32 / wide_q2_exhaustive_top10/BMP/100000      |   253.31 |    247.73 |  -2.2% |            — |             — |
| block32 / wide_q2_exhaustive_top100/BMP/100000     |   332.46 |    334.45 |  +0.6% |            — |             — |
| block32 / wide_q2_gamma_quarter_top10/BMP/100000   |   110.21 |    104.24 |  -5.4% |            — |             — |
| block32 / wide_q2_uniform_top10/BMP/100000         |    30.85 |     32.50 |  +5.3% |            — |             — |
| block32 / wide_q2_uniform_top100/BMP/100000        |    62.10 |     63.80 |  +2.7% |            — |             — |
| block32 / wide_q4_default_gamma_top10/BMP/100000   |   591.26 |    566.60 |  -4.2% |            — |             — |
| block32 / wide_q4_exhaustive_top10/BMP/100000      |   574.26 |    575.04 |  +0.1% |            — |             — |
| block32 / wide_q4_exhaustive_top100/BMP/100000     |   813.89 |    805.19 |  -1.1% |            — |             — |
| block32 / wide_q4_gamma_quarter_top10/BMP/100000   |   226.54 |    205.51 |  -9.3% |            — |             — |
| block32 / wide_q4_uniform_top10/BMP/100000         |    75.19 |     78.45 |  +4.3% |            — |             — |
| block32 / wide_q4_uniform_top100/BMP/100000        |   128.99 |    123.87 |  -4.0% |            — |             — |
| block32 / wide_q60_default_gamma_top10/BMP/100000  |  6326.66 |   6412.27 |  +1.4% |            — |             — |
| block32 / wide_q60_exhaustive_top10/BMP/100000     |  6023.22 |   6253.02 |  +3.8% |            — |             — |
| block32 / wide_q60_exhaustive_top100/BMP/100000    |  6304.24 |   6266.57 |  -0.6% |            — |             — |
| block32 / wide_q60_gamma_quarter_top10/BMP/100000  |  1567.59 |   1620.39 |  +3.4% |            — |             — |
| block32 / wide_q60_uniform_top10/BMP/100000        |  2704.41 |   2798.20 |  +3.5% |            — |             — |
| block32 / wide_q60_uniform_top100/BMP/100000       |  3027.80 |   3180.63 |  +5.0% |            — |             — |
| block32 / wide_q8_default_gamma_top10/BMP/100000   |  1360.41 |   1317.31 |  -3.2% |            — |             — |
| block32 / wide_q8_exhaustive_top10/BMP/100000      |  1410.73 |   1246.08 | -11.7% |            — |             — |
| block32 / wide_q8_exhaustive_top100/BMP/100000     |  1702.73 |   1714.14 |  +0.7% |            — |             — |
| block32 / wide_q8_gamma_quarter_top10/BMP/100000   |   439.38 |    443.05 |  +0.8% |            — |             — |
| block32 / wide_q8_uniform_top10/BMP/100000         |   180.19 |    177.54 |  -1.5% |            — |             — |
| block32 / wide_q8_uniform_top100/BMP/100000        |   263.04 |    247.49 |  -5.9% |            — |             — |

## arm/scann

| Case                            | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| ------------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| scann_fast_scan/score_block/32  |     2.50 |      2.54 |  +1.8% |            — |             — |
| scann_fast_scan/score_block/384 |    27.73 |     27.20 |  -1.9% |            — |             — |
| scann_fast_scan/score_block/96  |     6.88 |      6.92 |  +0.6% |            — |             — |

## arm/tq

| Case              | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| ----------------- | -------: | --------: | -----: | -----------: | ------------: |
| tq_lut16_scan/768 |   122.38 |    123.43 |  +0.9% |            — |             — |

## x86/binary-c

| Case                         | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| ---------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| 256 bits / probes 1 / k10    |    15.02 |     14.41 |  -4.1% |        16.54 |         16.01 |
| 256 bits / probes 1 / k100   |    41.82 |     43.06 |  +3.0% |        45.25 |         48.48 |
| 256 bits / probes 32 / k10   |   397.44 |    342.70 | -13.8% |       408.77 |        355.86 |
| 256 bits / probes 32 / k100  |   453.12 |    413.79 |  -8.7% |       467.16 |        429.25 |
| 256 bits / probes 8 / k10    |   100.73 |     88.74 | -11.9% |       111.74 |        100.56 |
| 256 bits / probes 8 / k100   |   143.47 |    140.23 |  -2.3% |       164.21 |        153.13 |
| 2560 bits / probes 1 / k10   |    57.37 |     55.55 |  -3.2% |        65.95 |         64.39 |
| 2560 bits / probes 1 / k100  |    84.92 |     84.27 |  -0.8% |        93.89 |         92.65 |
| 2560 bits / probes 32 / k10  |  3102.15 |   3110.75 |  +0.3% |      3525.80 |       3459.77 |
| 2560 bits / probes 32 / k100 |  3074.25 |   3194.43 |  +3.9% |      3604.79 |       3585.33 |
| 2560 bits / probes 8 / k10   |   503.52 |    487.51 |  -3.2% |       558.65 |        501.32 |
| 2560 bits / probes 8 / k100  |   561.17 |    538.34 |  -4.1% |       588.47 |        554.41 |

## x86/binary-final

| Case                         | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| ---------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| 256 bits / probes 1 / k10    |    14.85 |     14.00 |  -5.7% |        16.20 |         15.63 |
| 256 bits / probes 1 / k100   |    41.59 |     42.47 |  +2.1% |        47.35 |         47.88 |
| 256 bits / probes 32 / k10   |   395.12 |    338.72 | -14.3% |       408.22 |        350.55 |
| 256 bits / probes 32 / k100  |   452.39 |    408.44 |  -9.7% |       472.88 |        425.99 |
| 256 bits / probes 8 / k10    |   100.19 |     88.46 | -11.7% |       111.48 |         99.57 |
| 256 bits / probes 8 / k100   |   143.22 |    138.44 |  -3.3% |       153.32 |        152.21 |
| 2560 bits / probes 1 / k10   |    56.88 |     55.50 |  -2.4% |       182.90 |         63.95 |
| 2560 bits / probes 1 / k100  |    84.06 |     84.16 |  +0.1% |        95.42 |         92.71 |
| 2560 bits / probes 32 / k10  |  2871.76 |   2871.03 |  -0.0% |      3323.53 |       3275.47 |
| 2560 bits / probes 32 / k100 |  3003.22 |   2895.75 |  -3.6% |      3358.19 |       3353.81 |
| 2560 bits / probes 8 / k10   |   497.10 |    479.81 |  -3.5% |       514.79 |        521.72 |
| 2560 bits / probes 8 / k100  |   539.69 |    561.04 |  +4.0% |       558.20 |        588.24 |

## x86/bmp-c-real

| Case             | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| ---------------- | -------: | --------: | -----: | -----------: | ------------: |
| k10 / gamma 0    |  7429.39 |   7468.50 |  +0.5% |     21651.52 |      22130.08 |
| k10 / gamma 100  |  1230.83 |   1234.62 |  +0.3% |      2152.18 |       2158.36 |
| k100 / gamma 0   | 13925.00 |  13999.36 |  +0.5% |     35542.06 |      36164.89 |
| k100 / gamma 100 |  1887.92 |   1940.14 |  +2.8% |      2841.76 |       2908.77 |

## x86/scann

| Case                            | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |
| ------------------------------- | -------: | --------: | -----: | -----------: | ------------: |
| scann_fast_scan/score_block/32  |     5.11 |      5.12 |  +0.2% |            — |             — |
| scann_fast_scan/score_block/384 |    45.62 |     45.42 |  -0.5% |            — |             — |
| scann_fast_scan/score_block/96  |    12.34 |     12.42 |  +0.6% |            — |             — |

## Process peak RSS

Ranges across runs; includes fixture generation, oracle, mappings and
runtime overhead. This is not incremental query heap usage.

| Campaign         | Baseline MiB | Candidate MiB |
| ---------------- | -----------: | ------------: |
| arm/binary-b     |  122.8–126.2 |   111.4–113.2 |
| arm/binary-final |  111.9–126.8 |   105.7–110.0 |
| arm/bmp-a        |  861.4–871.4 |   811.3–853.9 |
| arm/bmp-b        |  811.8–858.0 |   816.2–824.5 |
| arm/bmp-c        |  470.1–581.7 |   460.8–582.5 |
| arm/bmp-d        |  444.4–601.6 |   473.8–604.7 |
| arm/bmp-e        |  448.8–594.1 |   448.5–649.1 |
| arm/scann        |  163.9–203.9 |   169.0–206.2 |
| arm/tq           |   88.3–109.1 |     84.3–88.6 |
| x86/binary-c     |    96.5–96.6 |     96.6–96.8 |
| x86/binary-final |    96.6–96.7 |     96.7–97.0 |
| x86/bmp-c-real   |  715.2–715.3 |   715.2–715.2 |
| x86/scann        |  101.4–101.6 |   101.4–101.6 |
