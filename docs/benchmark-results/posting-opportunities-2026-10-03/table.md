# Complete opportunity experiment table

Ratios are baseline/candidate; above 1 is faster. Query totals sum the
median wall time of each query, pooling both timing rounds. They describe
this suite, not a production traffic mix. Paired ratios use each run separately.
`multi_total` includes all 30 targeted queries; `multi_active` excludes the
four duplicate/missing-term controls, leaving 26 three- to six-term queries.

## Primitive screens

The current dispatcher is the baseline. BSR includes conversion and both
ordinal ranks; VP2 includes right-ordinal recovery. All calls use bounded
output. Real traces contain 7,118 sampled ARM calls; x86 replay is not a
sample of the 10M index's posting distribution.

**BSR rows are invalid for cross-block use:** the initial single-block
oracle missed lost tails when a decoded block ends inside a state group.
These timings are archived experiments, not valid adoption evidence.
The corrected BSR prototype also fails the corpus oracle; its campaign has no timing results.

| Host | Screen             | Fixture              | Candidate                               | Baseline ns/call | Candidate ns/call |  Ratio |
| ---- | ------------------ | -------------------- | --------------------------------------- | ---------------: | ----------------: | -----: |
| arm  | screen             | identical-1          | bsr_packed (invalid resume)             |             3.59 |             93.83 | 0.038× |
| arm  | screen             | identical-1          | bsr_stream (invalid resume)             |             3.59 |             28.92 | 0.124× |
| arm  | screen             | identical-1          | bsr_conditional (invalid resume)        |             3.59 |              4.06 | 0.885× |
| arm  | screen             | identical-128        | bsr_packed (invalid resume)             |           187.41 |            168.85 | 1.110× |
| arm  | screen             | identical-128        | bsr_stream (invalid resume)             |           187.41 |            189.06 | 0.991× |
| arm  | screen             | identical-128        | bsr_conditional (invalid resume)        |           187.41 |            189.51 | 0.989× |
| arm  | screen             | balanced_dense-1     | bsr_packed (invalid resume)             |             4.46 |            110.31 | 0.040× |
| arm  | screen             | balanced_dense-1     | bsr_stream (invalid resume)             |             4.46 |             17.36 | 0.257× |
| arm  | screen             | balanced_dense-1     | bsr_conditional (invalid resume)        |             4.46 |              4.81 | 0.928× |
| arm  | screen             | balanced_dense-128   | bsr_packed (invalid resume)             |           227.16 |            151.44 | 1.500× |
| arm  | screen             | balanced_dense-128   | bsr_stream (invalid resume)             |           227.16 |            165.73 | 1.371× |
| arm  | screen             | balanced_dense-128   | bsr_conditional (invalid resume)        |           227.16 |            166.28 | 1.366× |
| arm  | screen             | balanced_sparse-1    | bsr_packed (invalid resume)             |            54.91 |            337.06 | 0.163× |
| arm  | screen             | balanced_sparse-1    | bsr_stream (invalid resume)             |            54.91 |            465.49 | 0.118× |
| arm  | screen             | balanced_sparse-1    | bsr_conditional (invalid resume)        |            54.91 |             55.61 | 0.987× |
| arm  | screen             | balanced_sparse-128  | bsr_packed (invalid resume)             |           176.46 |            450.02 | 0.392× |
| arm  | screen             | balanced_sparse-128  | bsr_stream (invalid resume)             |           176.46 |           1315.93 | 0.134× |
| arm  | screen             | balanced_sparse-128  | bsr_conditional (invalid resume)        |           176.46 |            179.75 | 0.982× |
| arm  | screen             | left_8x_sparse-1     | bsr_packed (invalid resume)             |             7.57 |            204.67 | 0.037× |
| arm  | screen             | left_8x_sparse-1     | bsr_stream (invalid resume)             |             7.57 |             23.65 | 0.320× |
| arm  | screen             | left_8x_sparse-1     | bsr_conditional (invalid resume)        |             7.57 |              7.99 | 0.947× |
| arm  | screen             | left_8x_sparse-128   | bsr_packed (invalid resume)             |            41.30 |            220.84 | 0.187× |
| arm  | screen             | left_8x_sparse-128   | bsr_stream (invalid resume)             |            41.30 |            114.82 | 0.360× |
| arm  | screen             | left_8x_sparse-128   | bsr_conditional (invalid resume)        |            41.30 |             41.68 | 0.991× |
| arm  | screen             | right_8x_sparse-1    | bsr_packed (invalid resume)             |            21.45 |            205.03 | 0.105× |
| arm  | screen             | right_8x_sparse-1    | bsr_stream (invalid resume)             |            21.45 |             17.59 | 1.220× |
| arm  | screen             | right_8x_sparse-1    | bsr_conditional (invalid resume)        |            21.45 |             21.86 | 0.981× |
| arm  | screen             | right_8x_sparse-128  | bsr_packed (invalid resume)             |           183.67 |            220.86 | 0.832× |
| arm  | screen             | right_8x_sparse-128  | bsr_stream (invalid resume)             |           183.67 |             88.46 | 2.076× |
| arm  | screen             | right_8x_sparse-128  | bsr_conditional (invalid resume)        |           183.67 |            184.45 | 0.996× |
| arm  | screen             | left_64x_sparse-1    | bsr_packed (invalid resume)             |            10.34 |            221.02 | 0.047× |
| arm  | screen             | left_64x_sparse-1    | bsr_stream (invalid resume)             |            10.34 |             24.81 | 0.417× |
| arm  | screen             | left_64x_sparse-1    | bsr_conditional (invalid resume)        |            10.34 |             10.86 | 0.952× |
| arm  | screen             | left_64x_sparse-128  | bsr_packed (invalid resume)             |            15.35 |            220.95 | 0.069× |
| arm  | screen             | left_64x_sparse-128  | bsr_stream (invalid resume)             |            15.35 |             36.71 | 0.418× |
| arm  | screen             | left_64x_sparse-128  | bsr_conditional (invalid resume)        |            15.35 |             16.41 | 0.935× |
| arm  | screen             | right_64x_sparse-1   | bsr_packed (invalid resume)             |            49.16 |            218.55 | 0.225× |
| arm  | screen             | right_64x_sparse-1   | bsr_stream (invalid resume)             |            49.16 |             23.50 | 2.092× |
| arm  | screen             | right_64x_sparse-1   | bsr_conditional (invalid resume)        |            49.16 |             49.41 | 0.995× |
| arm  | screen             | right_64x_sparse-128 | bsr_packed (invalid resume)             |           104.48 |            221.10 | 0.473× |
| arm  | screen             | right_64x_sparse-128 | bsr_stream (invalid resume)             |           104.48 |             34.03 | 3.070× |
| arm  | screen             | right_64x_sparse-128 | bsr_conditional (invalid resume)        |           104.48 |            106.78 | 0.978× |
| arm  | replay             | trace-bitmap         | bsr_packed (invalid resume)             |           111.25 |            162.45 | 0.685× |
| arm  | replay             | trace-bitmap         | bsr_stream (invalid resume)             |           111.25 |            129.83 | 0.857× |
| arm  | replay             | trace-bitmap         | bsr_conditional (invalid resume)        |           111.25 |            101.78 | 1.093× |
| arm  | replay             | trace-rgb            | bsr_packed (invalid resume)             |            95.73 |            153.50 | 0.624× |
| arm  | replay             | trace-rgb            | bsr_stream (invalid resume)             |            95.73 |            121.22 | 0.790× |
| arm  | replay             | trace-rgb            | bsr_conditional (invalid resume)        |            95.73 |             97.18 | 0.985× |
| arm  | replay             | trace-multi-bitmap   | bsr_packed (invalid resume)             |            80.47 |            167.49 | 0.480× |
| arm  | replay             | trace-multi-bitmap   | bsr_stream (invalid resume)             |            80.47 |            108.36 | 0.743× |
| arm  | replay             | trace-multi-bitmap   | bsr_conditional (invalid resume)        |            80.47 |             87.22 | 0.923× |
| arm  | replay             | trace-multi-rgb      | bsr_packed (invalid resume)             |            82.37 |            157.30 | 0.524× |
| arm  | replay             | trace-multi-rgb      | bsr_stream (invalid resume)             |            82.37 |             97.32 | 0.846× |
| arm  | replay             | trace-multi-rgb      | bsr_conditional (invalid resume)        |            82.37 |             80.85 | 1.019× |
| arm  | orientation-screen | identical-1          | oriented                                |             3.84 |              4.95 | 0.775× |
| arm  | orientation-screen | identical-128        | oriented                                |           215.21 |            215.59 | 0.998× |
| arm  | orientation-screen | balanced_dense-1     | oriented                                |             4.89 |              5.76 | 0.849× |
| arm  | orientation-screen | balanced_dense-128   | oriented                                |           247.77 |            250.00 | 0.991× |
| arm  | orientation-screen | balanced_sparse-1    | oriented                                |            54.46 |             55.62 | 0.979× |
| arm  | orientation-screen | balanced_sparse-128  | oriented                                |           175.39 |            174.34 | 1.006× |
| arm  | orientation-screen | left_8x_sparse-1     | oriented                                |             7.75 |              8.48 | 0.914× |
| arm  | orientation-screen | left_8x_sparse-128   | oriented                                |            44.34 |             45.22 | 0.980× |
| arm  | orientation-screen | right_8x_sparse-1    | oriented                                |            21.63 |              8.90 | 2.430× |
| arm  | orientation-screen | right_8x_sparse-128  | oriented                                |           187.06 |             49.63 | 3.769× |
| arm  | orientation-screen | left_64x_sparse-1    | oriented                                |            10.59 |             11.33 | 0.935× |
| arm  | orientation-screen | left_64x_sparse-128  | oriented                                |            15.44 |             16.20 | 0.953× |
| arm  | orientation-screen | right_64x_sparse-1   | oriented                                |            49.15 |             11.65 | 4.218× |
| arm  | orientation-screen | right_64x_sparse-128 | oriented                                |           105.62 |             16.74 | 6.310× |
| arm  | orientation-replay | trace-bitmap         | oriented                                |           121.84 |            129.98 | 0.937× |
| arm  | orientation-replay | trace-rgb            | oriented                                |           111.72 |            120.26 | 0.929× |
| arm  | orientation-replay | trace-multi-bitmap   | oriented                                |            96.30 |            105.56 | 0.912× |
| arm  | orientation-replay | trace-multi-rgb      | oriented                                |           103.14 |             96.76 | 1.066× |
| x86  | screen             | identical-1          | bsr_packed (invalid resume)             |            13.45 |            184.53 | 0.073× |
| x86  | screen             | identical-1          | bsr_stream (invalid resume)             |            13.45 |             45.83 | 0.294× |
| x86  | screen             | identical-1          | bsr_conditional (invalid resume)        |            13.45 |             13.82 | 0.974× |
| x86  | screen             | identical-1          | vp2                                     |            13.45 |             17.21 | 0.782× |
| x86  | screen             | identical-1          | vp2_conditional                         |            13.45 |             20.28 | 0.663× |
| x86  | screen             | identical-128        | bsr_packed (invalid resume)             |           187.59 |            636.95 | 0.295× |
| x86  | screen             | identical-128        | bsr_stream (invalid resume)             |           187.59 |            610.87 | 0.307× |
| x86  | screen             | identical-128        | bsr_conditional (invalid resume)        |           187.59 |            614.75 | 0.305× |
| x86  | screen             | identical-128        | vp2                                     |           187.59 |            318.18 | 0.590× |
| x86  | screen             | identical-128        | vp2_conditional                         |           187.59 |            320.44 | 0.585× |
| x86  | screen             | balanced_dense-1     | bsr_packed (invalid resume)             |            13.87 |            204.11 | 0.068× |
| x86  | screen             | balanced_dense-1     | bsr_stream (invalid resume)             |            13.87 |             27.38 | 0.507× |
| x86  | screen             | balanced_dense-1     | bsr_conditional (invalid resume)        |            13.87 |             14.13 | 0.982× |
| x86  | screen             | balanced_dense-1     | vp2                                     |            13.87 |             17.28 | 0.803× |
| x86  | screen             | balanced_dense-1     | vp2_conditional                         |            13.87 |             20.52 | 0.676× |
| x86  | screen             | balanced_dense-128   | bsr_packed (invalid resume)             |           446.11 |            441.49 | 1.010× |
| x86  | screen             | balanced_dense-128   | bsr_stream (invalid resume)             |           446.11 |            388.64 | 1.148× |
| x86  | screen             | balanced_dense-128   | bsr_conditional (invalid resume)        |           446.11 |            388.29 | 1.149× |
| x86  | screen             | balanced_dense-128   | vp2                                     |           446.11 |            262.35 | 1.700× |
| x86  | screen             | balanced_dense-128   | vp2_conditional                         |           446.11 |            264.29 | 1.688× |
| x86  | screen             | balanced_sparse-1    | bsr_packed (invalid resume)             |            97.57 |            717.60 | 0.136× |
| x86  | screen             | balanced_sparse-1    | bsr_stream (invalid resume)             |            97.57 |            655.19 | 0.149× |
| x86  | screen             | balanced_sparse-1    | bsr_conditional (invalid resume)        |            97.57 |             98.28 | 0.993× |
| x86  | screen             | balanced_sparse-1    | vp2                                     |            97.57 |             45.61 | 2.139× |
| x86  | screen             | balanced_sparse-1    | vp2_conditional                         |            97.57 |             48.43 | 2.015× |
| x86  | screen             | balanced_sparse-128  | bsr_packed (invalid resume)             |           289.34 |           1461.95 | 0.198× |
| x86  | screen             | balanced_sparse-128  | bsr_stream (invalid resume)             |           289.34 |           2409.34 | 0.120× |
| x86  | screen             | balanced_sparse-128  | bsr_conditional (invalid resume)        |           289.34 |            289.80 | 0.998× |
| x86  | screen             | balanced_sparse-128  | vp2                                     |           289.34 |            127.18 | 2.275× |
| x86  | screen             | balanced_sparse-128  | vp2_conditional                         |           289.34 |            129.46 | 2.235× |
| x86  | screen             | left_8x_sparse-1     | bsr_packed (invalid resume)             |            26.10 |            418.42 | 0.062× |
| x86  | screen             | left_8x_sparse-1     | bsr_stream (invalid resume)             |            26.10 |             34.27 | 0.762× |
| x86  | screen             | left_8x_sparse-1     | bsr_conditional (invalid resume)        |            26.10 |             26.52 | 0.984× |
| x86  | screen             | left_8x_sparse-1     | vp2                                     |            26.10 |             22.09 | 1.182× |
| x86  | screen             | left_8x_sparse-1     | vp2_conditional                         |            26.10 |             28.67 | 0.910× |
| x86  | screen             | left_8x_sparse-128   | bsr_packed (invalid resume)             |            94.29 |            506.05 | 0.186× |
| x86  | screen             | left_8x_sparse-128   | bsr_stream (invalid resume)             |            94.29 |            150.21 | 0.628× |
| x86  | screen             | left_8x_sparse-128   | bsr_conditional (invalid resume)        |            94.29 |             94.72 | 0.995× |
| x86  | screen             | left_8x_sparse-128   | vp2                                     |            94.29 |             92.82 | 1.016× |
| x86  | screen             | left_8x_sparse-128   | vp2_conditional                         |            94.29 |             96.98 | 0.972× |
| x86  | screen             | right_8x_sparse-1    | bsr_packed (invalid resume)             |            41.43 |            424.97 | 0.097× |
| x86  | screen             | right_8x_sparse-1    | bsr_stream (invalid resume)             |            41.43 |             31.16 | 1.330× |
| x86  | screen             | right_8x_sparse-1    | bsr_conditional (invalid resume)        |            41.43 |             41.50 | 0.998× |
| x86  | screen             | right_8x_sparse-1    | vp2                                     |            41.43 |             21.40 | 1.936× |
| x86  | screen             | right_8x_sparse-1    | vp2_conditional                         |            41.43 |             44.64 | 0.928× |
| x86  | screen             | right_8x_sparse-128  | bsr_packed (invalid resume)             |           223.13 |            462.43 | 0.483× |
| x86  | screen             | right_8x_sparse-128  | bsr_stream (invalid resume)             |           223.13 |            149.07 | 1.497× |
| x86  | screen             | right_8x_sparse-128  | bsr_conditional (invalid resume)        |           223.13 |            221.28 | 1.008× |
| x86  | screen             | right_8x_sparse-128  | vp2                                     |           223.13 |             93.16 | 2.395× |
| x86  | screen             | right_8x_sparse-128  | vp2_conditional                         |           223.13 |            223.03 | 1.000× |
| x86  | screen             | left_64x_sparse-1    | bsr_packed (invalid resume)             |            24.71 |            445.73 | 0.055× |
| x86  | screen             | left_64x_sparse-1    | bsr_stream (invalid resume)             |            24.71 |             50.47 | 0.490× |
| x86  | screen             | left_64x_sparse-1    | bsr_conditional (invalid resume)        |            24.71 |             25.00 | 0.988× |
| x86  | screen             | left_64x_sparse-1    | vp2                                     |            24.71 |             49.95 | 0.495× |
| x86  | screen             | left_64x_sparse-1    | vp2_conditional                         |            24.71 |             27.62 | 0.895× |
| x86  | screen             | left_64x_sparse-128  | bsr_packed (invalid resume)             |            31.66 |            487.93 | 0.065× |
| x86  | screen             | left_64x_sparse-128  | bsr_stream (invalid resume)             |            31.66 |             69.35 | 0.457× |
| x86  | screen             | left_64x_sparse-128  | bsr_conditional (invalid resume)        |            31.66 |             32.79 | 0.965× |
| x86  | screen             | left_64x_sparse-128  | vp2                                     |            31.66 |             76.34 | 0.415× |
| x86  | screen             | left_64x_sparse-128  | vp2_conditional                         |            31.66 |             34.69 | 0.913× |
| x86  | screen             | right_64x_sparse-1   | bsr_packed (invalid resume)             |           145.74 |            407.61 | 0.358× |
| x86  | screen             | right_64x_sparse-1   | bsr_stream (invalid resume)             |           145.74 |             49.83 | 2.925× |
| x86  | screen             | right_64x_sparse-1   | bsr_conditional (invalid resume)        |           145.74 |            146.46 | 0.995× |
| x86  | screen             | right_64x_sparse-1   | vp2                                     |           145.74 |             50.05 | 2.912× |
| x86  | screen             | right_64x_sparse-1   | vp2_conditional                         |           145.74 |            148.86 | 0.979× |
| x86  | screen             | right_64x_sparse-128 | bsr_packed (invalid resume)             |           218.21 |            405.27 | 0.538× |
| x86  | screen             | right_64x_sparse-128 | bsr_stream (invalid resume)             |           218.21 |             69.82 | 3.125× |
| x86  | screen             | right_64x_sparse-128 | bsr_conditional (invalid resume)        |           218.21 |            220.99 | 0.987× |
| x86  | screen             | right_64x_sparse-128 | vp2                                     |           218.21 |             75.69 | 2.883× |
| x86  | screen             | right_64x_sparse-128 | vp2_conditional                         |           218.21 |            220.35 | 0.990× |
| x86  | replay             | trace-bitmap         | bsr_packed (invalid resume)             |           306.36 |            471.34 | 0.650× |
| x86  | replay             | trace-bitmap         | bsr_stream (invalid resume)             |           306.36 |            339.04 | 0.904× |
| x86  | replay             | trace-bitmap         | bsr_conditional (invalid resume)        |           306.36 |            324.46 | 0.944× |
| x86  | replay             | trace-bitmap         | vp2                                     |           306.36 |            268.11 | 1.143× |
| x86  | replay             | trace-bitmap         | vp2_conditional                         |           306.36 |            288.87 | 1.061× |
| x86  | replay             | trace-rgb            | bsr_packed (invalid resume)             |           296.27 |            457.59 | 0.647× |
| x86  | replay             | trace-rgb            | bsr_stream (invalid resume)             |           296.27 |            325.36 | 0.911× |
| x86  | replay             | trace-rgb            | bsr_conditional (invalid resume)        |           296.27 |            311.88 | 0.950× |
| x86  | replay             | trace-rgb            | vp2                                     |           296.27 |            263.83 | 1.123× |
| x86  | replay             | trace-rgb            | vp2_conditional                         |           296.27 |            285.36 | 1.038× |
| x86  | replay             | trace-multi-bitmap   | bsr_packed (invalid resume)             |           226.03 |            461.31 | 0.490× |
| x86  | replay             | trace-multi-bitmap   | bsr_stream (invalid resume)             |           226.03 |            298.12 | 0.758× |
| x86  | replay             | trace-multi-bitmap   | bsr_conditional (invalid resume)        |           226.03 |            256.98 | 0.880× |
| x86  | replay             | trace-multi-bitmap   | vp2                                     |           226.03 |            201.28 | 1.123× |
| x86  | replay             | trace-multi-bitmap   | vp2_conditional                         |           226.03 |            215.08 | 1.051× |
| x86  | replay             | trace-multi-rgb      | bsr_packed (invalid resume)             |           224.56 |            453.05 | 0.496× |
| x86  | replay             | trace-multi-rgb      | bsr_stream (invalid resume)             |           224.56 |            290.29 | 0.774× |
| x86  | replay             | trace-multi-rgb      | bsr_conditional (invalid resume)        |           224.56 |            260.39 | 0.862× |
| x86  | replay             | trace-multi-rgb      | vp2                                     |           224.56 |            197.56 | 1.137× |
| x86  | replay             | trace-multi-rgb      | vp2_conditional                         |           224.56 |            219.84 | 1.021× |
| x86  | orientation-screen | identical-1          | oriented                                |            13.22 |             14.92 | 0.886× |
| x86  | orientation-screen | identical-128        | oriented                                |           187.36 |            188.95 | 0.992× |
| x86  | orientation-screen | balanced_dense-1     | oriented                                |            13.51 |             15.01 | 0.900× |
| x86  | orientation-screen | balanced_dense-128   | oriented                                |           278.77 |            280.03 | 0.995× |
| x86  | orientation-screen | balanced_sparse-1    | oriented                                |            94.60 |             95.84 | 0.987× |
| x86  | orientation-screen | balanced_sparse-128  | oriented                                |           283.79 |            285.11 | 0.995× |
| x86  | orientation-screen | left_8x_sparse-1     | oriented                                |            25.70 |             27.71 | 0.927× |
| x86  | orientation-screen | left_8x_sparse-128   | oriented                                |            91.08 |             92.99 | 0.979× |
| x86  | orientation-screen | right_8x_sparse-1    | oriented                                |            41.11 |             28.04 | 1.466× |
| x86  | orientation-screen | right_8x_sparse-128  | oriented                                |           225.29 |             95.84 | 2.351× |
| x86  | orientation-screen | left_64x_sparse-1    | oriented                                |            24.79 |             27.05 | 0.916× |
| x86  | orientation-screen | left_64x_sparse-128  | oriented                                |            30.91 |             33.16 | 0.932× |
| x86  | orientation-screen | right_64x_sparse-1   | oriented                                |           147.79 |             27.69 | 5.337× |
| x86  | orientation-screen | right_64x_sparse-128 | oriented                                |           219.29 |             34.12 | 6.427× |
| x86  | orientation-replay | trace-bitmap         | oriented                                |           297.56 |            300.99 | 0.989× |
| x86  | orientation-replay | trace-rgb            | oriented                                |           290.38 |            292.22 | 0.994× |
| x86  | orientation-replay | trace-multi-bitmap   | oriented                                |           218.47 |            221.14 | 0.988× |
| x86  | orientation-replay | trace-multi-rgb      | oriented                                |           222.55 |            225.23 | 0.988× |
| x86  | popcnt-screen      | identical-1          | bsr_packed (invalid resume)             |            13.83 |            170.69 | 0.081× |
| x86  | popcnt-screen      | identical-1          | bsr_stream (invalid resume)             |            13.83 |             42.14 | 0.328× |
| x86  | popcnt-screen      | identical-1          | bsr_packed_popcnt (invalid resume)      |            13.83 |            182.88 | 0.076× |
| x86  | popcnt-screen      | identical-1          | bsr_stream_popcnt (invalid resume)      |            13.83 |             43.62 | 0.317× |
| x86  | popcnt-screen      | identical-1          | bsr_conditional_popcnt (invalid resume) |            13.83 |             14.12 | 0.980× |
| x86  | popcnt-screen      | identical-128        | bsr_packed (invalid resume)             |           185.19 |            634.21 | 0.292× |
| x86  | popcnt-screen      | identical-128        | bsr_stream (invalid resume)             |           185.19 |            597.19 | 0.310× |
| x86  | popcnt-screen      | identical-128        | bsr_packed_popcnt (invalid resume)      |           185.19 |            400.54 | 0.462× |
| x86  | popcnt-screen      | identical-128        | bsr_stream_popcnt (invalid resume)      |           185.19 |            365.41 | 0.507× |
| x86  | popcnt-screen      | identical-128        | bsr_conditional_popcnt (invalid resume) |           185.19 |            367.25 | 0.504× |
| x86  | popcnt-screen      | balanced_dense-1     | bsr_packed (invalid resume)             |            13.91 |            207.89 | 0.067× |
| x86  | popcnt-screen      | balanced_dense-1     | bsr_stream (invalid resume)             |            13.91 |             25.81 | 0.539× |
| x86  | popcnt-screen      | balanced_dense-1     | bsr_packed_popcnt (invalid resume)      |            13.91 |            207.68 | 0.067× |
| x86  | popcnt-screen      | balanced_dense-1     | bsr_stream_popcnt (invalid resume)      |            13.91 |             26.33 | 0.528× |
| x86  | popcnt-screen      | balanced_dense-1     | bsr_conditional_popcnt (invalid resume) |            13.91 |             14.36 | 0.969× |
| x86  | popcnt-screen      | balanced_dense-128   | bsr_packed (invalid resume)             |           285.08 |            453.05 | 0.629× |
| x86  | popcnt-screen      | balanced_dense-128   | bsr_stream (invalid resume)             |           285.08 |            388.51 | 0.734× |
| x86  | popcnt-screen      | balanced_dense-128   | bsr_packed_popcnt (invalid resume)      |           285.08 |            318.85 | 0.894× |
| x86  | popcnt-screen      | balanced_dense-128   | bsr_stream_popcnt (invalid resume)      |           285.08 |            262.05 | 1.088× |
| x86  | popcnt-screen      | balanced_dense-128   | bsr_conditional_popcnt (invalid resume) |           285.08 |            265.42 | 1.074× |
| x86  | popcnt-screen      | balanced_sparse-1    | bsr_packed (invalid resume)             |           101.71 |            844.47 | 0.120× |
| x86  | popcnt-screen      | balanced_sparse-1    | bsr_stream (invalid resume)             |           101.71 |            653.98 | 0.156× |
| x86  | popcnt-screen      | balanced_sparse-1    | bsr_packed_popcnt (invalid resume)      |           101.71 |            707.69 | 0.144× |
| x86  | popcnt-screen      | balanced_sparse-1    | bsr_stream_popcnt (invalid resume)      |           101.71 |            647.36 | 0.157× |
| x86  | popcnt-screen      | balanced_sparse-1    | bsr_conditional_popcnt (invalid resume) |           101.71 |            102.35 | 0.994× |
| x86  | popcnt-screen      | balanced_sparse-128  | bsr_packed (invalid resume)             |           294.45 |           1558.06 | 0.189× |
| x86  | popcnt-screen      | balanced_sparse-128  | bsr_stream (invalid resume)             |           294.45 |           2296.27 | 0.128× |
| x86  | popcnt-screen      | balanced_sparse-128  | bsr_packed_popcnt (invalid resume)      |           294.45 |           1543.35 | 0.191× |
| x86  | popcnt-screen      | balanced_sparse-128  | bsr_stream_popcnt (invalid resume)      |           294.45 |           2209.48 | 0.133× |
| x86  | popcnt-screen      | balanced_sparse-128  | bsr_conditional_popcnt (invalid resume) |           294.45 |            294.90 | 0.998× |
| x86  | popcnt-screen      | left_8x_sparse-1     | bsr_packed (invalid resume)             |            26.48 |            563.93 | 0.047× |
| x86  | popcnt-screen      | left_8x_sparse-1     | bsr_stream (invalid resume)             |            26.48 |             32.55 | 0.814× |
| x86  | popcnt-screen      | left_8x_sparse-1     | bsr_packed_popcnt (invalid resume)      |            26.48 |            436.56 | 0.061× |
| x86  | popcnt-screen      | left_8x_sparse-1     | bsr_stream_popcnt (invalid resume)      |            26.48 |             35.42 | 0.748× |
| x86  | popcnt-screen      | left_8x_sparse-1     | bsr_conditional_popcnt (invalid resume) |            26.48 |             26.75 | 0.990× |
| x86  | popcnt-screen      | left_8x_sparse-128   | bsr_packed (invalid resume)             |            92.41 |            604.46 | 0.153× |
| x86  | popcnt-screen      | left_8x_sparse-128   | bsr_stream (invalid resume)             |            92.41 |            131.98 | 0.700× |
| x86  | popcnt-screen      | left_8x_sparse-128   | bsr_packed_popcnt (invalid resume)      |            92.41 |            503.25 | 0.184× |
| x86  | popcnt-screen      | left_8x_sparse-128   | bsr_stream_popcnt (invalid resume)      |            92.41 |            144.82 | 0.638× |
| x86  | popcnt-screen      | left_8x_sparse-128   | bsr_conditional_popcnt (invalid resume) |            92.41 |             93.46 | 0.989× |
| x86  | popcnt-screen      | right_8x_sparse-1    | bsr_packed (invalid resume)             |            41.43 |            547.66 | 0.076× |
| x86  | popcnt-screen      | right_8x_sparse-1    | bsr_stream (invalid resume)             |            41.43 |             31.13 | 1.331× |
| x86  | popcnt-screen      | right_8x_sparse-1    | bsr_packed_popcnt (invalid resume)      |            41.43 |            416.54 | 0.099× |
| x86  | popcnt-screen      | right_8x_sparse-1    | bsr_stream_popcnt (invalid resume)      |            41.43 |             29.85 | 1.388× |
| x86  | popcnt-screen      | right_8x_sparse-1    | bsr_conditional_popcnt (invalid resume) |            41.43 |             41.40 | 1.001× |
| x86  | popcnt-screen      | right_8x_sparse-128  | bsr_packed (invalid resume)             |           216.09 |            585.79 | 0.369× |
| x86  | popcnt-screen      | right_8x_sparse-128  | bsr_stream (invalid resume)             |           216.09 |            145.10 | 1.489× |
| x86  | popcnt-screen      | right_8x_sparse-128  | bsr_packed_popcnt (invalid resume)      |           216.09 |            448.88 | 0.481× |
| x86  | popcnt-screen      | right_8x_sparse-128  | bsr_stream_popcnt (invalid resume)      |           216.09 |            130.16 | 1.660× |
| x86  | popcnt-screen      | right_8x_sparse-128  | bsr_conditional_popcnt (invalid resume) |           216.09 |            220.07 | 0.982× |
| x86  | popcnt-screen      | left_64x_sparse-1    | bsr_packed (invalid resume)             |            24.98 |            447.53 | 0.056× |
| x86  | popcnt-screen      | left_64x_sparse-1    | bsr_stream (invalid resume)             |            24.98 |             50.68 | 0.493× |
| x86  | popcnt-screen      | left_64x_sparse-1    | bsr_packed_popcnt (invalid resume)      |            24.98 |            432.48 | 0.058× |
| x86  | popcnt-screen      | left_64x_sparse-1    | bsr_stream_popcnt (invalid resume)      |            24.98 |             50.52 | 0.495× |
| x86  | popcnt-screen      | left_64x_sparse-1    | bsr_conditional_popcnt (invalid resume) |            24.98 |             25.15 | 0.993× |
| x86  | popcnt-screen      | left_64x_sparse-128  | bsr_packed (invalid resume)             |            31.67 |            475.27 | 0.067× |
| x86  | popcnt-screen      | left_64x_sparse-128  | bsr_stream (invalid resume)             |            31.67 |             69.70 | 0.454× |
| x86  | popcnt-screen      | left_64x_sparse-128  | bsr_packed_popcnt (invalid resume)      |            31.67 |            452.72 | 0.070× |
| x86  | popcnt-screen      | left_64x_sparse-128  | bsr_stream_popcnt (invalid resume)      |            31.67 |             69.62 | 0.455× |
| x86  | popcnt-screen      | left_64x_sparse-128  | bsr_conditional_popcnt (invalid resume) |            31.67 |             32.56 | 0.973× |
| x86  | popcnt-screen      | right_64x_sparse-1   | bsr_packed (invalid resume)             |           148.05 |            428.34 | 0.346× |
| x86  | popcnt-screen      | right_64x_sparse-1   | bsr_stream (invalid resume)             |           148.05 |             49.73 | 2.977× |
| x86  | popcnt-screen      | right_64x_sparse-1   | bsr_packed_popcnt (invalid resume)      |           148.05 |            428.22 | 0.346× |
| x86  | popcnt-screen      | right_64x_sparse-1   | bsr_stream_popcnt (invalid resume)      |           148.05 |             49.63 | 2.983× |
| x86  | popcnt-screen      | right_64x_sparse-1   | bsr_conditional_popcnt (invalid resume) |           148.05 |            149.72 | 0.989× |
| x86  | popcnt-screen      | right_64x_sparse-128 | bsr_packed (invalid resume)             |           220.78 |            462.81 | 0.477× |
| x86  | popcnt-screen      | right_64x_sparse-128 | bsr_stream (invalid resume)             |           220.78 |             68.42 | 3.227× |
| x86  | popcnt-screen      | right_64x_sparse-128 | bsr_packed_popcnt (invalid resume)      |           220.78 |            449.59 | 0.491× |
| x86  | popcnt-screen      | right_64x_sparse-128 | bsr_stream_popcnt (invalid resume)      |           220.78 |             68.32 | 3.231× |
| x86  | popcnt-screen      | right_64x_sparse-128 | bsr_conditional_popcnt (invalid resume) |           220.78 |            222.51 | 0.992× |
| x86  | popcnt-replay      | trace-bitmap         | bsr_packed (invalid resume)             |           297.11 |            480.18 | 0.619× |
| x86  | popcnt-replay      | trace-bitmap         | bsr_stream (invalid resume)             |           297.11 |            335.38 | 0.886× |
| x86  | popcnt-replay      | trace-bitmap         | bsr_packed_popcnt (invalid resume)      |           297.11 |            408.91 | 0.727× |
| x86  | popcnt-replay      | trace-bitmap         | bsr_stream_popcnt (invalid resume)      |           297.11 |            273.97 | 1.084× |
| x86  | popcnt-replay      | trace-bitmap         | bsr_conditional_popcnt (invalid resume) |           297.11 |            267.70 | 1.110× |
| x86  | popcnt-replay      | trace-rgb            | bsr_packed (invalid resume)             |           290.76 |            459.48 | 0.633× |
| x86  | popcnt-replay      | trace-rgb            | bsr_stream (invalid resume)             |           290.76 |            321.40 | 0.905× |
| x86  | popcnt-replay      | trace-rgb            | bsr_packed_popcnt (invalid resume)      |           290.76 |            391.82 | 0.742× |
| x86  | popcnt-replay      | trace-rgb            | bsr_stream_popcnt (invalid resume)      |           290.76 |            260.22 | 1.117× |
| x86  | popcnt-replay      | trace-rgb            | bsr_conditional_popcnt (invalid resume) |           290.76 |            256.10 | 1.135× |
| x86  | popcnt-replay      | trace-multi-bitmap   | bsr_packed (invalid resume)             |           220.62 |            466.20 | 0.473× |
| x86  | popcnt-replay      | trace-multi-bitmap   | bsr_stream (invalid resume)             |           220.62 |            296.56 | 0.744× |
| x86  | popcnt-replay      | trace-multi-bitmap   | bsr_packed_popcnt (invalid resume)      |           220.62 |            408.95 | 0.539× |
| x86  | popcnt-replay      | trace-multi-bitmap   | bsr_stream_popcnt (invalid resume)      |           220.62 |            248.36 | 0.888× |
| x86  | popcnt-replay      | trace-multi-bitmap   | bsr_conditional_popcnt (invalid resume) |           220.62 |            213.75 | 1.032× |
| x86  | popcnt-replay      | trace-multi-rgb      | bsr_packed (invalid resume)             |           220.70 |            453.82 | 0.486× |
| x86  | popcnt-replay      | trace-multi-rgb      | bsr_stream (invalid resume)             |           220.70 |            286.92 | 0.769× |
| x86  | popcnt-replay      | trace-multi-rgb      | bsr_packed_popcnt (invalid resume)      |           220.70 |            397.20 | 0.556× |
| x86  | popcnt-replay      | trace-multi-rgb      | bsr_stream_popcnt (invalid resume)      |           220.70 |            236.23 | 0.934× |
| x86  | popcnt-replay      | trace-multi-rgb      | bsr_conditional_popcnt (invalid resume) |           220.70 |            211.79 | 1.042× |

## Real-query comparisons

Every cell validates the independent exhaustive audit, complete recorded
ID/score-bit/count/plan equality, repetition counts and immutable file inventories.

| Campaign            | Host/index    | Comparison          | Limit | Queries | Before ms | After ms |  Ratio | Paired ratios |
| ------------------- | ------------- | ------------------- | ----: | ------: | --------: | -------: | -----: | ------------- |
| adaptive-integrated | arm/bitmap    | baseline/adaptive   |    10 |      30 |   139.338 |  140.101 | 0.995× | 0.986/0.998   |
| adaptive-integrated | arm/bitmap    | baseline/adaptive   |   100 |      30 |   139.223 |  138.844 | 1.003× | 1.002/1.005   |
| adaptive-integrated | arm/bitmap    | baseline/adaptive   |     0 |      30 |    10.116 |   10.030 | 1.009× | 1.021/1.002   |
| adaptive-integrated | arm/rgb       | baseline/adaptive   |    10 |      30 |   139.121 |  137.408 | 1.012× | 1.008/1.021   |
| adaptive-integrated | arm/rgb       | baseline/adaptive   |   100 |      30 |   138.824 |  138.764 | 1.000× | 1.000/1.000   |
| adaptive-integrated | arm/rgb       | baseline/adaptive   |     0 |      30 |     9.650 |    9.551 | 1.010× | 1.015/1.009   |
| adaptive-integrated | x86/default   | baseline/adaptive   |    10 |      30 |  2641.327 | 2577.239 | 1.025× | 1.024/1.024   |
| adaptive-integrated | x86/default   | baseline/adaptive   |   100 |      30 |  2643.609 | 2581.886 | 1.024× | 1.026/1.023   |
| adaptive-integrated | x86/default   | baseline/adaptive   |     0 |      30 |   203.147 |  197.418 | 1.029× | 1.029/1.027   |
| adaptive-integrated | x86/rgb-pairs | baseline/adaptive   |    10 |      30 |  2562.300 | 2524.365 | 1.015× | 1.015/1.016   |
| adaptive-integrated | x86/rgb-pairs | baseline/adaptive   |   100 |      30 |  2561.063 | 2531.413 | 1.012× | 1.013/1.012   |
| adaptive-integrated | x86/rgb-pairs | baseline/adaptive   |     0 |      30 |   197.200 |  196.549 | 1.003× | 1.036/0.981   |
| batched-integrated  | arm/bitmap    | baseline/batched    |    10 |      30 |   139.402 |  126.253 | 1.104× | 1.101/1.110   |
| batched-integrated  | arm/bitmap    | baseline/batched    |   100 |      30 |   139.793 |  127.659 | 1.095× | 1.085/1.098   |
| batched-integrated  | arm/bitmap    | baseline/batched    |     0 |      30 |    10.148 |   10.129 | 1.002× | 1.014/1.002   |
| batched-integrated  | arm/rgb       | baseline/batched    |    10 |      30 |   138.375 |  126.450 | 1.094× | 1.100/1.090   |
| batched-integrated  | arm/rgb       | baseline/batched    |   100 |      30 |   139.283 |  128.155 | 1.087× | 1.087/1.090   |
| batched-integrated  | arm/rgb       | baseline/batched    |     0 |      30 |     9.653 |    9.569 | 1.009× | 1.011/1.009   |
| batched-integrated  | x86/default   | baseline/batched    |    10 |      30 |  2639.876 | 2249.814 | 1.173× | 1.176/1.173   |
| batched-integrated  | x86/default   | baseline/batched    |   100 |      30 |  2640.815 | 2248.217 | 1.175× | 1.176/1.174   |
| batched-integrated  | x86/default   | baseline/batched    |     0 |      30 |   201.687 |  201.107 | 1.003× | 1.011/0.999   |
| batched-integrated  | x86/rgb-pairs | baseline/batched    |    10 |      30 |  2567.116 | 2144.784 | 1.197× | 1.194/1.201   |
| batched-integrated  | x86/rgb-pairs | baseline/batched    |   100 |      30 |  2561.177 | 2141.363 | 1.196× | 1.198/1.195   |
| batched-integrated  | x86/rgb-pairs | baseline/batched    |     0 |      30 |   192.921 |  193.823 | 0.995× | 1.005/0.990   |
| broad               | arm/bitmap    | baseline/batched    |    10 |     199 |   251.014 |  236.728 | 1.060× | 1.059/1.064   |
| broad               | arm/bitmap    | baseline/batched    |   100 |     199 |   363.823 |  351.636 | 1.035× | 1.034/1.037   |
| broad               | arm/bitmap    | baseline/batched    |     0 |     199 |   504.541 |  499.923 | 1.009× | 1.015/1.003   |
| broad               | arm/rgb       | baseline/batched    |    10 |     199 |   229.859 |  215.907 | 1.065× | 1.063/1.068   |
| broad               | arm/rgb       | baseline/batched    |   100 |     199 |   320.660 |  308.426 | 1.040× | 1.045/1.038   |
| broad               | arm/rgb       | baseline/batched    |     0 |     199 |   481.391 |  476.798 | 1.010× | 1.011/1.008   |
| batched-final       | arm/bitmap    | baseline/selected   |    10 |     199 |   248.638 |  237.988 | 1.045× | 1.048/1.037   |
| batched-final       | arm/bitmap    | baseline/selected   |   100 |     199 |   362.437 |  353.763 | 1.025× | 1.012/1.031   |
| batched-final       | arm/bitmap    | baseline/selected   |     0 |     199 |   501.128 |  499.311 | 1.004× | 0.997/1.010   |
| batched-final       | arm/rgb       | baseline/selected   |    10 |     199 |   227.398 |  215.571 | 1.055× | 1.058/1.053   |
| batched-final       | arm/rgb       | baseline/selected   |   100 |     199 |   318.027 |  306.631 | 1.037× | 1.036/1.038   |
| batched-final       | arm/rgb       | baseline/selected   |     0 |     199 |   479.720 |  476.634 | 1.006× | 1.005/1.009   |
| final               | arm/bitmap    | baseline/gated      |    10 |     199 |   249.878 |  233.633 | 1.070× | 1.080/1.059   |
| final               | arm/bitmap    | baseline/gated      |   100 |     199 |   362.790 |  348.023 | 1.042× | 1.045/1.040   |
| final               | arm/bitmap    | baseline/gated      |     0 |     199 |   503.110 |  498.567 | 1.009× | 1.020/1.001   |
| final               | arm/rgb       | baseline/gated      |    10 |     199 |   227.872 |  213.130 | 1.069× | 1.042/1.083   |
| final               | arm/rgb       | baseline/gated      |   100 |     199 |   322.429 |  303.569 | 1.062× | 1.064/1.065   |
| final               | arm/rgb       | baseline/gated      |     0 |     199 |   480.931 |  479.093 | 1.004× | 1.003/1.005   |
| final               | x86/default   | baseline/gated      |    10 |     199 |  3742.333 | 3358.864 | 1.114× | 1.115/1.114   |
| final               | x86/default   | baseline/wide-gated |    10 |     199 |  3742.333 | 3442.639 | 1.087× | 1.087/1.087   |
| final               | x86/default   | gated/wide-gated    |    10 |     199 |  3358.864 | 3442.639 | 0.976× | 0.975/0.976   |
| final               | x86/default   | baseline/gated      |   100 |     199 |  4954.747 | 4567.095 | 1.085× | 1.085/1.085   |
| final               | x86/default   | baseline/wide-gated |   100 |     199 |  4954.747 | 4664.280 | 1.062× | 1.063/1.062   |
| final               | x86/default   | gated/wide-gated    |   100 |     199 |  4567.095 | 4664.280 | 0.979× | 0.980/0.979   |
| final               | x86/default   | baseline/gated      |     0 |     199 |  8421.127 | 8334.788 | 1.010× | 1.019/1.000   |
| final               | x86/default   | baseline/wide-gated |     0 |     199 |  8421.127 | 8589.456 | 0.980× | 0.989/0.970   |
| final               | x86/default   | gated/wide-gated    |     0 |     199 |  8334.788 | 8589.456 | 0.970× | 0.970/0.971   |
| final               | x86/rgb-pairs | baseline/gated      |    10 |     199 |  3264.970 | 2818.491 | 1.158× | 1.157/1.160   |
| final               | x86/rgb-pairs | baseline/wide-gated |    10 |     199 |  3264.970 | 2916.082 | 1.120× | 1.117/1.123   |
| final               | x86/rgb-pairs | gated/wide-gated    |    10 |     199 |  2818.491 | 2916.082 | 0.967× | 0.965/0.968   |
| final               | x86/rgb-pairs | baseline/gated      |   100 |     199 |  3626.160 | 3167.825 | 1.145× | 1.144/1.146   |
| final               | x86/rgb-pairs | baseline/wide-gated |   100 |     199 |  3626.160 | 3267.178 | 1.110× | 1.112/1.107   |
| final               | x86/rgb-pairs | gated/wide-gated    |   100 |     199 |  3167.825 | 3267.178 | 0.970× | 0.972/0.967   |
| final               | x86/rgb-pairs | baseline/gated      |     0 |     199 |  2225.169 | 2228.303 | 0.999× | 1.003/0.996   |
| final               | x86/rgb-pairs | baseline/wide-gated |     0 |     199 |  2225.169 | 2257.965 | 0.985× | 0.989/0.983   |
| final               | x86/rgb-pairs | gated/wide-gated    |     0 |     199 |  2228.303 | 2257.965 | 0.987× | 0.987/0.987   |
| confirm             | arm/bitmap    | baseline/gated      |    10 |     199 |   370.249 |  327.057 | 1.132× | 1.180/1.099   |
| confirm             | arm/bitmap    | baseline/gated      |   100 |     199 |   607.149 |  507.630 | 1.196× | 1.426/1.037   |
| confirm             | arm/bitmap    | baseline/gated      |     0 |     199 |   803.042 |  825.465 | 0.973× | 1.148/0.839   |
| confirm             | arm/rgb       | baseline/gated      |    10 |     199 |   327.821 |  355.545 | 0.922× | 1.064/0.778   |
| confirm             | arm/rgb       | baseline/gated      |   100 |     199 |   480.307 |  460.582 | 1.043× | 0.982/1.102   |
| confirm             | arm/rgb       | baseline/gated      |     0 |     199 |   706.757 |  867.746 | 0.814× | 0.677/0.988   |
| selected-final      | x86/default   | baseline/gated      |    10 |     199 |  3806.512 | 3438.694 | 1.107× | 1.105/1.109   |
| selected-final      | x86/default   | baseline/gated      |   100 |     199 |  5041.924 | 4675.562 | 1.078× | 1.079/1.079   |
| selected-final      | x86/default   | baseline/gated      |     0 |     199 |  8460.391 | 8427.763 | 1.004× | 1.003/1.003   |
| selected-final      | x86/rgb-pairs | baseline/gated      |    10 |     199 |  3298.924 | 2856.107 | 1.155× | 1.156/1.154   |
| selected-final      | x86/rgb-pairs | baseline/gated      |   100 |     199 |  3686.802 | 3217.028 | 1.146× | 1.144/1.147   |
| selected-final      | x86/rgb-pairs | baseline/gated      |     0 |     199 |  2269.126 | 2292.990 | 0.990× | 0.999/0.982   |

## Final per-family results

| Campaign       | Host/index    | Comparison          | Family             | Limit |  Ratio | Paired ratios |
| -------------- | ------------- | ------------------- | ------------------ | ----: | -----: | ------------- |
| final          | arm/bitmap    | baseline/gated      | multi_total        |    10 | 1.125× | 1.130/1.120   |
| final          | arm/bitmap    | baseline/gated      | multi_active       |    10 | 1.109× | 1.115/1.103   |
| final          | arm/bitmap    | baseline/gated      | and_high_high      |    10 | 0.972× | 0.966/0.964   |
| final          | arm/bitmap    | baseline/gated      | and_high_low       |    10 | 0.982× | 1.002/0.969   |
| final          | arm/bitmap    | baseline/gated      | and_high_med       |    10 | 0.971× | 0.963/0.979   |
| final          | arm/bitmap    | baseline/gated      | and_multi          |    10 | 1.128× | 1.210/1.081   |
| final          | arm/bitmap    | baseline/gated      | high_phrase        |    10 | 1.012× | 1.043/0.992   |
| final          | arm/bitmap    | baseline/gated      | high_sloppy_phrase |    10 | 0.983× | 0.993/0.957   |
| final          | arm/bitmap    | baseline/gated      | high_term          |    10 | 1.029× | 1.026/1.029   |
| final          | arm/bitmap    | baseline/gated      | low_phrase         |    10 | 0.996× | 0.981/1.003   |
| final          | arm/bitmap    | baseline/gated      | low_sloppy_phrase  |    10 | 0.994× | 1.000/0.970   |
| final          | arm/bitmap    | baseline/gated      | low_term           |    10 | 0.912× | 0.858/1.081   |
| final          | arm/bitmap    | baseline/gated      | med_phrase         |    10 | 0.995× | 1.005/0.989   |
| final          | arm/bitmap    | baseline/gated      | med_sloppy_phrase  |    10 | 0.988× | 0.994/0.971   |
| final          | arm/bitmap    | baseline/gated      | med_term           |    10 | 0.971× | 0.900/1.028   |
| final          | arm/bitmap    | baseline/gated      | multi_balanced     |    10 | 1.129× | 1.148/1.116   |
| final          | arm/bitmap    | baseline/gated      | multi_boundary     |    10 | 1.198× | 1.198/1.197   |
| final          | arm/bitmap    | baseline/gated      | multi_ordered      |    10 | 1.117× | 1.111/1.121   |
| final          | arm/bitmap    | baseline/gated      | multi_skewed       |    10 | 1.015× | 1.021/1.006   |
| final          | arm/bitmap    | baseline/gated      | multi_wide         |    10 | 1.120× | 1.120/1.119   |
| final          | arm/bitmap    | baseline/gated      | or_high_high       |    10 | 1.005× | 1.003/1.007   |
| final          | arm/bitmap    | baseline/gated      | or_high_low        |    10 | 1.008× | 1.017/0.998   |
| final          | arm/bitmap    | baseline/gated      | or_high_med        |    10 | 0.996× | 0.999/0.995   |
| final          | arm/bitmap    | baseline/gated      | prefix3            |    10 | 0.995× | 0.993/0.991   |
| final          | arm/bitmap    | baseline/gated      | regex              |    10 | 0.975× | 0.976/0.976   |
| final          | arm/bitmap    | baseline/gated      | wildcard           |    10 | 0.985× | 0.998/0.940   |
| final          | arm/bitmap    | baseline/gated      | wildcard_scan      |    10 | 0.971× | 0.993/0.913   |
| final          | arm/bitmap    | baseline/gated      | multi_total        |   100 | 1.121× | 1.121/1.120   |
| final          | arm/bitmap    | baseline/gated      | multi_active       |   100 | 1.104× | 1.104/1.105   |
| final          | arm/bitmap    | baseline/gated      | and_high_high      |   100 | 0.972× | 0.981/0.957   |
| final          | arm/bitmap    | baseline/gated      | and_high_low       |   100 | 0.993× | 1.019/0.972   |
| final          | arm/bitmap    | baseline/gated      | and_high_med       |   100 | 0.995× | 0.994/0.988   |
| final          | arm/bitmap    | baseline/gated      | and_multi          |   100 | 1.081× | 1.084/1.093   |
| final          | arm/bitmap    | baseline/gated      | high_phrase        |   100 | 0.992× | 0.993/0.991   |
| final          | arm/bitmap    | baseline/gated      | high_sloppy_phrase |   100 | 0.996× | 1.007/0.970   |
| final          | arm/bitmap    | baseline/gated      | high_term          |   100 | 1.023× | 1.024/1.025   |
| final          | arm/bitmap    | baseline/gated      | low_phrase         |   100 | 1.001× | 1.014/0.990   |
| final          | arm/bitmap    | baseline/gated      | low_sloppy_phrase  |   100 | 1.004× | 1.013/0.998   |
| final          | arm/bitmap    | baseline/gated      | low_term           |   100 | 1.014× | 1.015/0.999   |
| final          | arm/bitmap    | baseline/gated      | med_phrase         |   100 | 0.992× | 0.997/0.989   |
| final          | arm/bitmap    | baseline/gated      | med_sloppy_phrase  |   100 | 1.003× | 1.006/0.996   |
| final          | arm/bitmap    | baseline/gated      | med_term           |   100 | 1.011× | 1.012/1.014   |
| final          | arm/bitmap    | baseline/gated      | multi_balanced     |   100 | 1.113× | 1.114/1.113   |
| final          | arm/bitmap    | baseline/gated      | multi_boundary     |   100 | 1.198× | 1.197/1.190   |
| final          | arm/bitmap    | baseline/gated      | multi_ordered      |   100 | 1.118× | 1.120/1.117   |
| final          | arm/bitmap    | baseline/gated      | multi_skewed       |   100 | 1.017× | 1.017/1.010   |
| final          | arm/bitmap    | baseline/gated      | multi_wide         |   100 | 1.123× | 1.121/1.127   |
| final          | arm/bitmap    | baseline/gated      | or_high_high       |   100 | 1.004× | 1.012/0.997   |
| final          | arm/bitmap    | baseline/gated      | or_high_low        |   100 | 1.014× | 1.022/1.006   |
| final          | arm/bitmap    | baseline/gated      | or_high_med        |   100 | 1.008× | 1.017/0.999   |
| final          | arm/bitmap    | baseline/gated      | prefix3            |   100 | 0.943× | 0.973/0.965   |
| final          | arm/bitmap    | baseline/gated      | regex              |   100 | 0.972× | 0.974/0.966   |
| final          | arm/bitmap    | baseline/gated      | wildcard           |   100 | 0.992× | 0.975/0.991   |
| final          | arm/bitmap    | baseline/gated      | wildcard_scan      |   100 | 0.936× | 0.946/0.930   |
| final          | arm/bitmap    | baseline/gated      | multi_total        |     0 | 1.009× | 1.008/1.010   |
| final          | arm/bitmap    | baseline/gated      | multi_active       |     0 | 1.010× | 1.009/1.010   |
| final          | arm/bitmap    | baseline/gated      | and_high_high      |     0 | 0.993× | 1.003/0.954   |
| final          | arm/bitmap    | baseline/gated      | and_high_low       |     0 | 1.002× | 1.003/1.003   |
| final          | arm/bitmap    | baseline/gated      | and_high_med       |     0 | 0.988× | 1.003/0.975   |
| final          | arm/bitmap    | baseline/gated      | and_multi          |     0 | 1.018× | 1.019/1.014   |
| final          | arm/bitmap    | baseline/gated      | high_phrase        |     0 | 1.011× | 1.025/1.001   |
| final          | arm/bitmap    | baseline/gated      | high_sloppy_phrase |     0 | 1.012× | 1.025/1.005   |
| final          | arm/bitmap    | baseline/gated      | high_term          |     0 | 0.974× | 1.014/0.949   |
| final          | arm/bitmap    | baseline/gated      | low_phrase         |     0 | 1.005× | 1.005/1.004   |
| final          | arm/bitmap    | baseline/gated      | low_sloppy_phrase  |     0 | 1.002× | 1.010/1.000   |
| final          | arm/bitmap    | baseline/gated      | low_term           |     0 | 1.035× | 1.027/1.043   |
| final          | arm/bitmap    | baseline/gated      | med_phrase         |     0 | 1.011× | 1.014/1.006   |
| final          | arm/bitmap    | baseline/gated      | med_sloppy_phrase  |     0 | 1.005× | 1.015/1.002   |
| final          | arm/bitmap    | baseline/gated      | med_term           |     0 | 1.023× | 1.015/1.031   |
| final          | arm/bitmap    | baseline/gated      | multi_balanced     |     0 | 1.009× | 1.010/1.007   |
| final          | arm/bitmap    | baseline/gated      | multi_boundary     |     0 | 1.004× | 1.001/1.005   |
| final          | arm/bitmap    | baseline/gated      | multi_ordered      |     0 | 1.009× | 1.011/1.008   |
| final          | arm/bitmap    | baseline/gated      | multi_skewed       |     0 | 0.998× | 0.997/1.000   |
| final          | arm/bitmap    | baseline/gated      | multi_wide         |     0 | 1.019× | 1.014/1.021   |
| final          | arm/bitmap    | baseline/gated      | or_high_high       |     0 | 0.977× | 1.000/0.888   |
| final          | arm/bitmap    | baseline/gated      | or_high_low        |     0 | 0.986× | 1.008/0.966   |
| final          | arm/bitmap    | baseline/gated      | or_high_med        |     0 | 0.981× | 1.002/0.947   |
| final          | arm/bitmap    | baseline/gated      | prefix3            |     0 | 1.014× | 1.011/1.021   |
| final          | arm/bitmap    | baseline/gated      | regex              |     0 | 0.992× | 1.001/0.987   |
| final          | arm/bitmap    | baseline/gated      | wildcard           |     0 | 1.014× | 1.022/1.005   |
| final          | arm/bitmap    | baseline/gated      | wildcard_scan      |     0 | 1.059× | 1.060/1.059   |
| final          | arm/rgb       | baseline/gated      | multi_total        |    10 | 1.126× | 1.113/1.136   |
| final          | arm/rgb       | baseline/gated      | multi_active       |    10 | 1.108× | 1.091/1.119   |
| final          | arm/rgb       | baseline/gated      | and_high_high      |    10 | 0.974× | 0.968/0.990   |
| final          | arm/rgb       | baseline/gated      | and_high_low       |    10 | 0.840× | 0.460/0.992   |
| final          | arm/rgb       | baseline/gated      | and_high_med       |    10 | 0.893× | 0.540/0.978   |
| final          | arm/rgb       | baseline/gated      | and_multi          |    10 | 1.091× | 1.084/1.099   |
| final          | arm/rgb       | baseline/gated      | high_phrase        |    10 | 0.960× | 0.898/1.012   |
| final          | arm/rgb       | baseline/gated      | high_sloppy_phrase |    10 | 1.003× | 0.989/1.015   |
| final          | arm/rgb       | baseline/gated      | high_term          |    10 | 1.141× | 1.130/1.117   |
| final          | arm/rgb       | baseline/gated      | low_phrase         |    10 | 1.003× | 1.015/1.010   |
| final          | arm/rgb       | baseline/gated      | low_sloppy_phrase  |    10 | 0.995× | 0.997/0.993   |
| final          | arm/rgb       | baseline/gated      | low_term           |    10 | 0.997× | 0.980/0.997   |
| final          | arm/rgb       | baseline/gated      | med_phrase         |    10 | 0.986× | 0.972/1.003   |
| final          | arm/rgb       | baseline/gated      | med_sloppy_phrase  |    10 | 0.999× | 0.992/0.997   |
| final          | arm/rgb       | baseline/gated      | med_term           |    10 | 1.060× | 1.038/1.125   |
| final          | arm/rgb       | baseline/gated      | multi_balanced     |    10 | 1.122× | 1.102/1.138   |
| final          | arm/rgb       | baseline/gated      | multi_boundary     |    10 | 1.208× | 1.210/1.207   |
| final          | arm/rgb       | baseline/gated      | multi_ordered      |    10 | 1.115× | 1.106/1.123   |
| final          | arm/rgb       | baseline/gated      | multi_skewed       |    10 | 1.001× | 0.953/1.033   |
| final          | arm/rgb       | baseline/gated      | multi_wide         |    10 | 1.129× | 1.128/1.130   |
| final          | arm/rgb       | baseline/gated      | or_high_high       |    10 | 0.929× | 0.796/0.997   |
| final          | arm/rgb       | baseline/gated      | or_high_low        |    10 | 0.993× | 0.872/1.030   |
| final          | arm/rgb       | baseline/gated      | or_high_med        |    10 | 0.913× | 0.663/0.968   |
| final          | arm/rgb       | baseline/gated      | prefix3            |    10 | 0.992× | 0.986/1.135   |
| final          | arm/rgb       | baseline/gated      | regex              |    10 | 0.969× | 0.943/0.988   |
| final          | arm/rgb       | baseline/gated      | wildcard           |    10 | 0.945× | 0.953/0.935   |
| final          | arm/rgb       | baseline/gated      | wildcard_scan      |    10 | 0.934× | 0.892/0.987   |
| final          | arm/rgb       | baseline/gated      | multi_total        |   100 | 1.140× | 1.139/1.147   |
| final          | arm/rgb       | baseline/gated      | multi_active       |   100 | 1.123× | 1.121/1.132   |
| final          | arm/rgb       | baseline/gated      | and_high_high      |   100 | 1.000× | 1.013/0.986   |
| final          | arm/rgb       | baseline/gated      | and_high_low       |   100 | 1.009× | 1.008/1.007   |
| final          | arm/rgb       | baseline/gated      | and_high_med       |   100 | 0.986× | 0.998/0.976   |
| final          | arm/rgb       | baseline/gated      | and_multi          |   100 | 1.078× | 1.021/1.171   |
| final          | arm/rgb       | baseline/gated      | high_phrase        |   100 | 0.994× | 0.992/0.991   |
| final          | arm/rgb       | baseline/gated      | high_sloppy_phrase |   100 | 1.015× | 1.026/1.017   |
| final          | arm/rgb       | baseline/gated      | high_term          |   100 | 1.006× | 0.999/1.007   |
| final          | arm/rgb       | baseline/gated      | low_phrase         |   100 | 1.006× | 1.036/0.992   |
| final          | arm/rgb       | baseline/gated      | low_sloppy_phrase  |   100 | 1.014× | 0.999/1.030   |
| final          | arm/rgb       | baseline/gated      | low_term           |   100 | 1.009× | 0.982/1.015   |
| final          | arm/rgb       | baseline/gated      | med_phrase         |   100 | 1.002× | 1.015/1.002   |
| final          | arm/rgb       | baseline/gated      | med_sloppy_phrase  |   100 | 1.050× | 1.085/1.009   |
| final          | arm/rgb       | baseline/gated      | med_term           |   100 | 1.023× | 1.019/1.025   |
| final          | arm/rgb       | baseline/gated      | multi_balanced     |   100 | 1.134× | 1.106/1.167   |
| final          | arm/rgb       | baseline/gated      | multi_boundary     |   100 | 1.216× | 1.219/1.212   |
| final          | arm/rgb       | baseline/gated      | multi_ordered      |   100 | 1.129× | 1.137/1.124   |
| final          | arm/rgb       | baseline/gated      | multi_skewed       |   100 | 1.026× | 1.034/1.024   |
| final          | arm/rgb       | baseline/gated      | multi_wide         |   100 | 1.145× | 1.161/1.138   |
| final          | arm/rgb       | baseline/gated      | or_high_high       |   100 | 0.981× | 0.986/0.981   |
| final          | arm/rgb       | baseline/gated      | or_high_low        |   100 | 0.993× | 0.990/0.986   |
| final          | arm/rgb       | baseline/gated      | or_high_med        |   100 | 1.006× | 1.015/0.959   |
| final          | arm/rgb       | baseline/gated      | prefix3            |   100 | 0.967× | 0.944/0.961   |
| final          | arm/rgb       | baseline/gated      | regex              |   100 | 0.974× | 0.966/0.984   |
| final          | arm/rgb       | baseline/gated      | wildcard           |   100 | 0.972× | 0.956/0.964   |
| final          | arm/rgb       | baseline/gated      | wildcard_scan      |   100 | 0.951× | 0.946/0.943   |
| final          | arm/rgb       | baseline/gated      | multi_total        |     0 | 1.025× | 1.040/1.010   |
| final          | arm/rgb       | baseline/gated      | multi_active       |     0 | 1.025× | 1.041/1.010   |
| final          | arm/rgb       | baseline/gated      | and_high_high      |     0 | 1.000× | 1.001/1.002   |
| final          | arm/rgb       | baseline/gated      | and_high_low       |     0 | 0.984× | 0.992/0.969   |
| final          | arm/rgb       | baseline/gated      | and_high_med       |     0 | 0.997× | 0.997/0.995   |
| final          | arm/rgb       | baseline/gated      | and_multi          |     0 | 1.000× | 1.004/0.994   |
| final          | arm/rgb       | baseline/gated      | high_phrase        |     0 | 1.006× | 1.002/1.010   |
| final          | arm/rgb       | baseline/gated      | high_sloppy_phrase |     0 | 1.004× | 1.003/1.005   |
| final          | arm/rgb       | baseline/gated      | high_term          |     0 | 1.006× | 1.012/1.000   |
| final          | arm/rgb       | baseline/gated      | low_phrase         |     0 | 1.003× | 1.003/1.005   |
| final          | arm/rgb       | baseline/gated      | low_sloppy_phrase  |     0 | 0.951× | 0.999/0.879   |
| final          | arm/rgb       | baseline/gated      | low_term           |     0 | 1.000× | 0.983/0.968   |
| final          | arm/rgb       | baseline/gated      | med_phrase         |     0 | 1.011× | 1.002/1.041   |
| final          | arm/rgb       | baseline/gated      | med_sloppy_phrase  |     0 | 0.993× | 1.001/0.956   |
| final          | arm/rgb       | baseline/gated      | med_term           |     0 | 1.039× | 1.078/0.987   |
| final          | arm/rgb       | baseline/gated      | multi_balanced     |     0 | 1.014× | 1.018/1.001   |
| final          | arm/rgb       | baseline/gated      | multi_boundary     |     0 | 1.012× | 1.021/1.003   |
| final          | arm/rgb       | baseline/gated      | multi_ordered      |     0 | 1.038× | 1.046/1.026   |
| final          | arm/rgb       | baseline/gated      | multi_skewed       |     0 | 1.014× | 1.035/1.007   |
| final          | arm/rgb       | baseline/gated      | multi_wide         |     0 | 1.038× | 1.064/1.013   |
| final          | arm/rgb       | baseline/gated      | or_high_high       |     0 | 0.997× | 0.998/0.995   |
| final          | arm/rgb       | baseline/gated      | or_high_low        |     0 | 0.995× | 0.979/1.003   |
| final          | arm/rgb       | baseline/gated      | or_high_med        |     0 | 0.998× | 1.001/0.996   |
| final          | arm/rgb       | baseline/gated      | prefix3            |     0 | 0.989× | 1.005/0.967   |
| final          | arm/rgb       | baseline/gated      | regex              |     0 | 0.996× | 0.998/0.992   |
| final          | arm/rgb       | baseline/gated      | wildcard           |     0 | 1.012× | 1.006/1.006   |
| final          | arm/rgb       | baseline/gated      | wildcard_scan      |     0 | 1.029× | 1.059/0.992   |
| final          | x86/default   | baseline/gated      | multi_total        |    10 | 1.160× | 1.160/1.161   |
| final          | x86/default   | baseline/gated      | multi_active       |    10 | 1.125× | 1.124/1.127   |
| final          | x86/default   | baseline/gated      | and_high_high      |    10 | 1.003× | 1.005/1.002   |
| final          | x86/default   | baseline/gated      | and_high_low       |    10 | 0.994× | 0.995/0.992   |
| final          | x86/default   | baseline/gated      | and_high_med       |    10 | 0.997× | 0.995/0.999   |
| final          | x86/default   | baseline/gated      | and_multi          |    10 | 1.153× | 1.156/1.151   |
| final          | x86/default   | baseline/gated      | high_phrase        |    10 | 1.000× | 1.027/0.984   |
| final          | x86/default   | baseline/gated      | high_sloppy_phrase |    10 | 0.995× | 1.020/0.979   |
| final          | x86/default   | baseline/gated      | high_term          |    10 | 0.993× | 0.995/0.992   |
| final          | x86/default   | baseline/gated      | low_phrase         |    10 | 0.995× | 1.004/0.989   |
| final          | x86/default   | baseline/gated      | low_sloppy_phrase  |    10 | 0.997× | 1.003/0.990   |
| final          | x86/default   | baseline/gated      | low_term           |    10 | 0.992× | 0.996/0.988   |
| final          | x86/default   | baseline/gated      | med_phrase         |    10 | 1.002× | 1.002/1.003   |
| final          | x86/default   | baseline/gated      | med_sloppy_phrase  |    10 | 0.994× | 0.995/0.993   |
| final          | x86/default   | baseline/gated      | med_term           |    10 | 0.994× | 0.990/0.996   |
| final          | x86/default   | baseline/gated      | multi_balanced     |    10 | 1.142× | 1.144/1.142   |
| final          | x86/default   | baseline/gated      | multi_boundary     |    10 | 1.326× | 1.326/1.325   |
| final          | x86/default   | baseline/gated      | multi_ordered      |    10 | 1.154× | 1.153/1.157   |
| final          | x86/default   | baseline/gated      | multi_skewed       |    10 | 0.997× | 0.996/0.999   |
| final          | x86/default   | baseline/gated      | multi_wide         |    10 | 1.147× | 1.144/1.150   |
| final          | x86/default   | baseline/gated      | or_high_high       |    10 | 0.997× | 0.998/0.995   |
| final          | x86/default   | baseline/gated      | or_high_low        |    10 | 0.993× | 0.987/0.999   |
| final          | x86/default   | baseline/gated      | or_high_med        |    10 | 0.992× | 0.996/0.993   |
| final          | x86/default   | baseline/gated      | prefix3            |    10 | 1.004× | 1.003/1.005   |
| final          | x86/default   | baseline/gated      | regex              |    10 | 1.011× | 1.019/1.001   |
| final          | x86/default   | baseline/gated      | wildcard           |    10 | 1.001× | 1.003/0.998   |
| final          | x86/default   | baseline/gated      | wildcard_scan      |    10 | 1.004× | 1.008/0.997   |
| final          | x86/default   | baseline/wide-gated | multi_total        |    10 | 1.123× | 1.123/1.124   |
| final          | x86/default   | baseline/wide-gated | multi_active       |    10 | 1.104× | 1.104/1.106   |
| final          | x86/default   | baseline/wide-gated | and_high_high      |    10 | 0.981× | 0.987/0.969   |
| final          | x86/default   | baseline/wide-gated | and_high_low       |    10 | 0.995× | 0.993/0.997   |
| final          | x86/default   | baseline/wide-gated | and_high_med       |    10 | 1.000× | 1.002/0.998   |
| final          | x86/default   | baseline/wide-gated | and_multi          |    10 | 1.105× | 1.106/1.107   |
| final          | x86/default   | baseline/wide-gated | high_phrase        |    10 | 1.020× | 1.032/1.009   |
| final          | x86/default   | baseline/wide-gated | high_sloppy_phrase |    10 | 0.993× | 1.015/0.976   |
| final          | x86/default   | baseline/wide-gated | high_term          |    10 | 0.993× | 0.999/0.992   |
| final          | x86/default   | baseline/wide-gated | low_phrase         |    10 | 0.999× | 0.992/1.016   |
| final          | x86/default   | baseline/wide-gated | low_sloppy_phrase  |    10 | 0.996× | 0.996/0.994   |
| final          | x86/default   | baseline/wide-gated | low_term           |    10 | 0.988× | 0.986/0.991   |
| final          | x86/default   | baseline/wide-gated | med_phrase         |    10 | 0.999× | 0.998/0.999   |
| final          | x86/default   | baseline/wide-gated | med_sloppy_phrase  |    10 | 0.990× | 0.992/0.989   |
| final          | x86/default   | baseline/wide-gated | med_term           |    10 | 0.995× | 0.991/1.001   |
| final          | x86/default   | baseline/wide-gated | multi_balanced     |    10 | 1.133× | 1.132/1.136   |
| final          | x86/default   | baseline/wide-gated | multi_boundary     |    10 | 1.205× | 1.205/1.204   |
| final          | x86/default   | baseline/wide-gated | multi_ordered      |    10 | 1.111× | 1.112/1.111   |
| final          | x86/default   | baseline/wide-gated | multi_skewed       |    10 | 0.943× | 0.943/0.945   |
| final          | x86/default   | baseline/wide-gated | multi_wide         |    10 | 1.140× | 1.139/1.141   |
| final          | x86/default   | baseline/wide-gated | or_high_high       |    10 | 1.003× | 1.004/1.002   |
| final          | x86/default   | baseline/wide-gated | or_high_low        |    10 | 0.995× | 0.994/0.995   |
| final          | x86/default   | baseline/wide-gated | or_high_med        |    10 | 0.981× | 0.985/0.985   |
| final          | x86/default   | baseline/wide-gated | prefix3            |    10 | 1.000× | 1.000/1.001   |
| final          | x86/default   | baseline/wide-gated | regex              |    10 | 0.989× | 1.000/0.977   |
| final          | x86/default   | baseline/wide-gated | wildcard           |    10 | 1.002× | 1.002/1.002   |
| final          | x86/default   | baseline/wide-gated | wildcard_scan      |    10 | 0.993× | 1.001/0.984   |
| final          | x86/default   | gated/wide-gated    | multi_total        |    10 | 0.968× | 0.968/0.968   |
| final          | x86/default   | gated/wide-gated    | multi_active       |    10 | 0.981× | 0.982/0.982   |
| final          | x86/default   | gated/wide-gated    | and_high_high      |    10 | 0.979× | 0.983/0.967   |
| final          | x86/default   | gated/wide-gated    | and_high_low       |    10 | 1.001× | 0.998/1.005   |
| final          | x86/default   | gated/wide-gated    | and_high_med       |    10 | 1.003× | 1.007/0.999   |
| final          | x86/default   | gated/wide-gated    | and_multi          |    10 | 0.958× | 0.957/0.961   |
| final          | x86/default   | gated/wide-gated    | high_phrase        |    10 | 1.020× | 1.006/1.025   |
| final          | x86/default   | gated/wide-gated    | high_sloppy_phrase |    10 | 0.998× | 0.995/0.997   |
| final          | x86/default   | gated/wide-gated    | high_term          |    10 | 1.000× | 1.004/1.000   |
| final          | x86/default   | gated/wide-gated    | low_phrase         |    10 | 1.004× | 0.988/1.027   |
| final          | x86/default   | gated/wide-gated    | low_sloppy_phrase  |    10 | 0.998× | 0.993/1.004   |
| final          | x86/default   | gated/wide-gated    | low_term           |    10 | 0.996× | 0.990/1.004   |
| final          | x86/default   | gated/wide-gated    | med_phrase         |    10 | 0.996× | 0.996/0.996   |
| final          | x86/default   | gated/wide-gated    | med_sloppy_phrase  |    10 | 0.997× | 0.997/0.996   |
| final          | x86/default   | gated/wide-gated    | med_term           |    10 | 1.001× | 1.001/1.005   |
| final          | x86/default   | gated/wide-gated    | multi_balanced     |    10 | 0.992× | 0.989/0.994   |
| final          | x86/default   | gated/wide-gated    | multi_boundary     |    10 | 0.909× | 0.909/0.909   |
| final          | x86/default   | gated/wide-gated    | multi_ordered      |    10 | 0.963× | 0.965/0.960   |
| final          | x86/default   | gated/wide-gated    | multi_skewed       |    10 | 0.946× | 0.947/0.945   |
| final          | x86/default   | gated/wide-gated    | multi_wide         |    10 | 0.993× | 0.995/0.993   |
| final          | x86/default   | gated/wide-gated    | or_high_high       |    10 | 1.006× | 1.006/1.007   |
| final          | x86/default   | gated/wide-gated    | or_high_low        |    10 | 1.002× | 1.007/0.996   |
| final          | x86/default   | gated/wide-gated    | or_high_med        |    10 | 0.990× | 0.989/0.993   |
| final          | x86/default   | gated/wide-gated    | prefix3            |    10 | 0.996× | 0.997/0.995   |
| final          | x86/default   | gated/wide-gated    | regex              |    10 | 0.979× | 0.981/0.976   |
| final          | x86/default   | gated/wide-gated    | wildcard           |    10 | 1.001× | 0.999/1.004   |
| final          | x86/default   | gated/wide-gated    | wildcard_scan      |    10 | 0.988× | 0.992/0.987   |
| final          | x86/default   | baseline/gated      | multi_total        |   100 | 1.159× | 1.160/1.159   |
| final          | x86/default   | baseline/gated      | multi_active       |   100 | 1.122× | 1.122/1.121   |
| final          | x86/default   | baseline/gated      | and_high_high      |   100 | 1.003× | 1.004/1.002   |
| final          | x86/default   | baseline/gated      | and_high_low       |   100 | 0.998× | 1.004/0.987   |
| final          | x86/default   | baseline/gated      | and_high_med       |   100 | 0.997× | 1.001/0.994   |
| final          | x86/default   | baseline/gated      | and_multi          |   100 | 1.157× | 1.155/1.159   |
| final          | x86/default   | baseline/gated      | high_phrase        |   100 | 1.002× | 0.999/1.001   |
| final          | x86/default   | baseline/gated      | high_sloppy_phrase |   100 | 1.000× | 1.007/0.997   |
| final          | x86/default   | baseline/gated      | high_term          |   100 | 0.993× | 0.992/0.995   |
| final          | x86/default   | baseline/gated      | low_phrase         |   100 | 1.007× | 1.004/1.033   |
| final          | x86/default   | baseline/gated      | low_sloppy_phrase  |   100 | 0.995× | 0.990/0.998   |
| final          | x86/default   | baseline/gated      | low_term           |   100 | 0.964× | 0.976/0.952   |
| final          | x86/default   | baseline/gated      | med_phrase         |   100 | 1.005× | 1.006/1.003   |
| final          | x86/default   | baseline/gated      | med_sloppy_phrase  |   100 | 0.998× | 0.998/0.998   |
| final          | x86/default   | baseline/gated      | med_term           |   100 | 0.987× | 0.997/0.981   |
| final          | x86/default   | baseline/gated      | multi_balanced     |   100 | 1.140× | 1.139/1.138   |
| final          | x86/default   | baseline/gated      | multi_boundary     |   100 | 1.336× | 1.336/1.336   |
| final          | x86/default   | baseline/gated      | multi_ordered      |   100 | 1.150× | 1.149/1.151   |
| final          | x86/default   | baseline/gated      | multi_skewed       |   100 | 0.997× | 1.000/0.995   |
| final          | x86/default   | baseline/gated      | multi_wide         |   100 | 1.142× | 1.143/1.142   |
| final          | x86/default   | baseline/gated      | or_high_high       |   100 | 1.002× | 1.005/1.002   |
| final          | x86/default   | baseline/gated      | or_high_low        |   100 | 0.996× | 0.986/1.009   |
| final          | x86/default   | baseline/gated      | or_high_med        |   100 | 1.000× | 0.998/1.007   |
| final          | x86/default   | baseline/gated      | prefix3            |   100 | 1.004× | 1.020/1.002   |
| final          | x86/default   | baseline/gated      | regex              |   100 | 1.016× | 1.029/1.009   |
| final          | x86/default   | baseline/gated      | wildcard           |   100 | 1.011× | 1.006/1.018   |
| final          | x86/default   | baseline/gated      | wildcard_scan      |   100 | 1.011× | 1.003/1.018   |
| final          | x86/default   | baseline/wide-gated | multi_total        |   100 | 1.121× | 1.122/1.122   |
| final          | x86/default   | baseline/wide-gated | multi_active       |   100 | 1.101× | 1.101/1.101   |
| final          | x86/default   | baseline/wide-gated | and_high_high      |   100 | 0.985× | 0.986/0.981   |
| final          | x86/default   | baseline/wide-gated | and_high_low       |   100 | 0.999× | 1.000/0.998   |
| final          | x86/default   | baseline/wide-gated | and_high_med       |   100 | 0.993× | 0.995/0.992   |
| final          | x86/default   | baseline/wide-gated | and_multi          |   100 | 1.108× | 1.107/1.108   |
| final          | x86/default   | baseline/wide-gated | high_phrase        |   100 | 0.995× | 0.992/0.995   |
| final          | x86/default   | baseline/wide-gated | high_sloppy_phrase |   100 | 0.993× | 1.000/0.988   |
| final          | x86/default   | baseline/wide-gated | high_term          |   100 | 0.995× | 0.995/0.991   |
| final          | x86/default   | baseline/wide-gated | low_phrase         |   100 | 0.991× | 0.994/1.018   |
| final          | x86/default   | baseline/wide-gated | low_sloppy_phrase  |   100 | 0.992× | 0.986/0.997   |
| final          | x86/default   | baseline/wide-gated | low_term           |   100 | 0.986× | 0.997/0.974   |
| final          | x86/default   | baseline/wide-gated | med_phrase         |   100 | 1.002× | 1.006/0.999   |
| final          | x86/default   | baseline/wide-gated | med_sloppy_phrase  |   100 | 0.987× | 0.990/0.988   |
| final          | x86/default   | baseline/wide-gated | med_term           |   100 | 1.003× | 1.019/0.992   |
| final          | x86/default   | baseline/wide-gated | multi_balanced     |   100 | 1.135× | 1.134/1.136   |
| final          | x86/default   | baseline/wide-gated | multi_boundary     |   100 | 1.211× | 1.211/1.211   |
| final          | x86/default   | baseline/wide-gated | multi_ordered      |   100 | 1.107× | 1.106/1.107   |
| final          | x86/default   | baseline/wide-gated | multi_skewed       |   100 | 0.942× | 0.942/0.943   |
| final          | x86/default   | baseline/wide-gated | multi_wide         |   100 | 1.130× | 1.132/1.130   |
| final          | x86/default   | baseline/wide-gated | or_high_high       |   100 | 0.997× | 1.002/0.992   |
| final          | x86/default   | baseline/wide-gated | or_high_low        |   100 | 0.995× | 0.989/0.999   |
| final          | x86/default   | baseline/wide-gated | or_high_med        |   100 | 1.001× | 1.002/0.990   |
| final          | x86/default   | baseline/wide-gated | prefix3            |   100 | 1.001× | 1.005/1.000   |
| final          | x86/default   | baseline/wide-gated | regex              |   100 | 0.999× | 1.018/0.984   |
| final          | x86/default   | baseline/wide-gated | wildcard           |   100 | 0.996× | 0.995/1.002   |
| final          | x86/default   | baseline/wide-gated | wildcard_scan      |   100 | 0.998× | 0.990/1.008   |
| final          | x86/default   | gated/wide-gated    | multi_total        |   100 | 0.967× | 0.967/0.968   |
| final          | x86/default   | gated/wide-gated    | multi_active       |   100 | 0.981× | 0.981/0.982   |
| final          | x86/default   | gated/wide-gated    | and_high_high      |   100 | 0.982× | 0.982/0.979   |
| final          | x86/default   | gated/wide-gated    | and_high_low       |   100 | 1.001× | 0.996/1.012   |
| final          | x86/default   | gated/wide-gated    | and_high_med       |   100 | 0.996× | 0.994/0.998   |
| final          | x86/default   | gated/wide-gated    | and_multi          |   100 | 0.957× | 0.958/0.956   |
| final          | x86/default   | gated/wide-gated    | high_phrase        |   100 | 0.993× | 0.993/0.994   |
| final          | x86/default   | gated/wide-gated    | high_sloppy_phrase |   100 | 0.993× | 0.993/0.991   |
| final          | x86/default   | gated/wide-gated    | high_term          |   100 | 1.002× | 1.003/0.996   |
| final          | x86/default   | gated/wide-gated    | low_phrase         |   100 | 0.984× | 0.990/0.985   |
| final          | x86/default   | gated/wide-gated    | low_sloppy_phrase  |   100 | 0.997× | 0.996/0.999   |
| final          | x86/default   | gated/wide-gated    | low_term           |   100 | 1.023× | 1.021/1.023   |
| final          | x86/default   | gated/wide-gated    | med_phrase         |   100 | 0.997× | 1.001/0.996   |
| final          | x86/default   | gated/wide-gated    | med_sloppy_phrase  |   100 | 0.989× | 0.991/0.990   |
| final          | x86/default   | gated/wide-gated    | med_term           |   100 | 1.017× | 1.022/1.011   |
| final          | x86/default   | gated/wide-gated    | multi_balanced     |   100 | 0.996× | 0.996/0.998   |
| final          | x86/default   | gated/wide-gated    | multi_boundary     |   100 | 0.906× | 0.907/0.906   |
| final          | x86/default   | gated/wide-gated    | multi_ordered      |   100 | 0.962× | 0.963/0.962   |
| final          | x86/default   | gated/wide-gated    | multi_skewed       |   100 | 0.944× | 0.942/0.947   |
| final          | x86/default   | gated/wide-gated    | multi_wide         |   100 | 0.989× | 0.990/0.989   |
| final          | x86/default   | gated/wide-gated    | or_high_high       |   100 | 0.994× | 0.997/0.990   |
| final          | x86/default   | gated/wide-gated    | or_high_low        |   100 | 0.998× | 1.004/0.990   |
| final          | x86/default   | gated/wide-gated    | or_high_med        |   100 | 1.000× | 1.005/0.984   |
| final          | x86/default   | gated/wide-gated    | prefix3            |   100 | 0.997× | 0.984/0.998   |
| final          | x86/default   | gated/wide-gated    | regex              |   100 | 0.983× | 0.990/0.976   |
| final          | x86/default   | gated/wide-gated    | wildcard           |   100 | 0.986× | 0.989/0.984   |
| final          | x86/default   | gated/wide-gated    | wildcard_scan      |   100 | 0.987× | 0.987/0.990   |
| final          | x86/default   | baseline/gated      | multi_total        |     0 | 1.003× | 0.993/1.007   |
| final          | x86/default   | baseline/gated      | multi_active       |     0 | 1.002× | 0.992/1.005   |
| final          | x86/default   | baseline/gated      | and_high_high      |     0 | 0.988× | 0.989/0.992   |
| final          | x86/default   | baseline/gated      | and_high_low       |     0 | 1.005× | 1.014/0.995   |
| final          | x86/default   | baseline/gated      | and_high_med       |     0 | 0.992× | 0.987/0.999   |
| final          | x86/default   | baseline/gated      | and_multi          |     0 | 1.000× | 1.001/0.973   |
| final          | x86/default   | baseline/gated      | high_phrase        |     0 | 1.014× | 1.025/1.001   |
| final          | x86/default   | baseline/gated      | high_sloppy_phrase |     0 | 1.004× | 1.006/0.998   |
| final          | x86/default   | baseline/gated      | high_term          |     0 | 1.006× | 1.006/0.997   |
| final          | x86/default   | baseline/gated      | low_phrase         |     0 | 1.009× | 1.022/0.995   |
| final          | x86/default   | baseline/gated      | low_sloppy_phrase  |     0 | 0.988× | 0.989/0.987   |
| final          | x86/default   | baseline/gated      | low_term           |     0 | 1.021× | 1.028/1.014   |
| final          | x86/default   | baseline/gated      | med_phrase         |     0 | 1.019× | 1.033/1.004   |
| final          | x86/default   | baseline/gated      | med_sloppy_phrase  |     0 | 0.997× | 0.998/0.991   |
| final          | x86/default   | baseline/gated      | med_term           |     0 | 1.004× | 1.021/0.993   |
| final          | x86/default   | baseline/gated      | multi_balanced     |     0 | 0.992× | 0.983/0.989   |
| final          | x86/default   | baseline/gated      | multi_boundary     |     0 | 1.021× | 1.020/1.052   |
| final          | x86/default   | baseline/gated      | multi_ordered      |     0 | 1.005× | 0.972/1.026   |
| final          | x86/default   | baseline/gated      | multi_skewed       |     0 | 1.016× | 1.009/1.013   |
| final          | x86/default   | baseline/gated      | multi_wide         |     0 | 1.002× | 0.999/1.002   |
| final          | x86/default   | baseline/gated      | or_high_high       |     0 | 0.985× | 0.985/0.984   |
| final          | x86/default   | baseline/gated      | or_high_low        |     0 | 1.009× | 1.008/1.006   |
| final          | x86/default   | baseline/gated      | or_high_med        |     0 | 0.991× | 0.986/0.994   |
| final          | x86/default   | baseline/gated      | prefix3            |     0 | 1.001× | 1.010/0.994   |
| final          | x86/default   | baseline/gated      | regex              |     0 | 0.977× | 0.975/0.975   |
| final          | x86/default   | baseline/gated      | wildcard           |     0 | 1.000× | 0.998/1.001   |
| final          | x86/default   | baseline/gated      | wildcard_scan      |     0 | 1.008× | 0.999/1.014   |
| final          | x86/default   | baseline/wide-gated | multi_total        |     0 | 0.998× | 0.994/0.998   |
| final          | x86/default   | baseline/wide-gated | multi_active       |     0 | 0.998× | 0.994/0.997   |
| final          | x86/default   | baseline/wide-gated | and_high_high      |     0 | 1.005× | 0.977/1.019   |
| final          | x86/default   | baseline/wide-gated | and_high_low       |     0 | 1.011× | 1.015/1.001   |
| final          | x86/default   | baseline/wide-gated | and_high_med       |     0 | 1.005× | 1.006/1.007   |
| final          | x86/default   | baseline/wide-gated | and_multi          |     0 | 1.002× | 1.022/0.972   |
| final          | x86/default   | baseline/wide-gated | high_phrase        |     0 | 0.966× | 0.977/0.954   |
| final          | x86/default   | baseline/wide-gated | high_sloppy_phrase |     0 | 0.984× | 0.990/0.969   |
| final          | x86/default   | baseline/wide-gated | high_term          |     0 | 0.982× | 0.997/0.964   |
| final          | x86/default   | baseline/wide-gated | low_phrase         |     0 | 1.012× | 1.025/0.998   |
| final          | x86/default   | baseline/wide-gated | low_sloppy_phrase  |     0 | 1.003× | 1.000/1.003   |
| final          | x86/default   | baseline/wide-gated | low_term           |     0 | 0.980× | 0.976/1.000   |
| final          | x86/default   | baseline/wide-gated | med_phrase         |     0 | 0.993× | 1.003/0.981   |
| final          | x86/default   | baseline/wide-gated | med_sloppy_phrase  |     0 | 1.003× | 1.003/0.999   |
| final          | x86/default   | baseline/wide-gated | med_term           |     0 | 0.997× | 1.014/0.973   |
| final          | x86/default   | baseline/wide-gated | multi_balanced     |     0 | 0.996× | 0.981/1.001   |
| final          | x86/default   | baseline/wide-gated | multi_boundary     |     0 | 1.008× | 0.990/1.027   |
| final          | x86/default   | baseline/wide-gated | multi_ordered      |     0 | 1.008× | 1.021/1.001   |
| final          | x86/default   | baseline/wide-gated | multi_skewed       |     0 | 1.007× | 0.996/1.007   |
| final          | x86/default   | baseline/wide-gated | multi_wide         |     0 | 0.988× | 0.992/0.985   |
| final          | x86/default   | baseline/wide-gated | or_high_high       |     0 | 1.009× | 1.007/1.013   |
| final          | x86/default   | baseline/wide-gated | or_high_low        |     0 | 1.009× | 1.008/1.003   |
| final          | x86/default   | baseline/wide-gated | or_high_med        |     0 | 1.002× | 0.996/1.008   |
| final          | x86/default   | baseline/wide-gated | prefix3            |     0 | 1.001× | 1.004/0.998   |
| final          | x86/default   | baseline/wide-gated | regex              |     0 | 0.985× | 0.979/0.995   |
| final          | x86/default   | baseline/wide-gated | wildcard           |     0 | 0.989× | 0.979/1.004   |
| final          | x86/default   | baseline/wide-gated | wildcard_scan      |     0 | 0.980× | 0.949/1.004   |
| final          | x86/default   | gated/wide-gated    | multi_total        |     0 | 0.995× | 1.001/0.991   |
| final          | x86/default   | gated/wide-gated    | multi_active       |     0 | 0.995× | 1.002/0.992   |
| final          | x86/default   | gated/wide-gated    | and_high_high      |     0 | 1.017× | 0.988/1.027   |
| final          | x86/default   | gated/wide-gated    | and_high_low       |     0 | 1.006× | 1.001/1.006   |
| final          | x86/default   | gated/wide-gated    | and_high_med       |     0 | 1.013× | 1.019/1.007   |
| final          | x86/default   | gated/wide-gated    | and_multi          |     0 | 1.002× | 1.021/0.999   |
| final          | x86/default   | gated/wide-gated    | high_phrase        |     0 | 0.953× | 0.953/0.953   |
| final          | x86/default   | gated/wide-gated    | high_sloppy_phrase |     0 | 0.980× | 0.984/0.971   |
| final          | x86/default   | gated/wide-gated    | high_term          |     0 | 0.976× | 0.992/0.967   |
| final          | x86/default   | gated/wide-gated    | low_phrase         |     0 | 1.003× | 1.002/1.003   |
| final          | x86/default   | gated/wide-gated    | low_sloppy_phrase  |     0 | 1.015× | 1.011/1.017   |
| final          | x86/default   | gated/wide-gated    | low_term           |     0 | 0.961× | 0.949/0.985   |
| final          | x86/default   | gated/wide-gated    | med_phrase         |     0 | 0.974× | 0.972/0.978   |
| final          | x86/default   | gated/wide-gated    | med_sloppy_phrase  |     0 | 1.006× | 1.005/1.008   |
| final          | x86/default   | gated/wide-gated    | med_term           |     0 | 0.993× | 0.994/0.979   |
| final          | x86/default   | gated/wide-gated    | multi_balanced     |     0 | 1.004× | 0.998/1.012   |
| final          | x86/default   | gated/wide-gated    | multi_boundary     |     0 | 0.987× | 0.970/0.976   |
| final          | x86/default   | gated/wide-gated    | multi_ordered      |     0 | 1.004× | 1.050/0.975   |
| final          | x86/default   | gated/wide-gated    | multi_skewed       |     0 | 0.991× | 0.987/0.994   |
| final          | x86/default   | gated/wide-gated    | multi_wide         |     0 | 0.986× | 0.994/0.983   |
| final          | x86/default   | gated/wide-gated    | or_high_high       |     0 | 1.024× | 1.022/1.029   |
| final          | x86/default   | gated/wide-gated    | or_high_low        |     0 | 1.000× | 1.000/0.997   |
| final          | x86/default   | gated/wide-gated    | or_high_med        |     0 | 1.011× | 1.011/1.014   |
| final          | x86/default   | gated/wide-gated    | prefix3            |     0 | 0.999× | 0.994/1.004   |
| final          | x86/default   | gated/wide-gated    | regex              |     0 | 1.008× | 1.004/1.021   |
| final          | x86/default   | gated/wide-gated    | wildcard           |     0 | 0.990× | 0.981/1.003   |
| final          | x86/default   | gated/wide-gated    | wildcard_scan      |     0 | 0.973× | 0.950/0.991   |
| final          | x86/rgb-pairs | baseline/gated      | multi_total        |    10 | 1.197× | 1.195/1.199   |
| final          | x86/rgb-pairs | baseline/gated      | multi_active       |    10 | 1.160× | 1.159/1.161   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_high      |    10 | 1.007× | 1.010/1.006   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_low       |    10 | 0.994× | 0.987/0.994   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_med       |    10 | 0.997× | 1.000/0.993   |
| final          | x86/rgb-pairs | baseline/gated      | and_multi          |    10 | 1.150× | 1.147/1.152   |
| final          | x86/rgb-pairs | baseline/gated      | high_phrase        |    10 | 0.996× | 0.987/1.006   |
| final          | x86/rgb-pairs | baseline/gated      | high_sloppy_phrase |    10 | 1.045× | 1.028/1.046   |
| final          | x86/rgb-pairs | baseline/gated      | high_term          |    10 | 1.011× | 1.010/1.015   |
| final          | x86/rgb-pairs | baseline/gated      | low_phrase         |    10 | 1.026× | 1.027/1.018   |
| final          | x86/rgb-pairs | baseline/gated      | low_sloppy_phrase  |    10 | 1.004× | 1.003/1.007   |
| final          | x86/rgb-pairs | baseline/gated      | low_term           |    10 | 1.002× | 0.999/1.004   |
| final          | x86/rgb-pairs | baseline/gated      | med_phrase         |    10 | 0.997× | 1.001/1.002   |
| final          | x86/rgb-pairs | baseline/gated      | med_sloppy_phrase  |    10 | 1.002× | 0.999/1.003   |
| final          | x86/rgb-pairs | baseline/gated      | med_term           |    10 | 0.997× | 0.997/0.997   |
| final          | x86/rgb-pairs | baseline/gated      | multi_balanced     |    10 | 1.168× | 1.169/1.167   |
| final          | x86/rgb-pairs | baseline/gated      | multi_boundary     |    10 | 1.364× | 1.356/1.369   |
| final          | x86/rgb-pairs | baseline/gated      | multi_ordered      |    10 | 1.179× | 1.174/1.185   |
| final          | x86/rgb-pairs | baseline/gated      | multi_skewed       |    10 | 1.017× | 1.017/1.021   |
| final          | x86/rgb-pairs | baseline/gated      | multi_wide         |    10 | 1.198× | 1.197/1.200   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_high       |    10 | 1.011× | 1.012/1.009   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_low        |    10 | 0.995× | 0.989/1.001   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_med        |    10 | 0.997× | 0.998/0.997   |
| final          | x86/rgb-pairs | baseline/gated      | prefix3            |    10 | 1.007× | 1.024/1.002   |
| final          | x86/rgb-pairs | baseline/gated      | regex              |    10 | 1.018× | 1.025/1.007   |
| final          | x86/rgb-pairs | baseline/gated      | wildcard           |    10 | 1.004× | 1.003/1.005   |
| final          | x86/rgb-pairs | baseline/gated      | wildcard_scan      |    10 | 1.003× | 1.004/0.998   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_total        |    10 | 1.147× | 1.144/1.150   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_active       |    10 | 1.126× | 1.124/1.128   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_high      |    10 | 1.000× | 0.992/1.002   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_low       |    10 | 1.006× | 1.007/1.005   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_med       |    10 | 0.996× | 0.999/0.989   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_multi          |    10 | 1.126× | 1.119/1.135   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_phrase        |    10 | 0.993× | 0.995/0.991   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_sloppy_phrase |    10 | 1.031× | 1.005/1.037   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_term          |    10 | 0.987× | 0.987/0.986   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_phrase         |    10 | 1.009× | 1.026/0.981   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_sloppy_phrase  |    10 | 1.004× | 0.999/1.011   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_term           |    10 | 0.994× | 0.992/0.995   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_phrase         |    10 | 0.997× | 0.997/1.007   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_sloppy_phrase  |    10 | 0.998× | 0.997/0.998   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_term           |    10 | 0.991× | 0.989/0.993   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_balanced     |    10 | 1.145× | 1.145/1.145   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_boundary     |    10 | 1.236× | 1.229/1.242   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_ordered      |    10 | 1.119× | 1.112/1.130   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_skewed       |    10 | 0.963× | 0.963/0.966   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_wide         |    10 | 1.173× | 1.170/1.175   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_high       |    10 | 1.007× | 1.012/0.996   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_low        |    10 | 0.993× | 0.987/1.003   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_med        |    10 | 0.994× | 1.004/0.979   |
| final          | x86/rgb-pairs | baseline/wide-gated | prefix3            |    10 | 0.998× | 0.999/0.996   |
| final          | x86/rgb-pairs | baseline/wide-gated | regex              |    10 | 1.013× | 1.033/1.009   |
| final          | x86/rgb-pairs | baseline/wide-gated | wildcard           |    10 | 0.997× | 0.998/0.992   |
| final          | x86/rgb-pairs | baseline/wide-gated | wildcard_scan      |    10 | 0.995× | 0.997/0.991   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_total        |    10 | 0.958× | 0.957/0.959   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_active       |    10 | 0.970× | 0.969/0.972   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_high      |    10 | 0.993× | 0.981/0.996   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_low       |    10 | 1.012× | 1.020/1.011   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_med       |    10 | 0.999× | 0.999/0.996   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_multi          |    10 | 0.979× | 0.975/0.986   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_phrase        |    10 | 0.997× | 1.009/0.985   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_sloppy_phrase |    10 | 0.987× | 0.978/0.991   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_term          |    10 | 0.976× | 0.978/0.971   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_phrase         |    10 | 0.983× | 0.999/0.963   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_sloppy_phrase  |    10 | 1.000× | 0.997/1.004   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_term           |    10 | 0.992× | 0.993/0.991   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_phrase         |    10 | 1.000× | 0.997/1.005   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_sloppy_phrase  |    10 | 0.996× | 0.997/0.995   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_term           |    10 | 0.993× | 0.992/0.996   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_balanced     |    10 | 0.980× | 0.979/0.981   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_boundary     |    10 | 0.906× | 0.906/0.907   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_ordered      |    10 | 0.949× | 0.947/0.953   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_skewed       |    10 | 0.947× | 0.947/0.946   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_wide         |    10 | 0.979× | 0.977/0.979   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_high       |    10 | 0.996× | 1.000/0.987   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_low        |    10 | 0.998× | 0.998/1.003   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_med        |    10 | 0.997× | 1.007/0.982   |
| final          | x86/rgb-pairs | gated/wide-gated    | prefix3            |    10 | 0.991× | 0.975/0.994   |
| final          | x86/rgb-pairs | gated/wide-gated    | regex              |    10 | 0.996× | 1.009/1.002   |
| final          | x86/rgb-pairs | gated/wide-gated    | wildcard           |    10 | 0.992× | 0.995/0.987   |
| final          | x86/rgb-pairs | gated/wide-gated    | wildcard_scan      |    10 | 0.992× | 0.993/0.993   |
| final          | x86/rgb-pairs | baseline/gated      | multi_total        |   100 | 1.200× | 1.199/1.200   |
| final          | x86/rgb-pairs | baseline/gated      | multi_active       |   100 | 1.165× | 1.164/1.166   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_high      |   100 | 1.003× | 0.999/1.006   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_low       |   100 | 0.995× | 1.003/0.988   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_med       |   100 | 0.994× | 0.996/0.993   |
| final          | x86/rgb-pairs | baseline/gated      | and_multi          |   100 | 1.193× | 1.194/1.192   |
| final          | x86/rgb-pairs | baseline/gated      | high_phrase        |   100 | 0.994× | 0.991/0.997   |
| final          | x86/rgb-pairs | baseline/gated      | high_sloppy_phrase |   100 | 0.989× | 0.988/0.989   |
| final          | x86/rgb-pairs | baseline/gated      | high_term          |   100 | 0.991× | 0.983/0.996   |
| final          | x86/rgb-pairs | baseline/gated      | low_phrase         |   100 | 0.994× | 0.984/1.005   |
| final          | x86/rgb-pairs | baseline/gated      | low_sloppy_phrase  |   100 | 0.998× | 0.998/0.999   |
| final          | x86/rgb-pairs | baseline/gated      | low_term           |   100 | 1.001× | 0.979/1.021   |
| final          | x86/rgb-pairs | baseline/gated      | med_phrase         |   100 | 0.998× | 0.998/0.991   |
| final          | x86/rgb-pairs | baseline/gated      | med_sloppy_phrase  |   100 | 1.003× | 1.002/1.004   |
| final          | x86/rgb-pairs | baseline/gated      | med_term           |   100 | 0.999× | 0.985/1.011   |
| final          | x86/rgb-pairs | baseline/gated      | multi_balanced     |   100 | 1.168× | 1.169/1.168   |
| final          | x86/rgb-pairs | baseline/gated      | multi_boundary     |   100 | 1.357× | 1.358/1.356   |
| final          | x86/rgb-pairs | baseline/gated      | multi_ordered      |   100 | 1.180× | 1.181/1.180   |
| final          | x86/rgb-pairs | baseline/gated      | multi_skewed       |   100 | 1.017× | 1.013/1.020   |
| final          | x86/rgb-pairs | baseline/gated      | multi_wide         |   100 | 1.212× | 1.211/1.214   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_high       |   100 | 1.001× | 1.001/1.001   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_low        |   100 | 0.990× | 0.989/0.992   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_med        |   100 | 0.989× | 0.983/0.997   |
| final          | x86/rgb-pairs | baseline/gated      | prefix3            |   100 | 1.002× | 0.985/1.013   |
| final          | x86/rgb-pairs | baseline/gated      | regex              |   100 | 1.012× | 1.009/1.017   |
| final          | x86/rgb-pairs | baseline/gated      | wildcard           |   100 | 1.005× | 1.007/1.002   |
| final          | x86/rgb-pairs | baseline/gated      | wildcard_scan      |   100 | 1.002× | 1.002/1.002   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_total        |   100 | 1.149× | 1.152/1.146   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_active       |   100 | 1.132× | 1.132/1.131   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_high      |   100 | 0.999× | 1.001/0.994   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_low       |   100 | 1.005× | 1.010/1.002   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_med       |   100 | 1.004× | 1.004/1.003   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_multi          |   100 | 1.159× | 1.158/1.161   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_phrase        |   100 | 0.996× | 0.992/0.997   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_sloppy_phrase |   100 | 0.983× | 0.990/0.977   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_term          |   100 | 0.982× | 0.984/0.981   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_phrase         |   100 | 0.982× | 0.977/0.992   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_sloppy_phrase  |   100 | 1.001× | 1.002/1.001   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_term           |   100 | 0.990× | 0.988/0.996   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_phrase         |   100 | 0.991× | 0.980/1.001   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_sloppy_phrase  |   100 | 0.995× | 0.997/0.990   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_term           |   100 | 0.996× | 1.003/0.987   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_balanced     |   100 | 1.146× | 1.146/1.146   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_boundary     |   100 | 1.224× | 1.233/1.210   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_ordered      |   100 | 1.123× | 1.127/1.116   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_skewed       |   100 | 0.965× | 0.965/0.964   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_wide         |   100 | 1.187× | 1.187/1.187   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_high       |   100 | 1.001× | 1.002/1.001   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_low        |   100 | 0.995× | 0.996/0.994   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_med        |   100 | 0.992× | 0.991/0.989   |
| final          | x86/rgb-pairs | baseline/wide-gated | prefix3            |   100 | 1.002× | 0.983/1.013   |
| final          | x86/rgb-pairs | baseline/wide-gated | regex              |   100 | 1.015× | 1.008/1.024   |
| final          | x86/rgb-pairs | baseline/wide-gated | wildcard           |   100 | 0.993× | 0.999/0.992   |
| final          | x86/rgb-pairs | baseline/wide-gated | wildcard_scan      |   100 | 0.996× | 0.998/0.996   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_total        |   100 | 0.958× | 0.960/0.955   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_active       |   100 | 0.971× | 0.973/0.970   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_high      |   100 | 0.995× | 1.001/0.988   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_low       |   100 | 1.009× | 1.007/1.015   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_med       |   100 | 1.010× | 1.008/1.010   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_multi          |   100 | 0.972× | 0.970/0.974   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_phrase        |   100 | 1.001× | 1.000/1.000   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_sloppy_phrase |   100 | 0.994× | 1.002/0.988   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_term          |   100 | 0.991× | 1.001/0.985   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_phrase         |   100 | 0.988× | 0.993/0.987   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_sloppy_phrase  |   100 | 1.003× | 1.004/1.001   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_term           |   100 | 0.989× | 1.009/0.975   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_phrase         |   100 | 0.994× | 0.982/1.009   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_sloppy_phrase  |   100 | 0.992× | 0.995/0.986   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_term           |   100 | 0.996× | 1.018/0.977   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_balanced     |   100 | 0.981× | 0.980/0.981   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_boundary     |   100 | 0.902× | 0.908/0.893   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_ordered      |   100 | 0.951× | 0.954/0.946   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_skewed       |   100 | 0.948× | 0.952/0.945   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_wide         |   100 | 0.979× | 0.980/0.978   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_high       |   100 | 1.000× | 1.001/1.000   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_low        |   100 | 1.005× | 1.007/1.001   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_med        |   100 | 1.003× | 1.008/0.992   |
| final          | x86/rgb-pairs | gated/wide-gated    | prefix3            |   100 | 0.999× | 0.999/1.000   |
| final          | x86/rgb-pairs | gated/wide-gated    | regex              |   100 | 1.003× | 0.998/1.007   |
| final          | x86/rgb-pairs | gated/wide-gated    | wildcard           |   100 | 0.989× | 0.991/0.989   |
| final          | x86/rgb-pairs | gated/wide-gated    | wildcard_scan      |   100 | 0.994× | 0.996/0.994   |
| final          | x86/rgb-pairs | baseline/gated      | multi_total        |     0 | 1.005× | 1.009/1.004   |
| final          | x86/rgb-pairs | baseline/gated      | multi_active       |     0 | 1.007× | 1.007/1.006   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_high      |     0 | 0.994× | 1.001/0.987   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_low       |     0 | 1.002× | 1.011/0.994   |
| final          | x86/rgb-pairs | baseline/gated      | and_high_med       |     0 | 1.003× | 1.006/1.000   |
| final          | x86/rgb-pairs | baseline/gated      | and_multi          |     0 | 1.016× | 1.016/1.032   |
| final          | x86/rgb-pairs | baseline/gated      | high_phrase        |     0 | 1.016× | 1.000/1.028   |
| final          | x86/rgb-pairs | baseline/gated      | high_sloppy_phrase |     0 | 0.993× | 0.997/0.993   |
| final          | x86/rgb-pairs | baseline/gated      | high_term          |     0 | 1.015× | 0.997/1.019   |
| final          | x86/rgb-pairs | baseline/gated      | low_phrase         |     0 | 1.011× | 1.005/1.019   |
| final          | x86/rgb-pairs | baseline/gated      | low_sloppy_phrase  |     0 | 0.990× | 1.003/0.982   |
| final          | x86/rgb-pairs | baseline/gated      | low_term           |     0 | 1.043× | 1.075/1.010   |
| final          | x86/rgb-pairs | baseline/gated      | med_phrase         |     0 | 1.016× | 1.014/1.017   |
| final          | x86/rgb-pairs | baseline/gated      | med_sloppy_phrase  |     0 | 0.998× | 1.002/0.994   |
| final          | x86/rgb-pairs | baseline/gated      | med_term           |     0 | 0.999× | 1.021/0.992   |
| final          | x86/rgb-pairs | baseline/gated      | multi_balanced     |     0 | 1.007× | 1.005/1.009   |
| final          | x86/rgb-pairs | baseline/gated      | multi_boundary     |     0 | 0.980× | 1.045/0.958   |
| final          | x86/rgb-pairs | baseline/gated      | multi_ordered      |     0 | 1.014× | 1.018/1.014   |
| final          | x86/rgb-pairs | baseline/gated      | multi_skewed       |     0 | 1.018× | 1.020/1.012   |
| final          | x86/rgb-pairs | baseline/gated      | multi_wide         |     0 | 0.996× | 0.995/0.997   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_high       |     0 | 0.989× | 0.995/0.983   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_low        |     0 | 1.021× | 1.017/1.018   |
| final          | x86/rgb-pairs | baseline/gated      | or_high_med        |     0 | 0.992× | 0.998/0.986   |
| final          | x86/rgb-pairs | baseline/gated      | prefix3            |     0 | 1.008× | 1.011/1.003   |
| final          | x86/rgb-pairs | baseline/gated      | regex              |     0 | 0.998× | 1.004/0.992   |
| final          | x86/rgb-pairs | baseline/gated      | wildcard           |     0 | 0.987× | 0.996/0.977   |
| final          | x86/rgb-pairs | baseline/gated      | wildcard_scan      |     0 | 1.001× | 1.000/1.001   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_total        |     0 | 1.004× | 1.002/1.005   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_active       |     0 | 1.005× | 1.004/1.006   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_high      |     0 | 1.006× | 1.013/1.001   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_low       |     0 | 0.985× | 1.006/0.964   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_high_med       |     0 | 1.003× | 0.988/0.999   |
| final          | x86/rgb-pairs | baseline/wide-gated | and_multi          |     0 | 1.006× | 1.016/0.996   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_phrase        |     0 | 1.009× | 1.015/0.997   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_sloppy_phrase |     0 | 0.965× | 0.969/0.965   |
| final          | x86/rgb-pairs | baseline/wide-gated | high_term          |     0 | 1.028× | 1.006/1.050   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_phrase         |     0 | 1.006× | 1.010/1.006   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_sloppy_phrase  |     0 | 0.990× | 0.993/0.981   |
| final          | x86/rgb-pairs | baseline/wide-gated | low_term           |     0 | 1.045× | 1.095/0.997   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_phrase         |     0 | 0.995× | 1.003/0.986   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_sloppy_phrase  |     0 | 0.985× | 0.989/0.985   |
| final          | x86/rgb-pairs | baseline/wide-gated | med_term           |     0 | 1.009× | 1.018/0.996   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_balanced     |     0 | 1.005× | 0.997/1.010   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_boundary     |     0 | 0.978× | 0.973/0.994   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_ordered      |     0 | 1.019× | 1.014/1.017   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_skewed       |     0 | 1.005× | 1.014/0.997   |
| final          | x86/rgb-pairs | baseline/wide-gated | multi_wide         |     0 | 0.999× | 0.997/1.004   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_high       |     0 | 1.009× | 1.013/1.004   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_low        |     0 | 1.011× | 1.019/1.008   |
| final          | x86/rgb-pairs | baseline/wide-gated | or_high_med        |     0 | 0.999× | 1.004/0.993   |
| final          | x86/rgb-pairs | baseline/wide-gated | prefix3            |     0 | 1.003× | 1.007/1.001   |
| final          | x86/rgb-pairs | baseline/wide-gated | regex              |     0 | 0.985× | 0.997/0.976   |
| final          | x86/rgb-pairs | baseline/wide-gated | wildcard           |     0 | 0.960× | 0.938/0.989   |
| final          | x86/rgb-pairs | baseline/wide-gated | wildcard_scan      |     0 | 0.993× | 0.988/1.001   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_total        |     0 | 0.998× | 0.993/1.002   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_active       |     0 | 0.998× | 0.997/0.999   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_high      |     0 | 1.012× | 1.012/1.014   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_low       |     0 | 0.983× | 0.994/0.970   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_high_med       |     0 | 1.000× | 0.983/0.998   |
| final          | x86/rgb-pairs | gated/wide-gated    | and_multi          |     0 | 0.990× | 1.000/0.965   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_phrase        |     0 | 0.993× | 1.015/0.970   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_sloppy_phrase |     0 | 0.972× | 0.972/0.971   |
| final          | x86/rgb-pairs | gated/wide-gated    | high_term          |     0 | 1.014× | 1.009/1.031   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_phrase         |     0 | 0.995× | 1.005/0.987   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_sloppy_phrase  |     0 | 1.000× | 0.991/0.998   |
| final          | x86/rgb-pairs | gated/wide-gated    | low_term           |     0 | 1.001× | 1.019/0.988   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_phrase         |     0 | 0.980× | 0.989/0.969   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_sloppy_phrase  |     0 | 0.987× | 0.988/0.991   |
| final          | x86/rgb-pairs | gated/wide-gated    | med_term           |     0 | 1.009× | 0.998/1.004   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_balanced     |     0 | 0.998× | 0.992/1.002   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_boundary     |     0 | 0.999× | 0.932/1.038   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_ordered      |     0 | 1.005× | 0.996/1.002   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_skewed       |     0 | 0.987× | 0.994/0.985   |
| final          | x86/rgb-pairs | gated/wide-gated    | multi_wide         |     0 | 1.003× | 1.002/1.007   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_high       |     0 | 1.020× | 1.018/1.021   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_low        |     0 | 0.991× | 1.001/0.990   |
| final          | x86/rgb-pairs | gated/wide-gated    | or_high_med        |     0 | 1.007× | 1.007/1.007   |
| final          | x86/rgb-pairs | gated/wide-gated    | prefix3            |     0 | 0.996× | 0.996/0.998   |
| final          | x86/rgb-pairs | gated/wide-gated    | regex              |     0 | 0.988× | 0.993/0.983   |
| final          | x86/rgb-pairs | gated/wide-gated    | wildcard           |     0 | 0.972× | 0.941/1.012   |
| final          | x86/rgb-pairs | gated/wide-gated    | wildcard_scan      |     0 | 0.993× | 0.988/1.000   |
| confirm        | arm/bitmap    | baseline/gated      | multi_total        |    10 | 1.204× | 1.299/1.140   |
| confirm        | arm/bitmap    | baseline/gated      | multi_active       |    10 | 1.220× | 1.309/1.150   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_high      |    10 | 1.034× | 1.068/1.003   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_low       |    10 | 0.942× | 1.152/0.907   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_med       |    10 | 0.946× | 0.962/0.932   |
| confirm        | arm/bitmap    | baseline/gated      | and_multi          |    10 | 1.192× | 1.180/1.189   |
| confirm        | arm/bitmap    | baseline/gated      | high_phrase        |    10 | 1.005× | 1.013/0.998   |
| confirm        | arm/bitmap    | baseline/gated      | high_sloppy_phrase |    10 | 1.028× | 1.074/0.999   |
| confirm        | arm/bitmap    | baseline/gated      | high_term          |    10 | 1.051× | 1.159/0.978   |
| confirm        | arm/bitmap    | baseline/gated      | low_phrase         |    10 | 1.011× | 1.025/1.023   |
| confirm        | arm/bitmap    | baseline/gated      | low_sloppy_phrase  |    10 | 1.085× | 1.029/1.128   |
| confirm        | arm/bitmap    | baseline/gated      | low_term           |    10 | 1.118× | 1.115/1.097   |
| confirm        | arm/bitmap    | baseline/gated      | med_phrase         |    10 | 1.012× | 1.005/0.994   |
| confirm        | arm/bitmap    | baseline/gated      | med_sloppy_phrase  |    10 | 1.026× | 1.030/1.042   |
| confirm        | arm/bitmap    | baseline/gated      | med_term           |    10 | 1.104× | 1.082/1.080   |
| confirm        | arm/bitmap    | baseline/gated      | multi_balanced     |    10 | 1.327× | 1.476/1.291   |
| confirm        | arm/bitmap    | baseline/gated      | multi_boundary     |    10 | 1.131× | 1.254/1.099   |
| confirm        | arm/bitmap    | baseline/gated      | multi_ordered      |    10 | 1.214× | 1.203/1.135   |
| confirm        | arm/bitmap    | baseline/gated      | multi_skewed       |    10 | 1.072× | 1.200/0.954   |
| confirm        | arm/bitmap    | baseline/gated      | multi_wide         |    10 | 1.185× | 1.245/1.112   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_high       |    10 | 1.008× | 1.022/0.997   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_low        |    10 | 1.001× | 0.986/1.007   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_med        |    10 | 1.000× | 0.994/1.013   |
| confirm        | arm/bitmap    | baseline/gated      | prefix3            |    10 | 1.452× | 1.049/2.011   |
| confirm        | arm/bitmap    | baseline/gated      | regex              |    10 | 1.074× | 0.969/1.164   |
| confirm        | arm/bitmap    | baseline/gated      | wildcard           |    10 | 1.110× | 0.945/1.378   |
| confirm        | arm/bitmap    | baseline/gated      | wildcard_scan      |    10 | 1.087× | 0.963/1.228   |
| confirm        | arm/bitmap    | baseline/gated      | multi_total        |   100 | 1.339× | 1.687/1.076   |
| confirm        | arm/bitmap    | baseline/gated      | multi_active       |   100 | 1.321× | 1.637/1.083   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_high      |   100 | 0.955× | 1.088/0.812   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_low       |   100 | 0.949× | 1.147/0.862   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_med       |   100 | 0.934× | 1.051/0.867   |
| confirm        | arm/bitmap    | baseline/gated      | and_multi          |   100 | 1.082× | 1.285/0.939   |
| confirm        | arm/bitmap    | baseline/gated      | high_phrase        |   100 | 1.099× | 1.126/1.075   |
| confirm        | arm/bitmap    | baseline/gated      | high_sloppy_phrase |   100 | 1.260× | 1.630/0.897   |
| confirm        | arm/bitmap    | baseline/gated      | high_term          |   100 | 1.054× | 1.164/0.955   |
| confirm        | arm/bitmap    | baseline/gated      | low_phrase         |   100 | 1.323× | 1.854/1.061   |
| confirm        | arm/bitmap    | baseline/gated      | low_sloppy_phrase  |   100 | 1.053× | 1.379/0.878   |
| confirm        | arm/bitmap    | baseline/gated      | low_term           |   100 | 0.976× | 1.342/0.858   |
| confirm        | arm/bitmap    | baseline/gated      | med_phrase         |   100 | 1.184× | 1.354/1.109   |
| confirm        | arm/bitmap    | baseline/gated      | med_sloppy_phrase  |   100 | 1.152× | 1.382/1.032   |
| confirm        | arm/bitmap    | baseline/gated      | med_term           |   100 | 0.955× | 1.136/0.838   |
| confirm        | arm/bitmap    | baseline/gated      | multi_balanced     |   100 | 1.220× | 1.475/1.031   |
| confirm        | arm/bitmap    | baseline/gated      | multi_boundary     |   100 | 1.411× | 1.883/1.042   |
| confirm        | arm/bitmap    | baseline/gated      | multi_ordered      |   100 | 1.342× | 2.088/0.828   |
| confirm        | arm/bitmap    | baseline/gated      | multi_skewed       |   100 | 1.061× | 1.315/0.914   |
| confirm        | arm/bitmap    | baseline/gated      | multi_wide         |   100 | 1.523× | 1.774/1.335   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_high       |   100 | 1.022× | 1.193/0.920   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_low        |   100 | 1.064× | 1.162/0.972   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_med        |   100 | 1.036× | 1.092/0.979   |
| confirm        | arm/bitmap    | baseline/gated      | prefix3            |   100 | 1.069× | 1.163/0.919   |
| confirm        | arm/bitmap    | baseline/gated      | regex              |   100 | 1.133× | 1.703/0.867   |
| confirm        | arm/bitmap    | baseline/gated      | wildcard           |   100 | 1.095× | 1.530/0.925   |
| confirm        | arm/bitmap    | baseline/gated      | wildcard_scan      |   100 | 1.043× | 1.390/0.877   |
| confirm        | arm/bitmap    | baseline/gated      | multi_total        |     0 | 1.008× | 0.861/1.151   |
| confirm        | arm/bitmap    | baseline/gated      | multi_active       |     0 | 1.008× | 0.860/1.152   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_high      |     0 | 1.012× | 1.380/0.740   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_low       |     0 | 1.016× | 1.245/0.770   |
| confirm        | arm/bitmap    | baseline/gated      | and_high_med       |     0 | 1.004× | 1.381/0.642   |
| confirm        | arm/bitmap    | baseline/gated      | and_multi          |     0 | 0.992× | 0.871/1.291   |
| confirm        | arm/bitmap    | baseline/gated      | high_phrase        |     0 | 0.977× | 1.241/0.772   |
| confirm        | arm/bitmap    | baseline/gated      | high_sloppy_phrase |     0 | 0.895× | 0.959/0.798   |
| confirm        | arm/bitmap    | baseline/gated      | high_term          |     0 | 0.995× | 1.224/0.825   |
| confirm        | arm/bitmap    | baseline/gated      | low_phrase         |     0 | 1.032× | 1.157/0.901   |
| confirm        | arm/bitmap    | baseline/gated      | low_sloppy_phrase  |     0 | 0.932× | 0.871/0.989   |
| confirm        | arm/bitmap    | baseline/gated      | low_term           |     0 | 1.033× | 1.260/0.838   |
| confirm        | arm/bitmap    | baseline/gated      | med_phrase         |     0 | 0.994× | 1.083/0.929   |
| confirm        | arm/bitmap    | baseline/gated      | med_sloppy_phrase  |     0 | 0.944× | 0.913/0.973   |
| confirm        | arm/bitmap    | baseline/gated      | med_term           |     0 | 1.035× | 1.196/0.815   |
| confirm        | arm/bitmap    | baseline/gated      | multi_balanced     |     0 | 0.988× | 0.868/1.123   |
| confirm        | arm/bitmap    | baseline/gated      | multi_boundary     |     0 | 0.996× | 0.868/1.134   |
| confirm        | arm/bitmap    | baseline/gated      | multi_ordered      |     0 | 0.981× | 0.811/1.164   |
| confirm        | arm/bitmap    | baseline/gated      | multi_skewed       |     0 | 0.993× | 0.860/1.107   |
| confirm        | arm/bitmap    | baseline/gated      | multi_wide         |     0 | 1.052× | 0.880/1.208   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_high       |     0 | 1.017× | 1.356/0.747   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_low        |     0 | 1.051× | 1.381/0.757   |
| confirm        | arm/bitmap    | baseline/gated      | or_high_med        |     0 | 0.969× | 1.290/0.762   |
| confirm        | arm/bitmap    | baseline/gated      | prefix3            |     0 | 0.921× | 0.866/0.998   |
| confirm        | arm/bitmap    | baseline/gated      | regex              |     0 | 0.889× | 0.728/1.023   |
| confirm        | arm/bitmap    | baseline/gated      | wildcard           |     0 | 0.997× | 0.919/0.981   |
| confirm        | arm/bitmap    | baseline/gated      | wildcard_scan      |     0 | 1.037× | 0.893/1.185   |
| confirm        | arm/rgb       | baseline/gated      | multi_total        |    10 | 0.980× | 1.205/0.772   |
| confirm        | arm/rgb       | baseline/gated      | multi_active       |    10 | 0.924× | 1.185/0.705   |
| confirm        | arm/rgb       | baseline/gated      | and_high_high      |    10 | 0.863× | 0.726/1.022   |
| confirm        | arm/rgb       | baseline/gated      | and_high_low       |    10 | 0.752× | 0.471/0.989   |
| confirm        | arm/rgb       | baseline/gated      | and_high_med       |    10 | 0.881× | 0.730/1.017   |
| confirm        | arm/rgb       | baseline/gated      | and_multi          |    10 | 0.766× | 0.995/0.597   |
| confirm        | arm/rgb       | baseline/gated      | high_phrase        |    10 | 0.844× | 0.788/0.942   |
| confirm        | arm/rgb       | baseline/gated      | high_sloppy_phrase |    10 | 0.938× | 0.938/0.913   |
| confirm        | arm/rgb       | baseline/gated      | high_term          |    10 | 0.787× | 0.584/0.921   |
| confirm        | arm/rgb       | baseline/gated      | low_phrase         |    10 | 0.879× | 0.881/0.880   |
| confirm        | arm/rgb       | baseline/gated      | low_sloppy_phrase  |    10 | 0.737× | 0.900/0.609   |
| confirm        | arm/rgb       | baseline/gated      | low_term           |    10 | 0.745× | 0.470/1.008   |
| confirm        | arm/rgb       | baseline/gated      | med_phrase         |    10 | 0.910× | 0.911/0.935   |
| confirm        | arm/rgb       | baseline/gated      | med_sloppy_phrase  |    10 | 0.883× | 0.965/0.815   |
| confirm        | arm/rgb       | baseline/gated      | med_term           |    10 | 0.832× | 0.656/0.993   |
| confirm        | arm/rgb       | baseline/gated      | multi_balanced     |    10 | 0.843× | 1.069/0.650   |
| confirm        | arm/rgb       | baseline/gated      | multi_boundary     |    10 | 1.298× | 1.292/1.243   |
| confirm        | arm/rgb       | baseline/gated      | multi_ordered      |    10 | 1.208× | 1.388/1.066   |
| confirm        | arm/rgb       | baseline/gated      | multi_skewed       |    10 | 0.853× | 1.207/0.570   |
| confirm        | arm/rgb       | baseline/gated      | multi_wide         |    10 | 0.934× | 1.213/0.719   |
| confirm        | arm/rgb       | baseline/gated      | or_high_high       |    10 | 0.885× | 0.765/1.010   |
| confirm        | arm/rgb       | baseline/gated      | or_high_low        |    10 | 0.787× | 0.622/0.886   |
| confirm        | arm/rgb       | baseline/gated      | or_high_med        |    10 | 0.813× | 0.689/0.955   |
| confirm        | arm/rgb       | baseline/gated      | prefix3            |    10 | 0.779× | 0.906/0.714   |
| confirm        | arm/rgb       | baseline/gated      | regex              |    10 | 0.774× | 0.877/0.691   |
| confirm        | arm/rgb       | baseline/gated      | wildcard           |    10 | 0.753× | 0.788/0.773   |
| confirm        | arm/rgb       | baseline/gated      | wildcard_scan      |    10 | 0.736× | 0.877/0.605   |
| confirm        | arm/rgb       | baseline/gated      | multi_total        |   100 | 1.057× | 1.056/1.046   |
| confirm        | arm/rgb       | baseline/gated      | multi_active       |   100 | 1.088× | 1.102/1.059   |
| confirm        | arm/rgb       | baseline/gated      | and_high_high      |   100 | 0.986× | 0.920/1.042   |
| confirm        | arm/rgb       | baseline/gated      | and_high_low       |   100 | 0.988× | 0.878/1.180   |
| confirm        | arm/rgb       | baseline/gated      | and_high_med       |   100 | 0.978× | 0.875/1.098   |
| confirm        | arm/rgb       | baseline/gated      | and_multi          |   100 | 1.230× | 1.138/1.218   |
| confirm        | arm/rgb       | baseline/gated      | high_phrase        |   100 | 1.002× | 0.899/1.110   |
| confirm        | arm/rgb       | baseline/gated      | high_sloppy_phrase |   100 | 1.094× | 0.936/1.403   |
| confirm        | arm/rgb       | baseline/gated      | high_term          |   100 | 1.052× | 1.186/1.032   |
| confirm        | arm/rgb       | baseline/gated      | low_phrase         |   100 | 1.035× | 0.933/1.163   |
| confirm        | arm/rgb       | baseline/gated      | low_sloppy_phrase  |   100 | 1.120× | 1.015/1.214   |
| confirm        | arm/rgb       | baseline/gated      | low_term           |   100 | 1.010× | 1.086/1.299   |
| confirm        | arm/rgb       | baseline/gated      | med_phrase         |   100 | 0.995× | 0.871/1.175   |
| confirm        | arm/rgb       | baseline/gated      | med_sloppy_phrase  |   100 | 1.038× | 0.962/1.051   |
| confirm        | arm/rgb       | baseline/gated      | med_term           |   100 | 1.020× | 1.071/0.994   |
| confirm        | arm/rgb       | baseline/gated      | multi_balanced     |   100 | 1.164× | 1.131/1.156   |
| confirm        | arm/rgb       | baseline/gated      | multi_boundary     |   100 | 0.932× | 0.887/0.984   |
| confirm        | arm/rgb       | baseline/gated      | multi_ordered      |   100 | 0.925× | 0.867/0.984   |
| confirm        | arm/rgb       | baseline/gated      | multi_skewed       |   100 | 1.096× | 1.033/1.112   |
| confirm        | arm/rgb       | baseline/gated      | multi_wide         |   100 | 1.082× | 1.215/0.972   |
| confirm        | arm/rgb       | baseline/gated      | or_high_high       |   100 | 0.985× | 0.937/1.044   |
| confirm        | arm/rgb       | baseline/gated      | or_high_low        |   100 | 0.996× | 1.038/0.975   |
| confirm        | arm/rgb       | baseline/gated      | or_high_med        |   100 | 0.992× | 0.957/1.032   |
| confirm        | arm/rgb       | baseline/gated      | prefix3            |   100 | 0.948× | 0.943/1.013   |
| confirm        | arm/rgb       | baseline/gated      | regex              |   100 | 1.059× | 0.982/1.213   |
| confirm        | arm/rgb       | baseline/gated      | wildcard           |   100 | 1.174× | 1.027/1.476   |
| confirm        | arm/rgb       | baseline/gated      | wildcard_scan      |   100 | 1.073× | 0.961/1.424   |
| confirm        | arm/rgb       | baseline/gated      | multi_total        |     0 | 0.966× | 1.046/0.935   |
| confirm        | arm/rgb       | baseline/gated      | multi_active       |     0 | 0.967× | 1.049/0.934   |
| confirm        | arm/rgb       | baseline/gated      | and_high_high      |     0 | 0.685× | 0.630/0.714   |
| confirm        | arm/rgb       | baseline/gated      | and_high_low       |     0 | 0.759× | 0.745/0.754   |
| confirm        | arm/rgb       | baseline/gated      | and_high_med       |     0 | 0.651× | 0.608/0.663   |
| confirm        | arm/rgb       | baseline/gated      | and_multi          |     0 | 0.935× | 0.905/0.900   |
| confirm        | arm/rgb       | baseline/gated      | high_phrase        |     0 | 0.800× | 0.672/0.957   |
| confirm        | arm/rgb       | baseline/gated      | high_sloppy_phrase |     0 | 0.881× | 0.616/1.264   |
| confirm        | arm/rgb       | baseline/gated      | high_term          |     0 | 0.812× | 0.763/0.854   |
| confirm        | arm/rgb       | baseline/gated      | low_phrase         |     0 | 0.879× | 0.735/1.095   |
| confirm        | arm/rgb       | baseline/gated      | low_sloppy_phrase  |     0 | 0.906× | 0.883/0.929   |
| confirm        | arm/rgb       | baseline/gated      | low_term           |     0 | 0.813× | 0.753/0.871   |
| confirm        | arm/rgb       | baseline/gated      | med_phrase         |     0 | 0.800× | 0.645/1.019   |
| confirm        | arm/rgb       | baseline/gated      | med_sloppy_phrase  |     0 | 0.806× | 0.692/0.956   |
| confirm        | arm/rgb       | baseline/gated      | med_term           |     0 | 0.794× | 0.765/0.839   |
| confirm        | arm/rgb       | baseline/gated      | multi_balanced     |     0 | 0.937× | 0.943/0.908   |
| confirm        | arm/rgb       | baseline/gated      | multi_boundary     |     0 | 0.956× | 0.994/0.943   |
| confirm        | arm/rgb       | baseline/gated      | multi_ordered      |     0 | 1.008× | 1.031/0.991   |
| confirm        | arm/rgb       | baseline/gated      | multi_skewed       |     0 | 0.887× | 0.885/0.907   |
| confirm        | arm/rgb       | baseline/gated      | multi_wide         |     0 | 1.036× | 1.271/0.954   |
| confirm        | arm/rgb       | baseline/gated      | or_high_high       |     0 | 0.696× | 0.644/0.750   |
| confirm        | arm/rgb       | baseline/gated      | or_high_low        |     0 | 0.762× | 0.582/0.841   |
| confirm        | arm/rgb       | baseline/gated      | or_high_med        |     0 | 0.621× | 0.560/0.712   |
| confirm        | arm/rgb       | baseline/gated      | prefix3            |     0 | 0.925× | 0.928/0.907   |
| confirm        | arm/rgb       | baseline/gated      | regex              |     0 | 0.971× | 0.938/0.979   |
| confirm        | arm/rgb       | baseline/gated      | wildcard           |     0 | 0.911× | 0.857/0.928   |
| confirm        | arm/rgb       | baseline/gated      | wildcard_scan      |     0 | 0.939× | 0.871/0.989   |
| selected-final | x86/default   | baseline/gated      | multi_total        |    10 | 1.156× | 1.155/1.157   |
| selected-final | x86/default   | baseline/gated      | multi_active       |    10 | 1.121× | 1.119/1.122   |
| selected-final | x86/default   | baseline/gated      | and_high_high      |    10 | 0.996× | 0.990/1.001   |
| selected-final | x86/default   | baseline/gated      | and_high_low       |    10 | 1.003× | 0.997/1.008   |
| selected-final | x86/default   | baseline/gated      | and_high_med       |    10 | 0.993× | 0.987/1.000   |
| selected-final | x86/default   | baseline/gated      | and_multi          |    10 | 1.148× | 1.149/1.148   |
| selected-final | x86/default   | baseline/gated      | high_phrase        |    10 | 0.968× | 0.961/0.980   |
| selected-final | x86/default   | baseline/gated      | high_sloppy_phrase |    10 | 0.993× | 0.990/0.993   |
| selected-final | x86/default   | baseline/gated      | high_term          |    10 | 1.004× | 1.006/1.006   |
| selected-final | x86/default   | baseline/gated      | low_phrase         |    10 | 1.001× | 0.996/1.003   |
| selected-final | x86/default   | baseline/gated      | low_sloppy_phrase  |    10 | 0.996× | 0.991/1.000   |
| selected-final | x86/default   | baseline/gated      | low_term           |    10 | 1.001× | 1.004/1.004   |
| selected-final | x86/default   | baseline/gated      | med_phrase         |    10 | 0.999× | 0.997/0.999   |
| selected-final | x86/default   | baseline/gated      | med_sloppy_phrase  |    10 | 0.988× | 0.989/0.992   |
| selected-final | x86/default   | baseline/gated      | med_term           |    10 | 0.994× | 1.001/0.991   |
| selected-final | x86/default   | baseline/gated      | multi_balanced     |    10 | 1.136× | 1.133/1.138   |
| selected-final | x86/default   | baseline/gated      | multi_boundary     |    10 | 1.326× | 1.328/1.325   |
| selected-final | x86/default   | baseline/gated      | multi_ordered      |    10 | 1.148× | 1.148/1.146   |
| selected-final | x86/default   | baseline/gated      | multi_skewed       |    10 | 0.993× | 0.991/0.995   |
| selected-final | x86/default   | baseline/gated      | multi_wide         |    10 | 1.145× | 1.144/1.147   |
| selected-final | x86/default   | baseline/gated      | or_high_high       |    10 | 0.987× | 0.984/0.991   |
| selected-final | x86/default   | baseline/gated      | or_high_low        |    10 | 1.002× | 1.005/0.999   |
| selected-final | x86/default   | baseline/gated      | or_high_med        |    10 | 0.947× | 0.931/0.967   |
| selected-final | x86/default   | baseline/gated      | prefix3            |    10 | 1.001× | 1.000/1.004   |
| selected-final | x86/default   | baseline/gated      | regex              |    10 | 0.989× | 0.986/0.993   |
| selected-final | x86/default   | baseline/gated      | wildcard           |    10 | 1.005× | 1.004/1.009   |
| selected-final | x86/default   | baseline/gated      | wildcard_scan      |    10 | 1.002× | 0.998/0.999   |
| selected-final | x86/default   | baseline/gated      | multi_total        |   100 | 1.158× | 1.158/1.159   |
| selected-final | x86/default   | baseline/gated      | multi_active       |   100 | 1.121× | 1.121/1.122   |
| selected-final | x86/default   | baseline/gated      | and_high_high      |   100 | 1.001× | 1.004/1.000   |
| selected-final | x86/default   | baseline/gated      | and_high_low       |   100 | 1.008× | 1.004/1.013   |
| selected-final | x86/default   | baseline/gated      | and_high_med       |   100 | 0.994× | 0.991/0.997   |
| selected-final | x86/default   | baseline/gated      | and_multi          |   100 | 1.155× | 1.156/1.154   |
| selected-final | x86/default   | baseline/gated      | high_phrase        |   100 | 0.993× | 0.996/0.994   |
| selected-final | x86/default   | baseline/gated      | high_sloppy_phrase |   100 | 0.983× | 1.004/0.968   |
| selected-final | x86/default   | baseline/gated      | high_term          |   100 | 0.997× | 0.994/1.002   |
| selected-final | x86/default   | baseline/gated      | low_phrase         |   100 | 0.955× | 0.954/0.962   |
| selected-final | x86/default   | baseline/gated      | low_sloppy_phrase  |   100 | 0.985× | 0.989/0.984   |
| selected-final | x86/default   | baseline/gated      | low_term           |   100 | 1.031× | 1.025/1.026   |
| selected-final | x86/default   | baseline/gated      | med_phrase         |   100 | 1.005× | 1.004/1.006   |
| selected-final | x86/default   | baseline/gated      | med_sloppy_phrase  |   100 | 0.988× | 0.990/0.985   |
| selected-final | x86/default   | baseline/gated      | med_term           |   100 | 1.010× | 1.001/1.024   |
| selected-final | x86/default   | baseline/gated      | multi_balanced     |   100 | 1.138× | 1.139/1.139   |
| selected-final | x86/default   | baseline/gated      | multi_boundary     |   100 | 1.335× | 1.335/1.337   |
| selected-final | x86/default   | baseline/gated      | multi_ordered      |   100 | 1.147× | 1.146/1.152   |
| selected-final | x86/default   | baseline/gated      | multi_skewed       |   100 | 0.999× | 1.001/0.998   |
| selected-final | x86/default   | baseline/gated      | multi_wide         |   100 | 1.143× | 1.140/1.142   |
| selected-final | x86/default   | baseline/gated      | or_high_high       |   100 | 0.990× | 0.992/0.989   |
| selected-final | x86/default   | baseline/gated      | or_high_low        |   100 | 0.997× | 1.007/0.987   |
| selected-final | x86/default   | baseline/gated      | or_high_med        |   100 | 0.963× | 0.969/0.962   |
| selected-final | x86/default   | baseline/gated      | prefix3            |   100 | 1.003× | 1.003/1.005   |
| selected-final | x86/default   | baseline/gated      | regex              |   100 | 0.982× | 1.002/0.975   |
| selected-final | x86/default   | baseline/gated      | wildcard           |   100 | 1.009× | 1.003/1.004   |
| selected-final | x86/default   | baseline/gated      | wildcard_scan      |   100 | 1.002× | 1.008/0.997   |
| selected-final | x86/default   | baseline/gated      | multi_total        |     0 | 1.008× | 1.025/0.993   |
| selected-final | x86/default   | baseline/gated      | multi_active       |     0 | 1.007× | 1.025/0.993   |
| selected-final | x86/default   | baseline/gated      | and_high_high      |     0 | 0.981× | 0.974/0.984   |
| selected-final | x86/default   | baseline/gated      | and_high_low       |     0 | 1.016× | 1.024/1.012   |
| selected-final | x86/default   | baseline/gated      | and_high_med       |     0 | 1.003× | 1.004/0.999   |
| selected-final | x86/default   | baseline/gated      | and_multi          |     0 | 1.046× | 1.058/1.009   |
| selected-final | x86/default   | baseline/gated      | high_phrase        |     0 | 1.002× | 1.002/0.999   |
| selected-final | x86/default   | baseline/gated      | high_sloppy_phrase |     0 | 1.011× | 1.017/1.013   |
| selected-final | x86/default   | baseline/gated      | high_term          |     0 | 1.008× | 1.000/1.007   |
| selected-final | x86/default   | baseline/gated      | low_phrase         |     0 | 1.010× | 1.012/1.009   |
| selected-final | x86/default   | baseline/gated      | low_sloppy_phrase  |     0 | 0.995× | 0.995/0.990   |
| selected-final | x86/default   | baseline/gated      | low_term           |     0 | 1.001× | 0.972/1.015   |
| selected-final | x86/default   | baseline/gated      | med_phrase         |     0 | 1.009× | 1.004/1.015   |
| selected-final | x86/default   | baseline/gated      | med_sloppy_phrase  |     0 | 1.000× | 0.996/1.006   |
| selected-final | x86/default   | baseline/gated      | med_term           |     0 | 1.012× | 0.991/1.036   |
| selected-final | x86/default   | baseline/gated      | multi_balanced     |     0 | 1.005× | 1.010/1.001   |
| selected-final | x86/default   | baseline/gated      | multi_boundary     |     0 | 1.015× | 1.033/0.991   |
| selected-final | x86/default   | baseline/gated      | multi_ordered      |     0 | 1.004× | 1.019/0.988   |
| selected-final | x86/default   | baseline/gated      | multi_skewed       |     0 | 1.020× | 1.048/0.992   |
| selected-final | x86/default   | baseline/gated      | multi_wide         |     0 | 1.003× | 1.025/0.989   |
| selected-final | x86/default   | baseline/gated      | or_high_high       |     0 | 0.985× | 0.975/0.995   |
| selected-final | x86/default   | baseline/gated      | or_high_low        |     0 | 1.005× | 1.010/1.005   |
| selected-final | x86/default   | baseline/gated      | or_high_med        |     0 | 0.996× | 0.992/1.002   |
| selected-final | x86/default   | baseline/gated      | prefix3            |     0 | 0.995× | 0.987/1.006   |
| selected-final | x86/default   | baseline/gated      | regex              |     0 | 0.993× | 0.993/0.992   |
| selected-final | x86/default   | baseline/gated      | wildcard           |     0 | 0.997× | 1.009/0.984   |
| selected-final | x86/default   | baseline/gated      | wildcard_scan      |     0 | 1.001× | 1.008/0.996   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_total        |    10 | 1.193× | 1.194/1.193   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_active       |    10 | 1.158× | 1.159/1.157   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_high      |    10 | 1.006× | 1.011/1.002   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_low       |    10 | 0.997× | 1.002/0.992   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_med       |    10 | 1.002× | 1.005/0.997   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_multi          |    10 | 1.142× | 1.141/1.135   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_phrase        |    10 | 0.995× | 0.987/0.995   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_sloppy_phrase |    10 | 1.008× | 1.027/1.001   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_term          |    10 | 0.986× | 0.994/0.986   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_phrase         |    10 | 1.012× | 1.024/0.973   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_sloppy_phrase  |    10 | 1.015× | 1.016/1.012   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_term           |    10 | 1.005× | 1.000/1.013   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_phrase         |    10 | 0.997× | 0.986/1.003   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_sloppy_phrase  |    10 | 1.007× | 1.012/1.008   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_term           |    10 | 1.000× | 0.998/1.005   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_balanced     |    10 | 1.167× | 1.167/1.166   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_boundary     |    10 | 1.356× | 1.354/1.357   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_ordered      |    10 | 1.176× | 1.181/1.177   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_skewed       |    10 | 1.014× | 1.013/1.017   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_wide         |    10 | 1.196× | 1.197/1.195   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_high       |    10 | 1.005× | 1.007/1.003   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_low        |    10 | 0.999× | 1.007/0.988   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_med        |    10 | 1.004× | 1.003/1.004   |
| selected-final | x86/rgb-pairs | baseline/gated      | prefix3            |    10 | 1.005× | 1.004/1.009   |
| selected-final | x86/rgb-pairs | baseline/gated      | regex              |    10 | 1.001× | 0.998/1.008   |
| selected-final | x86/rgb-pairs | baseline/gated      | wildcard           |    10 | 1.007× | 1.007/1.006   |
| selected-final | x86/rgb-pairs | baseline/gated      | wildcard_scan      |    10 | 1.004× | 1.007/1.001   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_total        |   100 | 1.203× | 1.200/1.204   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_active       |   100 | 1.168× | 1.166/1.169   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_high      |   100 | 1.001× | 0.999/1.003   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_low       |   100 | 1.000× | 1.006/0.998   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_med       |   100 | 0.991× | 0.989/0.993   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_multi          |   100 | 1.199× | 1.189/1.201   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_phrase        |   100 | 0.997× | 1.002/0.993   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_sloppy_phrase |   100 | 0.995× | 0.982/1.012   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_term          |   100 | 0.996× | 0.995/0.999   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_phrase         |   100 | 1.048× | 1.138/0.966   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_sloppy_phrase  |   100 | 0.999× | 1.005/0.995   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_term           |   100 | 1.017× | 1.020/1.001   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_phrase         |   100 | 1.001× | 1.011/1.002   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_sloppy_phrase  |   100 | 1.001× | 0.997/1.009   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_term           |   100 | 0.995× | 0.998/0.991   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_balanced     |   100 | 1.170× | 1.167/1.174   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_boundary     |   100 | 1.364× | 1.358/1.366   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_ordered      |   100 | 1.185× | 1.180/1.187   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_skewed       |   100 | 1.023× | 1.021/1.024   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_wide         |   100 | 1.214× | 1.215/1.211   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_high       |   100 | 1.004× | 1.001/1.005   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_low        |   100 | 0.998× | 1.000/0.995   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_med        |   100 | 0.997× | 1.006/0.980   |
| selected-final | x86/rgb-pairs | baseline/gated      | prefix3            |   100 | 1.010× | 1.003/1.020   |
| selected-final | x86/rgb-pairs | baseline/gated      | regex              |   100 | 0.989× | 0.987/0.991   |
| selected-final | x86/rgb-pairs | baseline/gated      | wildcard           |   100 | 1.012× | 1.013/1.017   |
| selected-final | x86/rgb-pairs | baseline/gated      | wildcard_scan      |   100 | 1.010× | 1.011/1.008   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_total        |     0 | 0.994× | 0.987/0.999   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_active       |     0 | 0.994× | 0.987/1.001   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_high      |     0 | 1.006× | 1.004/1.008   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_low       |     0 | 1.003× | 1.001/1.006   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_high_med       |     0 | 1.007× | 1.012/0.999   |
| selected-final | x86/rgb-pairs | baseline/gated      | and_multi          |     0 | 0.982× | 0.942/0.998   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_phrase        |     0 | 1.010× | 1.023/0.984   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_sloppy_phrase |     0 | 0.983× | 0.996/0.976   |
| selected-final | x86/rgb-pairs | baseline/gated      | high_term          |     0 | 1.197× | 1.018/1.386   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_phrase         |     0 | 0.991× | 0.992/0.992   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_sloppy_phrase  |     0 | 0.995× | 1.010/0.983   |
| selected-final | x86/rgb-pairs | baseline/gated      | low_term           |     0 | 1.184× | 1.004/1.408   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_phrase         |     0 | 0.990× | 0.997/0.982   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_sloppy_phrase  |     0 | 0.988× | 1.002/0.976   |
| selected-final | x86/rgb-pairs | baseline/gated      | med_term           |     0 | 1.342× | 1.010/1.530   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_balanced     |     0 | 0.995× | 0.990/1.000   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_boundary     |     0 | 0.983× | 0.993/0.964   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_ordered      |     0 | 0.991× | 0.986/0.992   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_skewed       |     0 | 0.997× | 0.978/1.019   |
| selected-final | x86/rgb-pairs | baseline/gated      | multi_wide         |     0 | 0.992× | 0.992/0.993   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_high       |     0 | 0.985× | 1.003/0.974   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_low        |     0 | 1.002× | 1.011/0.997   |
| selected-final | x86/rgb-pairs | baseline/gated      | or_high_med        |     0 | 0.998× | 1.004/0.989   |
| selected-final | x86/rgb-pairs | baseline/gated      | prefix3            |     0 | 1.008× | 1.007/1.012   |
| selected-final | x86/rgb-pairs | baseline/gated      | regex              |     0 | 0.996× | 0.986/1.004   |
| selected-final | x86/rgb-pairs | baseline/gated      | wildcard           |     0 | 1.000× | 0.992/1.007   |
| selected-final | x86/rgb-pairs | baseline/gated      | wildcard_scan      |     0 | 1.005× | 0.986/1.024   |

## Process resources

CPU is total process user + system time, including startup and warmups.
Peak RSS is the mean of the two process maxima; it includes mapped index pages.
Caller-thread timestamps in raw rows are not total query CPU and are not used here.

| Campaign            | Host/index    | Comparison          | Limit | CPU ratio | Before RSS MiB | After RSS MiB |
| ------------------- | ------------- | ------------------- | ----: | --------: | -------------: | ------------: |
| adaptive-integrated | arm/bitmap    | baseline/adaptive   |    10 |    0.993× |         112.04 |        112.45 |
| adaptive-integrated | arm/bitmap    | baseline/adaptive   |   100 |    1.005× |         112.08 |        112.22 |
| adaptive-integrated | arm/bitmap    | baseline/adaptive   |     0 |    1.020× |         109.83 |        109.98 |
| adaptive-integrated | arm/rgb       | baseline/adaptive   |    10 |    1.009× |         121.48 |        121.55 |
| adaptive-integrated | arm/rgb       | baseline/adaptive   |   100 |    1.000× |         121.50 |        121.70 |
| adaptive-integrated | arm/rgb       | baseline/adaptive   |     0 |    1.000× |         119.23 |        119.47 |
| adaptive-integrated | x86/default   | baseline/adaptive   |    10 |    1.018× |        1268.76 |       1268.74 |
| adaptive-integrated | x86/default   | baseline/adaptive   |   100 |    1.017× |        1269.12 |       1268.94 |
| adaptive-integrated | x86/default   | baseline/adaptive   |     0 |    1.030× |        1252.38 |       1252.21 |
| adaptive-integrated | x86/rgb-pairs | baseline/adaptive   |    10 |    1.010× |        1363.81 |       1363.75 |
| adaptive-integrated | x86/rgb-pairs | baseline/adaptive   |   100 |    1.008× |        1363.96 |       1363.99 |
| adaptive-integrated | x86/rgb-pairs | baseline/adaptive   |     0 |    1.012× |        1347.28 |       1347.22 |
| batched-integrated  | arm/bitmap    | baseline/batched    |    10 |    1.103× |         112.11 |        112.12 |
| batched-integrated  | arm/bitmap    | baseline/batched    |   100 |    1.092× |         112.31 |        112.58 |
| batched-integrated  | arm/bitmap    | baseline/batched    |     0 |    1.020× |         109.96 |        110.18 |
| batched-integrated  | arm/rgb       | baseline/batched    |    10 |    1.099× |         121.52 |        121.40 |
| batched-integrated  | arm/rgb       | baseline/batched    |   100 |    1.085× |         121.95 |        121.98 |
| batched-integrated  | arm/rgb       | baseline/batched    |     0 |    1.000× |         119.23 |        119.42 |
| batched-integrated  | x86/default   | baseline/batched    |    10 |    1.164× |        1268.69 |       1268.43 |
| batched-integrated  | x86/default   | baseline/batched    |   100 |    1.164× |        1268.91 |       1268.91 |
| batched-integrated  | x86/default   | baseline/batched    |     0 |    1.012× |        1252.34 |       1252.22 |
| batched-integrated  | x86/rgb-pairs | baseline/batched    |    10 |    1.188× |        1363.65 |       1363.67 |
| batched-integrated  | x86/rgb-pairs | baseline/batched    |   100 |    1.187× |        1364.07 |       1363.89 |
| batched-integrated  | x86/rgb-pairs | baseline/batched    |     0 |    1.005× |        1347.24 |       1347.14 |
| broad               | arm/bitmap    | baseline/batched    |    10 |    1.061× |         222.73 |        222.61 |
| broad               | arm/bitmap    | baseline/batched    |   100 |    1.033× |         226.64 |        224.55 |
| broad               | arm/bitmap    | baseline/batched    |     0 |    1.007× |         221.16 |        220.49 |
| broad               | arm/rgb       | baseline/batched    |    10 |    1.068× |         227.31 |        226.38 |
| broad               | arm/rgb       | baseline/batched    |   100 |    1.039× |         233.15 |        231.70 |
| broad               | arm/rgb       | baseline/batched    |     0 |    1.011× |         229.07 |        228.08 |
| batched-final       | arm/bitmap    | baseline/selected   |    10 |    1.042× |         222.38 |        224.25 |
| batched-final       | arm/bitmap    | baseline/selected   |   100 |    1.022× |         225.60 |        226.26 |
| batched-final       | arm/bitmap    | baseline/selected   |     0 |    1.003× |         220.88 |        220.91 |
| batched-final       | arm/rgb       | baseline/selected   |    10 |    1.054× |         224.65 |        225.24 |
| batched-final       | arm/rgb       | baseline/selected   |   100 |    1.038× |         229.80 |        231.71 |
| batched-final       | arm/rgb       | baseline/selected   |     0 |    1.005× |         227.61 |        227.73 |
| final               | arm/bitmap    | baseline/gated      |    10 |    1.065× |         222.36 |        220.91 |
| final               | arm/bitmap    | baseline/gated      |   100 |    1.043× |         226.69 |        225.65 |
| final               | arm/bitmap    | baseline/gated      |     0 |    1.011× |         220.55 |        220.90 |
| final               | arm/rgb       | baseline/gated      |    10 |    1.059× |         223.70 |        227.59 |
| final               | arm/rgb       | baseline/gated      |   100 |    1.062× |         233.48 |        232.30 |
| final               | arm/rgb       | baseline/gated      |     0 |    1.003× |         228.15 |        228.95 |
| final               | x86/default   | baseline/gated      |    10 |    1.111× |        2306.22 |       2306.04 |
| final               | x86/default   | baseline/wide-gated |    10 |    1.085× |        2306.22 |       2306.10 |
| final               | x86/default   | gated/wide-gated    |    10 |    0.977× |        2306.04 |       2306.10 |
| final               | x86/default   | baseline/gated      |   100 |    1.083× |        2335.19 |       2335.13 |
| final               | x86/default   | baseline/wide-gated |   100 |    1.061× |        2335.19 |       2335.16 |
| final               | x86/default   | gated/wide-gated    |   100 |    0.979× |        2335.13 |       2335.16 |
| final               | x86/default   | baseline/gated      |     0 |    1.010× |        2312.27 |       2312.28 |
| final               | x86/default   | baseline/wide-gated |     0 |    0.980× |        2312.27 |       2312.21 |
| final               | x86/default   | gated/wide-gated    |     0 |    0.971× |        2312.28 |       2312.21 |
| final               | x86/rgb-pairs | baseline/gated      |    10 |    1.154× |        2241.34 |       2241.40 |
| final               | x86/rgb-pairs | baseline/wide-gated |    10 |    1.117× |        2241.34 |       2241.48 |
| final               | x86/rgb-pairs | gated/wide-gated    |    10 |    0.968× |        2241.40 |       2241.48 |
| final               | x86/rgb-pairs | baseline/gated      |   100 |    1.140× |        2258.94 |       2258.85 |
| final               | x86/rgb-pairs | baseline/wide-gated |   100 |    1.106× |        2258.94 |       2258.96 |
| final               | x86/rgb-pairs | gated/wide-gated    |   100 |    0.970× |        2258.85 |       2258.96 |
| final               | x86/rgb-pairs | baseline/gated      |     0 |    1.000× |        2192.41 |       2192.24 |
| final               | x86/rgb-pairs | baseline/wide-gated |     0 |    0.988× |        2192.41 |       2192.25 |
| final               | x86/rgb-pairs | gated/wide-gated    |     0 |    0.987× |        2192.24 |       2192.25 |
| confirm             | arm/bitmap    | baseline/gated      |    10 |    1.131× |         224.51 |        223.91 |
| confirm             | arm/bitmap    | baseline/gated      |   100 |    1.180× |         227.62 |        227.86 |
| confirm             | arm/bitmap    | baseline/gated      |     0 |    0.971× |         223.91 |        224.01 |
| confirm             | arm/rgb       | baseline/gated      |    10 |    0.935× |         226.87 |        227.94 |
| confirm             | arm/rgb       | baseline/gated      |   100 |    1.050× |         233.27 |        234.43 |
| confirm             | arm/rgb       | baseline/gated      |     0 |    0.838× |         231.86 |        232.45 |
| selected-final      | x86/default   | baseline/gated      |    10 |    1.104× |        2305.43 |       2306.13 |
| selected-final      | x86/default   | baseline/gated      |   100 |    1.078× |        2334.39 |       2335.13 |
| selected-final      | x86/default   | baseline/gated      |     0 |    1.004× |        2311.71 |       2312.38 |
| selected-final      | x86/rgb-pairs | baseline/gated      |    10 |    1.149× |        2240.59 |       2241.28 |
| selected-final      | x86/rgb-pairs | baseline/gated      |   100 |    1.141× |        2258.14 |       2258.95 |
| selected-final      | x86/rgb-pairs | baseline/gated      |     0 |    0.992× |        2191.66 |       2192.34 |

## Rejected BSR corpus audit

The tail-corrected BSR prototype passes 185 query audits, then fails exact
ranked output for `+the +of +in +and`. The starting and gated binaries pass
all 199 audits. No BSR query timing is reported. Its reproducer and full
expected/actual vectors are retained in the evidence.

Validated 5,654 exhaustive audits and 260,508 recorded query executions (warmups excluded).
