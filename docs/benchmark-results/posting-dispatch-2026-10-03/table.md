# Complete conditional-dispatch comparison

Speedup = baseline / candidate elapsed time; above 1 is faster. B is the
density gate; C adds inlining and x86 galloping; D is the cleaned generic-build x86 hybrid.
Family times sum each query's median, then average the two phases.
Paired ratios retain forward/reverse-order drift; small ratios are not
automatically statistically significant. Count is an unchanged-path control
for ordinary AND, but phrase counts do exercise the shared kernel.

## Synthetic bounded blocks

| Host | Shape/trace      | Capacity | Kernel      | ns/call | Speedup |
| ---- | ---------------- | -------- | ----------- | ------- | ------- |
| arm  | identical        | 1        | production  | 4.78    | 1.000×  |
| arm  | identical        | 1        | conditional | 4.64    | 1.030×  |
| arm  | identical        | 1        | scalar      | 3.75    | 1.275×  |
| arm  | identical        | 1        | gallop      | 3.90    | 1.225×  |
| arm  | identical        | 1        | cross4      | 4.61    | 1.037×  |
| arm  | identical        | 1        | prefix8     | 4.60    | 1.039×  |
| arm  | identical        | 1        | simd_gallop | 4.50    | 1.062×  |
| arm  | identical        | 1        | byte_filter | 4.53    | 1.055×  |
| arm  | identical        | 128      | production  | 263.53  | 1.000×  |
| arm  | identical        | 128      | conditional | 133.91  | 1.968×  |
| arm  | identical        | 128      | scalar      | 106.92  | 2.465×  |
| arm  | identical        | 128      | gallop      | 187.08  | 1.409×  |
| arm  | identical        | 128      | cross4      | 136.54  | 1.930×  |
| arm  | identical        | 128      | prefix8     | 910.27  | 0.290×  |
| arm  | identical        | 128      | simd_gallop | 255.74  | 1.030×  |
| arm  | identical        | 128      | byte_filter | 140.32  | 1.878×  |
| arm  | balanced_dense   | 1        | production  | 10.45   | 1.000×  |
| arm  | balanced_dense   | 1        | conditional | 8.50    | 1.230×  |
| arm  | balanced_dense   | 1        | scalar      | 7.56    | 1.383×  |
| arm  | balanced_dense   | 1        | gallop      | 13.45   | 0.777×  |
| arm  | balanced_dense   | 1        | cross4      | 8.36    | 1.250×  |
| arm  | balanced_dense   | 1        | prefix8     | 13.09   | 0.799×  |
| arm  | balanced_dense   | 1        | simd_gallop | 8.55    | 1.223×  |
| arm  | balanced_dense   | 1        | byte_filter | 8.24    | 1.268×  |
| arm  | balanced_dense   | 128      | production  | 403.03  | 1.000×  |
| arm  | balanced_dense   | 128      | conditional | 280.06  | 1.439×  |
| arm  | balanced_dense   | 128      | scalar      | 347.32  | 1.160×  |
| arm  | balanced_dense   | 128      | gallop      | 978.02  | 0.412×  |
| arm  | balanced_dense   | 128      | cross4      | 282.56  | 1.426×  |
| arm  | balanced_dense   | 128      | prefix8     | 1247.30 | 0.323×  |
| arm  | balanced_dense   | 128      | simd_gallop | 342.21  | 1.178×  |
| arm  | balanced_dense   | 128      | byte_filter | 275.80  | 1.461×  |
| arm  | balanced_sparse  | 1        | production  | 94.52   | 1.000×  |
| arm  | balanced_sparse  | 1        | conditional | 94.92   | 0.996×  |
| arm  | balanced_sparse  | 1        | scalar      | 116.42  | 0.812×  |
| arm  | balanced_sparse  | 1        | gallop      | 401.43  | 0.235×  |
| arm  | balanced_sparse  | 1        | cross4      | 94.74   | 0.998×  |
| arm  | balanced_sparse  | 1        | prefix8     | 326.25  | 0.290×  |
| arm  | balanced_sparse  | 1        | simd_gallop | 84.52   | 1.118×  |
| arm  | balanced_sparse  | 1        | byte_filter | 74.64   | 1.266×  |
| arm  | balanced_sparse  | 128      | production  | 254.48  | 1.000×  |
| arm  | balanced_sparse  | 128      | conditional | 236.17  | 1.078×  |
| arm  | balanced_sparse  | 128      | scalar      | 281.46  | 0.904×  |
| arm  | balanced_sparse  | 128      | gallop      | 1012.92 | 0.251×  |
| arm  | balanced_sparse  | 128      | cross4      | 233.88  | 1.088×  |
| arm  | balanced_sparse  | 128      | prefix8     | 892.35  | 0.285×  |
| arm  | balanced_sparse  | 128      | simd_gallop | 207.28  | 1.228×  |
| arm  | balanced_sparse  | 128      | byte_filter | 194.13  | 1.311×  |
| arm  | left_8x_sparse   | 1        | production  | 10.91   | 1.000×  |
| arm  | left_8x_sparse   | 1        | conditional | 11.74   | 0.929×  |
| arm  | left_8x_sparse   | 1        | scalar      | 26.16   | 0.417×  |
| arm  | left_8x_sparse   | 1        | gallop      | 33.44   | 0.326×  |
| arm  | left_8x_sparse   | 1        | cross4      | 19.17   | 0.569×  |
| arm  | left_8x_sparse   | 1        | prefix8     | 17.48   | 0.624×  |
| arm  | left_8x_sparse   | 1        | simd_gallop | 15.07   | 0.724×  |
| arm  | left_8x_sparse   | 1        | byte_filter | 15.95   | 0.684×  |
| arm  | left_8x_sparse   | 128      | production  | 57.70   | 1.000×  |
| arm  | left_8x_sparse   | 128      | conditional | 59.52   | 0.969×  |
| arm  | left_8x_sparse   | 128      | scalar      | 149.79  | 0.385×  |
| arm  | left_8x_sparse   | 128      | gallop      | 253.93  | 0.227×  |
| arm  | left_8x_sparse   | 128      | cross4      | 119.47  | 0.483×  |
| arm  | left_8x_sparse   | 128      | prefix8     | 146.47  | 0.394×  |
| arm  | left_8x_sparse   | 128      | simd_gallop | 113.71  | 0.507×  |
| arm  | left_8x_sparse   | 128      | byte_filter | 110.32  | 0.523×  |
| arm  | right_8x_sparse  | 1        | production  | 36.71   | 1.000×  |
| arm  | right_8x_sparse  | 1        | conditional | 37.63   | 0.976×  |
| arm  | right_8x_sparse  | 1        | scalar      | 28.24   | 1.300×  |
| arm  | right_8x_sparse  | 1        | gallop      | 37.07   | 0.990×  |
| arm  | right_8x_sparse  | 1        | cross4      | 20.24   | 1.814×  |
| arm  | right_8x_sparse  | 1        | prefix8     | 115.80  | 0.317×  |
| arm  | right_8x_sparse  | 1        | simd_gallop | 31.53   | 1.164×  |
| arm  | right_8x_sparse  | 1        | byte_filter | 16.95   | 2.166×  |
| arm  | right_8x_sparse  | 128      | production  | 289.46  | 1.000×  |
| arm  | right_8x_sparse  | 128      | conditional | 290.23  | 0.997×  |
| arm  | right_8x_sparse  | 128      | scalar      | 169.66  | 1.706×  |
| arm  | right_8x_sparse  | 128      | gallop      | 257.38  | 1.125×  |
| arm  | right_8x_sparse  | 128      | cross4      | 128.62  | 2.250×  |
| arm  | right_8x_sparse  | 128      | prefix8     | 961.51  | 0.301×  |
| arm  | right_8x_sparse  | 128      | simd_gallop | 264.55  | 1.094×  |
| arm  | right_8x_sparse  | 128      | byte_filter | 118.80  | 2.437×  |
| arm  | left_64x_sparse  | 1        | production  | 14.86   | 1.000×  |
| arm  | left_64x_sparse  | 1        | conditional | 15.59   | 0.953×  |
| arm  | left_64x_sparse  | 1        | scalar      | 103.08  | 0.144×  |
| arm  | left_64x_sparse  | 1        | gallop      | 22.17   | 0.671×  |
| arm  | left_64x_sparse  | 1        | cross4      | 79.18   | 0.188×  |
| arm  | left_64x_sparse  | 1        | prefix8     | 36.70   | 0.405×  |
| arm  | left_64x_sparse  | 1        | simd_gallop | 12.10   | 1.228×  |
| arm  | left_64x_sparse  | 1        | byte_filter | 60.83   | 0.244×  |
| arm  | left_64x_sparse  | 128      | production  | 21.65   | 1.000×  |
| arm  | left_64x_sparse  | 128      | conditional | 22.50   | 0.962×  |
| arm  | left_64x_sparse  | 128      | scalar      | 156.23  | 0.139×  |
| arm  | left_64x_sparse  | 128      | gallop      | 38.03   | 0.569×  |
| arm  | left_64x_sparse  | 128      | cross4      | 117.80  | 0.184×  |
| arm  | left_64x_sparse  | 128      | prefix8     | 61.86   | 0.350×  |
| arm  | left_64x_sparse  | 128      | simd_gallop | 18.77   | 1.153×  |
| arm  | left_64x_sparse  | 128      | byte_filter | 95.51   | 0.227×  |
| arm  | right_64x_sparse | 1        | production  | 82.86   | 1.000×  |
| arm  | right_64x_sparse | 1        | conditional | 85.45   | 0.970×  |
| arm  | right_64x_sparse | 1        | scalar      | 115.06  | 0.720×  |
| arm  | right_64x_sparse | 1        | gallop      | 21.22   | 3.904×  |
| arm  | right_64x_sparse | 1        | cross4      | 88.82   | 0.933×  |
| arm  | right_64x_sparse | 1        | prefix8     | 657.05  | 0.126×  |
| arm  | right_64x_sparse | 1        | simd_gallop | 129.61  | 0.639×  |
| arm  | right_64x_sparse | 1        | byte_filter | 68.54   | 1.209×  |
| arm  | right_64x_sparse | 128      | production  | 150.23  | 1.000×  |
| arm  | right_64x_sparse | 128      | conditional | 156.30  | 0.961×  |
| arm  | right_64x_sparse | 128      | scalar      | 167.75  | 0.896×  |
| arm  | right_64x_sparse | 128      | gallop      | 34.20   | 4.393×  |
| arm  | right_64x_sparse | 128      | cross4      | 132.06  | 1.138×  |
| arm  | right_64x_sparse | 128      | prefix8     | 880.08  | 0.171×  |
| arm  | right_64x_sparse | 128      | simd_gallop | 178.77  | 0.840×  |
| arm  | right_64x_sparse | 128      | byte_filter | 95.78   | 1.568×  |
| x86  | identical        | 1        | production  | 15.58   | 1.000×  |
| x86  | identical        | 1        | conditional | 12.31   | 1.266×  |
| x86  | identical        | 1        | scalar      | 11.44   | 1.362×  |
| x86  | identical        | 1        | gallop      | 11.72   | 1.330×  |
| x86  | identical        | 1        | cross4      | 11.63   | 1.340×  |
| x86  | identical        | 1        | simd_gallop | 12.11   | 1.286×  |
| x86  | identical        | 1        | byte_filter | 11.70   | 1.331×  |
| x86  | identical        | 1        | cross16     | 12.98   | 1.200×  |
| x86  | identical        | 128      | production  | 857.19  | 1.000×  |
| x86  | identical        | 128      | conditional | 200.00  | 4.286×  |
| x86  | identical        | 128      | scalar      | 146.32  | 5.858×  |
| x86  | identical        | 128      | gallop      | 203.31  | 4.216×  |
| x86  | identical        | 128      | cross4      | 198.70  | 4.314×  |
| x86  | identical        | 128      | simd_gallop | 303.82  | 2.821×  |
| x86  | identical        | 128      | byte_filter | 220.59  | 3.886×  |
| x86  | identical        | 128      | cross16     | 190.79  | 4.493×  |
| x86  | balanced_dense   | 1        | production  | 20.35   | 1.000×  |
| x86  | balanced_dense   | 1        | conditional | 12.84   | 1.585×  |
| x86  | balanced_dense   | 1        | scalar      | 12.21   | 1.667×  |
| x86  | balanced_dense   | 1        | gallop      | 16.78   | 1.213×  |
| x86  | balanced_dense   | 1        | cross4      | 12.06   | 1.687×  |
| x86  | balanced_dense   | 1        | simd_gallop | 12.54   | 1.623×  |
| x86  | balanced_dense   | 1        | byte_filter | 12.06   | 1.687×  |
| x86  | balanced_dense   | 1        | cross16     | 12.96   | 1.571×  |
| x86  | balanced_dense   | 128      | production  | 861.37  | 1.000×  |
| x86  | balanced_dense   | 128      | conditional | 303.05  | 2.842×  |
| x86  | balanced_dense   | 128      | scalar      | 901.14  | 0.956×  |
| x86  | balanced_dense   | 128      | gallop      | 975.11  | 0.883×  |
| x86  | balanced_dense   | 128      | cross4      | 290.41  | 2.966×  |
| x86  | balanced_dense   | 128      | simd_gallop | 782.19  | 1.101×  |
| x86  | balanced_dense   | 128      | byte_filter | 292.77  | 2.942×  |
| x86  | balanced_dense   | 128      | cross16     | 287.33  | 2.998×  |
| x86  | balanced_sparse  | 1        | production  | 297.82  | 1.000×  |
| x86  | balanced_sparse  | 1        | conditional | 99.01   | 3.008×  |
| x86  | balanced_sparse  | 1        | scalar      | 229.83  | 1.296×  |
| x86  | balanced_sparse  | 1        | gallop      | 310.12  | 0.960×  |
| x86  | balanced_sparse  | 1        | cross4      | 98.92   | 3.011×  |
| x86  | balanced_sparse  | 1        | simd_gallop | 80.46   | 3.702×  |
| x86  | balanced_sparse  | 1        | byte_filter | 70.08   | 4.250×  |
| x86  | balanced_sparse  | 1        | cross16     | 70.76   | 4.209×  |
| x86  | balanced_sparse  | 128      | production  | 854.99  | 1.000×  |
| x86  | balanced_sparse  | 128      | conditional | 298.25  | 2.867×  |
| x86  | balanced_sparse  | 128      | scalar      | 1000.02 | 0.855×  |
| x86  | balanced_sparse  | 128      | gallop      | 1090.04 | 0.784×  |
| x86  | balanced_sparse  | 128      | cross4      | 292.93  | 2.919×  |
| x86  | balanced_sparse  | 128      | simd_gallop | 307.71  | 2.779×  |
| x86  | balanced_sparse  | 128      | byte_filter | 208.73  | 4.096×  |
| x86  | balanced_sparse  | 128      | cross16     | 220.46  | 3.878×  |
| x86  | left_8x_sparse   | 1        | production  | 26.52   | 1.000×  |
| x86  | left_8x_sparse   | 1        | conditional | 28.50   | 0.931×  |
| x86  | left_8x_sparse   | 1        | scalar      | 31.04   | 0.854×  |
| x86  | left_8x_sparse   | 1        | gallop      | 40.49   | 0.655×  |
| x86  | left_8x_sparse   | 1        | cross4      | 29.03   | 0.914×  |
| x86  | left_8x_sparse   | 1        | simd_gallop | 18.44   | 1.439×  |
| x86  | left_8x_sparse   | 1        | byte_filter | 23.12   | 1.147×  |
| x86  | left_8x_sparse   | 1        | cross16     | 21.52   | 1.232×  |
| x86  | left_8x_sparse   | 128      | production  | 159.77  | 1.000×  |
| x86  | left_8x_sparse   | 128      | conditional | 164.08  | 0.974×  |
| x86  | left_8x_sparse   | 128      | scalar      | 201.04  | 0.795×  |
| x86  | left_8x_sparse   | 128      | gallop      | 257.18  | 0.621×  |
| x86  | left_8x_sparse   | 128      | cross4      | 166.15  | 0.962×  |
| x86  | left_8x_sparse   | 128      | simd_gallop | 82.76   | 1.930×  |
| x86  | left_8x_sparse   | 128      | byte_filter | 127.24  | 1.256×  |
| x86  | left_8x_sparse   | 128      | cross16     | 126.41  | 1.264×  |
| x86  | right_8x_sparse  | 1        | production  | 126.37  | 1.000×  |
| x86  | right_8x_sparse  | 1        | conditional | 129.14  | 0.979×  |
| x86  | right_8x_sparse  | 1        | scalar      | 42.40   | 2.980×  |
| x86  | right_8x_sparse  | 1        | gallop      | 37.19   | 3.398×  |
| x86  | right_8x_sparse  | 1        | cross4      | 25.77   | 4.904×  |
| x86  | right_8x_sparse  | 1        | simd_gallop | 30.69   | 4.117×  |
| x86  | right_8x_sparse  | 1        | byte_filter | 20.83   | 6.068×  |
| x86  | right_8x_sparse  | 1        | cross16     | 21.09   | 5.992×  |
| x86  | right_8x_sparse  | 128      | production  | 915.09  | 1.000×  |
| x86  | right_8x_sparse  | 128      | conditional | 919.30  | 0.995×  |
| x86  | right_8x_sparse  | 128      | scalar      | 188.72  | 4.849×  |
| x86  | right_8x_sparse  | 128      | gallop      | 256.50  | 3.568×  |
| x86  | right_8x_sparse  | 128      | cross4      | 167.14  | 5.475×  |
| x86  | right_8x_sparse  | 128      | simd_gallop | 225.25  | 4.063×  |
| x86  | right_8x_sparse  | 128      | byte_filter | 129.51  | 7.066×  |
| x86  | right_8x_sparse  | 128      | cross16     | 126.00  | 7.262×  |
| x86  | left_64x_sparse  | 1        | production  | 36.96   | 1.000×  |
| x86  | left_64x_sparse  | 1        | conditional | 39.23   | 0.942×  |
| x86  | left_64x_sparse  | 1        | scalar      | 117.16  | 0.316×  |
| x86  | left_64x_sparse  | 1        | gallop      | 31.08   | 1.189×  |
| x86  | left_64x_sparse  | 1        | cross4      | 99.12   | 0.373×  |
| x86  | left_64x_sparse  | 1        | simd_gallop | 16.17   | 2.286×  |
| x86  | left_64x_sparse  | 1        | byte_filter | 71.15   | 0.520×  |
| x86  | left_64x_sparse  | 1        | cross16     | 77.15   | 0.479×  |
| x86  | left_64x_sparse  | 128      | production  | 57.69   | 1.000×  |
| x86  | left_64x_sparse  | 128      | conditional | 60.64   | 0.951×  |
| x86  | left_64x_sparse  | 128      | scalar      | 176.95  | 0.326×  |
| x86  | left_64x_sparse  | 128      | gallop      | 48.66   | 1.186×  |
| x86  | left_64x_sparse  | 128      | cross4      | 150.23  | 0.384×  |
| x86  | left_64x_sparse  | 128      | simd_gallop | 21.25   | 2.715×  |
| x86  | left_64x_sparse  | 128      | byte_filter | 106.72  | 0.541×  |
| x86  | left_64x_sparse  | 128      | cross16     | 118.90  | 0.485×  |
| x86  | right_64x_sparse | 1        | production  | 585.23  | 1.000×  |
| x86  | right_64x_sparse | 1        | conditional | 587.98  | 0.995×  |
| x86  | right_64x_sparse | 1        | scalar      | 105.31  | 5.557×  |
| x86  | right_64x_sparse | 1        | gallop      | 29.55   | 19.803× |
| x86  | right_64x_sparse | 1        | cross4      | 99.85   | 5.861×  |
| x86  | right_64x_sparse | 1        | simd_gallop | 131.73  | 4.443×  |
| x86  | right_64x_sparse | 1        | byte_filter | 71.00   | 8.243×  |
| x86  | right_64x_sparse | 1        | cross16     | 76.31   | 7.669×  |
| x86  | right_64x_sparse | 128      | production  | 895.84  | 1.000×  |
| x86  | right_64x_sparse | 128      | conditional | 901.19  | 0.994×  |
| x86  | right_64x_sparse | 128      | scalar      | 161.19  | 5.558×  |
| x86  | right_64x_sparse | 128      | gallop      | 46.18   | 19.400× |
| x86  | right_64x_sparse | 128      | cross4      | 150.36  | 5.958×  |
| x86  | right_64x_sparse | 128      | simd_gallop | 200.10  | 4.477×  |
| x86  | right_64x_sparse | 128      | byte_filter | 107.69  | 8.319×  |
| x86  | right_64x_sparse | 128      | cross16     | 118.46  | 7.563×  |

## Sampled real-block replay

| Host | Shape/trace        | Capacity | Kernel      | ns/call | Speedup |
| ---- | ------------------ | -------- | ----------- | ------- | ------- |
| arm  | trace-bitmap       | recorded | production  | 104.42  | 1.000×  |
| arm  | trace-bitmap       | recorded | conditional | 105.51  | 0.990×  |
| arm  | trace-bitmap       | recorded | scalar      | 189.29  | 0.552×  |
| arm  | trace-bitmap       | recorded | gallop      | 270.87  | 0.386×  |
| arm  | trace-bitmap       | recorded | cross4      | 126.77  | 0.824×  |
| arm  | trace-bitmap       | recorded | simd_gallop | 123.29  | 0.847×  |
| arm  | trace-bitmap       | recorded | byte_filter | 106.39  | 0.982×  |
| arm  | trace-bitmap       | recorded | hybrid      | 93.08   | 1.122×  |
| arm  | trace-bitmap       | recorded | filtered    | 91.23   | 1.145×  |
| arm  | trace-rgb          | recorded | production  | 99.66   | 1.000×  |
| arm  | trace-rgb          | recorded | conditional | 90.90   | 1.096×  |
| arm  | trace-rgb          | recorded | scalar      | 176.55  | 0.564×  |
| arm  | trace-rgb          | recorded | gallop      | 255.22  | 0.390×  |
| arm  | trace-rgb          | recorded | cross4      | 125.42  | 0.795×  |
| arm  | trace-rgb          | recorded | simd_gallop | 108.23  | 0.921×  |
| arm  | trace-rgb          | recorded | byte_filter | 97.07   | 1.027×  |
| arm  | trace-rgb          | recorded | hybrid      | 90.42   | 1.102×  |
| arm  | trace-rgb          | recorded | filtered    | 90.07   | 1.107×  |
| arm  | trace-multi-bitmap | recorded | production  | 92.90   | 1.000×  |
| arm  | trace-multi-bitmap | recorded | conditional | 62.16   | 1.494×  |
| arm  | trace-multi-bitmap | recorded | scalar      | 136.71  | 0.680×  |
| arm  | trace-multi-bitmap | recorded | gallop      | 181.67  | 0.511×  |
| arm  | trace-multi-bitmap | recorded | cross4      | 75.54   | 1.230×  |
| arm  | trace-multi-bitmap | recorded | simd_gallop | 87.09   | 1.067×  |
| arm  | trace-multi-bitmap | recorded | byte_filter | 72.39   | 1.283×  |
| arm  | trace-multi-bitmap | recorded | hybrid      | 66.32   | 1.401×  |
| arm  | trace-multi-bitmap | recorded | filtered    | 66.23   | 1.403×  |
| arm  | trace-multi-rgb    | recorded | production  | 90.74   | 1.000×  |
| arm  | trace-multi-rgb    | recorded | conditional | 65.98   | 1.375×  |
| arm  | trace-multi-rgb    | recorded | scalar      | 128.09  | 0.708×  |
| arm  | trace-multi-rgb    | recorded | gallop      | 178.08  | 0.510×  |
| arm  | trace-multi-rgb    | recorded | cross4      | 72.14   | 1.258×  |
| arm  | trace-multi-rgb    | recorded | simd_gallop | 86.33   | 1.051×  |
| arm  | trace-multi-rgb    | recorded | byte_filter | 72.66   | 1.249×  |
| arm  | trace-multi-rgb    | recorded | hybrid      | 63.28   | 1.434×  |
| arm  | trace-multi-rgb    | recorded | filtered    | 66.22   | 1.370×  |
| x86  | trace-bitmap       | recorded | production  | 336.30  | 1.000×  |
| x86  | trace-bitmap       | recorded | conditional | 338.32  | 0.994×  |
| x86  | trace-bitmap       | recorded | scalar      | 386.41  | 0.870×  |
| x86  | trace-bitmap       | recorded | gallop      | 462.27  | 0.727×  |
| x86  | trace-bitmap       | recorded | cross4      | 335.57  | 1.002×  |
| x86  | trace-bitmap       | recorded | simd_gallop | 260.70  | 1.290×  |
| x86  | trace-bitmap       | recorded | byte_filter | 336.02  | 1.001×  |
| x86  | trace-bitmap       | recorded | hybrid      | 314.38  | 1.070×  |
| x86  | trace-bitmap       | recorded | filtered    | 331.89  | 1.013×  |
| x86  | trace-bitmap       | recorded | cross16     | 317.18  | 1.060×  |
| x86  | trace-rgb          | recorded | production  | 339.49  | 1.000×  |
| x86  | trace-rgb          | recorded | conditional | 327.38  | 1.037×  |
| x86  | trace-rgb          | recorded | scalar      | 371.50  | 0.914×  |
| x86  | trace-rgb          | recorded | gallop      | 453.83  | 0.748×  |
| x86  | trace-rgb          | recorded | cross4      | 324.58  | 1.046×  |
| x86  | trace-rgb          | recorded | simd_gallop | 257.89  | 1.316×  |
| x86  | trace-rgb          | recorded | byte_filter | 329.78  | 1.029×  |
| x86  | trace-rgb          | recorded | hybrid      | 305.90  | 1.110×  |
| x86  | trace-rgb          | recorded | filtered    | 323.85  | 1.048×  |
| x86  | trace-rgb          | recorded | cross16     | 314.99  | 1.078×  |
| x86  | trace-multi-bitmap | recorded | production  | 308.08  | 1.000×  |
| x86  | trace-multi-bitmap | recorded | conditional | 241.53  | 1.276×  |
| x86  | trace-multi-bitmap | recorded | scalar      | 297.63  | 1.035×  |
| x86  | trace-multi-bitmap | recorded | gallop      | 362.18  | 0.851×  |
| x86  | trace-multi-bitmap | recorded | cross4      | 231.63  | 1.330×  |
| x86  | trace-multi-bitmap | recorded | simd_gallop | 223.75  | 1.377×  |
| x86  | trace-multi-bitmap | recorded | byte_filter | 239.48  | 1.286×  |
| x86  | trace-multi-bitmap | recorded | hybrid      | 231.33  | 1.332×  |
| x86  | trace-multi-bitmap | recorded | filtered    | 239.39  | 1.287×  |
| x86  | trace-multi-bitmap | recorded | cross16     | 250.78  | 1.228×  |
| x86  | trace-multi-rgb    | recorded | production  | 324.19  | 1.000×  |
| x86  | trace-multi-rgb    | recorded | conditional | 250.41  | 1.295×  |
| x86  | trace-multi-rgb    | recorded | scalar      | 291.11  | 1.114×  |
| x86  | trace-multi-rgb    | recorded | gallop      | 359.92  | 0.901×  |
| x86  | trace-multi-rgb    | recorded | cross4      | 230.24  | 1.408×  |
| x86  | trace-multi-rgb    | recorded | simd_gallop | 225.53  | 1.437×  |
| x86  | trace-multi-rgb    | recorded | byte_filter | 237.52  | 1.365×  |
| x86  | trace-multi-rgb    | recorded | hybrid      | 235.52  | 1.376×  |
| x86  | trace-multi-rgb    | recorded | filtered    | 252.25  | 1.285×  |
| x86  | trace-multi-rgb    | recorded | cross16     | 245.34  | 1.321×  |

## Selected generic release, 199 queries

| Host/index    | Limit | Family             | Variant | A sum µs   | Candidate sum µs | Speedup | Forward/reverse |
| ------------- | ----- | ------------------ | ------- | ---------- | ---------------- | ------- | --------------- |
| x86/default   | 10    | high_term          | D       | 905.18     | 903.40           | 1.002×  | 0.984/1.020     |
| x86/default   | 10    | all                | D       | 3915369.43 | 3747554.81       | 1.045×  | 1.046/1.044     |
| x86/default   | 10    | med_term           | D       | 317.89     | 316.47           | 1.004×  | 0.993/1.017     |
| x86/default   | 10    | low_term           | D       | 180.90     | 178.25           | 1.015×  | 1.001/1.029     |
| x86/default   | 10    | and_high_high      | D       | 73071.09   | 73522.93         | 0.994×  | 1.002/0.986     |
| x86/default   | 10    | and_high_med       | D       | 39268.03   | 39293.86         | 0.999×  | 1.005/0.993     |
| x86/default   | 10    | and_high_low       | D       | 2852.64    | 2854.51          | 0.999×  | 1.001/0.997     |
| x86/default   | 10    | or_high_high       | D       | 136877.24  | 137849.94        | 0.993×  | 0.994/0.992     |
| x86/default   | 10    | or_high_med        | D       | 45912.51   | 45833.73         | 1.002×  | 1.000/1.004     |
| x86/default   | 10    | or_high_low        | D       | 5580.01    | 5631.66          | 0.991×  | 0.988/0.993     |
| x86/default   | 10    | high_phrase        | D       | 58867.32   | 59599.94         | 0.988×  | 0.992/0.984     |
| x86/default   | 10    | med_phrase         | D       | 185896.65  | 186926.46        | 0.994×  | 0.995/0.994     |
| x86/default   | 10    | low_phrase         | D       | 26432.61   | 26573.50         | 0.995×  | 0.995/0.995     |
| x86/default   | 10    | high_sloppy_phrase | D       | 16449.90   | 16304.13         | 1.009×  | 1.014/1.004     |
| x86/default   | 10    | med_sloppy_phrase  | D       | 132799.17  | 132132.42        | 1.005×  | 1.005/1.005     |
| x86/default   | 10    | low_sloppy_phrase  | D       | 211405.47  | 207835.92        | 1.017×  | 1.019/1.015     |
| x86/default   | 10    | wildcard           | D       | 2040.55    | 2040.49          | 1.000×  | 1.002/0.998     |
| x86/default   | 10    | prefix3            | D       | 164.70     | 163.66           | 1.006×  | 0.997/1.016     |
| x86/default   | 10    | wildcard_scan      | D       | 4488.59    | 4446.03          | 1.010×  | 1.005/1.014     |
| x86/default   | 10    | regex              | D       | 11106.16   | 11126.56         | 0.998×  | 0.992/1.005     |
| x86/default   | 10    | multi_balanced     | D       | 816727.58  | 760180.90        | 1.074×  | 1.078/1.071     |
| x86/default   | 10    | and_multi          | D       | 222349.42  | 219379.58        | 1.014×  | 1.015/1.012     |
| x86/default   | 10    | multi_skewed       | D       | 262249.55  | 258628.74        | 1.014×  | 1.017/1.011     |
| x86/default   | 10    | multi_wide         | D       | 768365.50  | 738304.90        | 1.041×  | 1.042/1.040     |
| x86/default   | 10    | multi_ordered      | D       | 325305.65  | 302890.24        | 1.074×  | 1.070/1.078     |
| x86/default   | 10    | multi_boundary     | D       | 565755.09  | 514636.59        | 1.099×  | 1.096/1.103     |
| x86/default   | 100   | high_term          | D       | 2899.60    | 2884.41          | 1.005×  | 1.018/0.993     |
| x86/default   | 100   | all                | D       | 5122957.70 | 4966708.17       | 1.031×  | 1.031/1.031     |
| x86/default   | 100   | med_term           | D       | 1707.64    | 1717.98          | 0.994×  | 1.015/0.974     |
| x86/default   | 100   | low_term           | D       | 986.44     | 991.20           | 0.995×  | 1.036/0.957     |
| x86/default   | 100   | and_high_high      | D       | 136807.73  | 136899.22        | 0.999×  | 0.999/1.000     |
| x86/default   | 100   | and_high_med       | D       | 54871.58   | 54771.17         | 1.002×  | 0.998/1.006     |
| x86/default   | 100   | and_high_low       | D       | 3258.82    | 3256.20          | 1.001×  | 1.002/0.999     |
| x86/default   | 100   | or_high_high       | D       | 193130.56  | 195264.28        | 0.989×  | 0.993/0.985     |
| x86/default   | 100   | or_high_med        | D       | 60399.30   | 61813.49         | 0.977×  | 0.992/0.962     |
| x86/default   | 100   | or_high_low        | D       | 11393.56   | 11427.05         | 0.997×  | 1.005/0.990     |
| x86/default   | 100   | high_phrase        | D       | 335000.98  | 336212.92        | 0.996×  | 0.999/0.994     |
| x86/default   | 100   | med_phrase         | D       | 636092.83  | 641863.89        | 0.991×  | 0.991/0.991     |
| x86/default   | 100   | low_phrase         | D       | 81055.49   | 83908.59         | 0.966×  | 0.976/0.956     |
| x86/default   | 100   | high_sloppy_phrase | D       | 44426.66   | 44379.82         | 1.001×  | 0.998/1.004     |
| x86/default   | 100   | med_sloppy_phrase  | D       | 251199.28  | 250080.49        | 1.004×  | 1.006/1.003     |
| x86/default   | 100   | low_sloppy_phrase  | D       | 327509.73  | 324987.88        | 1.008×  | 1.011/1.004     |
| x86/default   | 100   | wildcard           | D       | 1795.46    | 1792.58          | 1.002×  | 1.005/0.999     |
| x86/default   | 100   | prefix3            | D       | 175.76     | 174.03           | 1.010×  | 1.003/1.017     |
| x86/default   | 100   | wildcard_scan      | D       | 4546.06    | 4496.46          | 1.011×  | 1.016/1.006     |
| x86/default   | 100   | regex              | D       | 11053.03   | 11056.70         | 1.000×  | 0.997/1.002     |
| x86/default   | 100   | multi_balanced     | D       | 817725.91  | 759456.91        | 1.077×  | 1.076/1.078     |
| x86/default   | 100   | and_multi          | D       | 224631.02  | 220347.73        | 1.019×  | 1.016/1.023     |
| x86/default   | 100   | multi_skewed       | D       | 263183.95  | 259030.68        | 1.016×  | 1.017/1.015     |
| x86/default   | 100   | multi_wide         | D       | 764228.93  | 736541.09        | 1.038×  | 1.037/1.038     |
| x86/default   | 100   | multi_ordered      | D       | 325746.56  | 303130.29        | 1.075×  | 1.073/1.076     |
| x86/default   | 100   | multi_boundary     | D       | 569130.81  | 520223.13        | 1.094×  | 1.089/1.099     |
| x86/default   | count | high_term          | D       | 4.39       | 4.37             | 1.004×  | 0.997/1.011     |
| x86/default   | count | all                | D       | 8522913.97 | 8343245.68       | 1.022×  | 1.023/1.020     |
| x86/default   | count | med_term           | D       | 5.55       | 5.62             | 0.988×  | 1.022/0.955     |
| x86/default   | count | low_term           | D       | 5.26       | 5.38             | 0.979×  | 0.994/0.964     |
| x86/default   | count | and_high_high      | D       | 57507.22   | 57248.83         | 1.005×  | 1.001/1.008     |
| x86/default   | count | and_high_med       | D       | 36690.11   | 36766.61         | 0.998×  | 1.003/0.993     |
| x86/default   | count | and_high_low       | D       | 1291.13    | 1303.82          | 0.990×  | 0.999/0.982     |
| x86/default   | count | or_high_high       | D       | 54592.08   | 54666.58         | 0.999×  | 1.009/0.988     |
| x86/default   | count | or_high_med        | D       | 34856.76   | 34580.94         | 1.008×  | 1.017/0.999     |
| x86/default   | count | or_high_low        | D       | 2033.88    | 2057.75          | 0.988×  | 0.992/0.985     |
| x86/default   | count | high_phrase        | D       | 4371413.38 | 4273647.72       | 1.023×  | 1.021/1.025     |
| x86/default   | count | med_phrase         | D       | 1683290.31 | 1654915.74       | 1.017×  | 1.019/1.015     |
| x86/default   | count | low_phrase         | D       | 314387.62  | 315944.49        | 0.995×  | 0.990/1.000     |
| x86/default   | count | high_sloppy_phrase | D       | 628246.43  | 607783.28        | 1.034×  | 1.045/1.022     |
| x86/default   | count | med_sloppy_phrase  | D       | 720351.32  | 697123.16        | 1.033×  | 1.044/1.022     |
| x86/default   | count | low_sloppy_phrase  | D       | 369873.09  | 359795.92        | 1.028×  | 1.043/1.014     |
| x86/default   | count | wildcard           | D       | 4946.48    | 5025.71          | 0.984×  | 0.979/0.989     |
| x86/default   | count | prefix3            | D       | 3188.00    | 3178.05          | 1.003×  | 1.000/1.007     |
| x86/default   | count | wildcard_scan      | D       | 6189.94    | 6137.48          | 1.009×  | 0.984/1.034     |
| x86/default   | count | regex              | D       | 16077.56   | 15680.97         | 1.025×  | 1.018/1.033     |
| x86/default   | count | multi_balanced     | D       | 55498.37   | 54581.39         | 1.017×  | 1.015/1.018     |
| x86/default   | count | and_multi          | D       | 14053.46   | 13829.78         | 1.016×  | 0.991/1.042     |
| x86/default   | count | multi_skewed       | D       | 41922.36   | 42423.04         | 0.988×  | 0.981/0.995     |
| x86/default   | count | multi_wide         | D       | 65402.34   | 65434.31         | 1.000×  | 0.994/1.005     |
| x86/default   | count | multi_ordered      | D       | 30963.42   | 30993.65         | 0.999×  | 1.003/0.995     |
| x86/default   | count | multi_boundary     | D       | 10123.51   | 10111.12         | 1.001×  | 0.995/1.007     |
| x86/rgb-pairs | 10    | high_term          | D       | 1020.02    | 1016.64          | 1.003×  | 1.010/0.997     |
| x86/rgb-pairs | 10    | all                | D       | 3457988.95 | 3260313.71       | 1.061×  | 1.061/1.061     |
| x86/rgb-pairs | 10    | med_term           | D       | 378.49     | 377.66           | 1.002×  | 1.004/1.001     |
| x86/rgb-pairs | 10    | low_term           | D       | 187.27     | 187.19           | 1.000×  | 1.004/0.997     |
| x86/rgb-pairs | 10    | and_high_high      | D       | 40251.32   | 40651.74         | 0.990×  | 0.991/0.990     |
| x86/rgb-pairs | 10    | and_high_med       | D       | 20251.18   | 20482.19         | 0.989×  | 0.987/0.990     |
| x86/rgb-pairs | 10    | and_high_low       | D       | 1417.56    | 1421.25          | 0.997×  | 0.993/1.001     |
| x86/rgb-pairs | 10    | or_high_high       | D       | 49112.90   | 49262.37         | 0.997×  | 1.002/0.992     |
| x86/rgb-pairs | 10    | or_high_med        | D       | 20068.20   | 20157.30         | 0.996×  | 0.990/1.001     |
| x86/rgb-pairs | 10    | or_high_low        | D       | 2852.92    | 2840.44          | 1.004×  | 1.008/1.001     |
| x86/rgb-pairs | 10    | high_phrase        | D       | 1387.47    | 1383.26          | 1.003×  | 0.997/1.009     |
| x86/rgb-pairs | 10    | med_phrase         | D       | 3316.62    | 3317.73          | 1.000×  | 0.989/1.011     |
| x86/rgb-pairs | 10    | low_phrase         | D       | 6432.36    | 6382.64          | 1.008×  | 1.004/1.012     |
| x86/rgb-pairs | 10    | high_sloppy_phrase | D       | 18412.81   | 18267.31         | 1.008×  | 0.989/1.027     |
| x86/rgb-pairs | 10    | med_sloppy_phrase  | D       | 126874.68  | 126400.11        | 1.004×  | 1.006/1.002     |
| x86/rgb-pairs | 10    | low_sloppy_phrase  | D       | 212292.09  | 211010.93        | 1.006×  | 1.006/1.006     |
| x86/rgb-pairs | 10    | wildcard           | D       | 1863.45    | 1853.61          | 1.005×  | 1.002/1.008     |
| x86/rgb-pairs | 10    | prefix3            | D       | 163.90     | 164.87           | 0.994×  | 0.992/0.996     |
| x86/rgb-pairs | 10    | wildcard_scan      | D       | 4975.48    | 4961.38          | 1.003×  | 0.999/1.006     |
| x86/rgb-pairs | 10    | regex              | D       | 11947.43   | 11950.95         | 1.000×  | 1.004/0.995     |
| x86/rgb-pairs | 10    | multi_balanced     | D       | 816390.89  | 752315.77        | 1.085×  | 1.084/1.086     |
| x86/rgb-pairs | 10    | and_multi          | D       | 222832.39  | 217368.28        | 1.025×  | 1.024/1.027     |
| x86/rgb-pairs | 10    | multi_skewed       | D       | 242849.31  | 235228.38        | 1.032×  | 1.032/1.033     |
| x86/rgb-pairs | 10    | multi_wide         | D       | 753827.29  | 719054.10        | 1.048×  | 1.052/1.045     |
| x86/rgb-pairs | 10    | multi_ordered      | D       | 322319.50  | 296942.54        | 1.085×  | 1.083/1.088     |
| x86/rgb-pairs | 10    | multi_boundary     | D       | 576563.42  | 517315.09        | 1.115×  | 1.113/1.116     |
| x86/rgb-pairs | 100   | high_term          | D       | 3493.90    | 3437.92          | 1.016×  | 1.030/1.002     |
| x86/rgb-pairs | 100   | all                | D       | 3834417.90 | 3627467.92       | 1.057×  | 1.058/1.056     |
| x86/rgb-pairs | 100   | med_term           | D       | 2092.29    | 2068.43          | 1.012×  | 1.022/1.001     |
| x86/rgb-pairs | 100   | low_term           | D       | 752.65     | 747.41           | 1.007×  | 1.029/0.986     |
| x86/rgb-pairs | 100   | and_high_high      | D       | 80498.78   | 80550.74         | 0.999×  | 0.998/1.000     |
| x86/rgb-pairs | 100   | and_high_med       | D       | 32204.34   | 32131.20         | 1.002×  | 1.003/1.001     |
| x86/rgb-pairs | 100   | and_high_low       | D       | 1550.02    | 1540.87          | 1.006×  | 1.008/1.004     |
| x86/rgb-pairs | 100   | or_high_high       | D       | 95073.57   | 95791.34         | 0.993×  | 0.998/0.987     |
| x86/rgb-pairs | 100   | or_high_med        | D       | 34839.71   | 34894.88         | 0.998×  | 1.000/0.997     |
| x86/rgb-pairs | 100   | or_high_low        | D       | 8841.58    | 8947.68          | 0.988×  | 0.990/0.987     |
| x86/rgb-pairs | 100   | high_phrase        | D       | 6327.17    | 6401.29          | 0.988×  | 0.997/0.980     |
| x86/rgb-pairs | 100   | med_phrase         | D       | 12359.26   | 12514.55         | 0.988×  | 0.996/0.980     |
| x86/rgb-pairs | 100   | low_phrase         | D       | 18603.41   | 18955.40         | 0.981×  | 1.009/0.955     |
| x86/rgb-pairs | 100   | high_sloppy_phrase | D       | 43682.06   | 43396.58         | 1.007×  | 1.005/1.008     |
| x86/rgb-pairs | 100   | med_sloppy_phrase  | D       | 221672.26  | 221016.00        | 1.003×  | 1.003/1.003     |
| x86/rgb-pairs | 100   | low_sloppy_phrase  | D       | 287655.44  | 283901.82        | 1.013×  | 1.014/1.013     |
| x86/rgb-pairs | 100   | wildcard           | D       | 1880.72    | 1880.25          | 1.000×  | 0.999/1.001     |
| x86/rgb-pairs | 100   | prefix3            | D       | 175.85     | 175.87           | 1.000×  | 1.003/0.996     |
| x86/rgb-pairs | 100   | wildcard_scan      | D       | 5049.32    | 4989.52          | 1.012×  | 1.018/1.006     |
| x86/rgb-pairs | 100   | regex              | D       | 12106.70   | 11972.39         | 1.011×  | 1.009/1.013     |
| x86/rgb-pairs | 100   | multi_balanced     | D       | 817922.65  | 753240.13        | 1.086×  | 1.087/1.085     |
| x86/rgb-pairs | 100   | and_multi          | D       | 226100.01  | 225701.08        | 1.002×  | 1.001/1.003     |
| x86/rgb-pairs | 100   | multi_skewed       | D       | 244624.21  | 236887.53        | 1.033×  | 1.035/1.031     |
| x86/rgb-pairs | 100   | multi_wide         | D       | 773104.40  | 731043.69        | 1.058×  | 1.057/1.058     |
| x86/rgb-pairs | 100   | multi_ordered      | D       | 323293.90  | 298284.96        | 1.084×  | 1.083/1.085     |
| x86/rgb-pairs | 100   | multi_boundary     | D       | 580513.71  | 516996.38        | 1.123×  | 1.124/1.122     |
| x86/rgb-pairs | count | high_term          | D       | 4.44       | 4.40             | 1.010×  | 0.980/1.040     |
| x86/rgb-pairs | count | all                | D       | 2284634.95 | 2239183.63       | 1.020×  | 1.012/1.028     |
| x86/rgb-pairs | count | med_term           | D       | 5.68       | 5.48             | 1.037×  | 1.042/1.033     |
| x86/rgb-pairs | count | low_term           | D       | 5.75       | 5.42             | 1.061×  | 1.030/1.090     |
| x86/rgb-pairs | count | and_high_high      | D       | 40848.92   | 40811.70         | 1.001×  | 0.991/1.011     |
| x86/rgb-pairs | count | and_high_med       | D       | 26536.35   | 26431.01         | 1.004×  | 1.004/1.004     |
| x86/rgb-pairs | count | and_high_low       | D       | 533.37     | 531.25           | 1.004×  | 1.001/1.007     |
| x86/rgb-pairs | count | or_high_high       | D       | 42084.87   | 42009.15         | 1.002×  | 0.992/1.012     |
| x86/rgb-pairs | count | or_high_med        | D       | 26495.76   | 26493.97         | 1.000×  | 0.999/1.001     |
| x86/rgb-pairs | count | or_high_low        | D       | 808.65     | 800.24           | 1.011×  | 1.005/1.016     |
| x86/rgb-pairs | count | high_phrase        | D       | 14.46      | 14.52            | 0.996×  | 0.985/1.007     |
| x86/rgb-pairs | count | med_phrase         | D       | 206367.73  | 206328.92        | 1.000×  | 0.991/1.009     |
| x86/rgb-pairs | count | low_phrase         | D       | 123012.58  | 123466.41        | 0.996×  | 0.991/1.002     |
| x86/rgb-pairs | count | high_sloppy_phrase | D       | 589501.91  | 562798.50        | 1.047×  | 1.037/1.057     |
| x86/rgb-pairs | count | med_sloppy_phrase  | D       | 665684.02  | 652391.43        | 1.020×  | 1.012/1.029     |
| x86/rgb-pairs | count | low_sloppy_phrase  | D       | 319964.33  | 314910.44        | 1.016×  | 1.009/1.023     |
| x86/rgb-pairs | count | wildcard           | D       | 3947.47    | 3935.80          | 1.003×  | 0.995/1.011     |
| x86/rgb-pairs | count | prefix3            | D       | 2882.59    | 2901.25          | 0.994×  | 0.990/0.997     |
| x86/rgb-pairs | count | wildcard_scan      | D       | 6261.10    | 6111.53          | 1.024×  | 1.021/1.028     |
| x86/rgb-pairs | count | regex              | D       | 16870.22   | 16626.82         | 1.015×  | 1.005/1.025     |
| x86/rgb-pairs | count | multi_balanced     | D       | 52116.89   | 51280.50         | 1.016×  | 1.006/1.027     |
| x86/rgb-pairs | count | and_multi          | D       | 14642.96   | 14878.89         | 0.984×  | 0.945/1.025     |
| x86/rgb-pairs | count | multi_skewed       | D       | 46170.71   | 46814.25         | 0.986×  | 0.980/0.993     |
| x86/rgb-pairs | count | multi_wide         | D       | 62270.19   | 61844.03         | 1.007×  | 1.008/1.006     |
| x86/rgb-pairs | count | multi_ordered      | D       | 26384.33   | 26489.44         | 0.996×  | 1.006/0.987     |
| x86/rgb-pairs | count | multi_boundary     | D       | 11219.67   | 11298.30         | 0.993×  | 0.993/0.994     |

## Broad queries

| Host/index    | Limit | Family             | Variant | A sum µs   | Candidate sum µs | Speedup | Forward/reverse |
| ------------- | ----- | ------------------ | ------- | ---------- | ---------------- | ------- | --------------- |
| arm/bitmap    | 10    | high_term          | B       | 161.60     | 162.31           | 0.996×  | 0.980/1.011     |
| arm/bitmap    | 10    | all                | B       | 126710.51  | 126344.27        | 1.003×  | 1.006/1.000     |
| arm/bitmap    | 10    | med_term           | B       | 116.77     | 107.81           | 1.083×  | 0.995/1.171     |
| arm/bitmap    | 10    | low_term           | B       | 64.83      | 64.44            | 1.006×  | 0.985/1.027     |
| arm/bitmap    | 10    | and_high_high      | B       | 5163.40    | 5079.02          | 1.017×  | 0.976/1.058     |
| arm/bitmap    | 10    | and_high_med       | B       | 2325.06    | 2267.60          | 1.025×  | 1.046/1.004     |
| arm/bitmap    | 10    | and_high_low       | B       | 455.64     | 229.79           | 1.983×  | 2.928/1.030     |
| arm/bitmap    | 10    | or_high_high       | B       | 10258.35   | 10046.67         | 1.021×  | 1.061/0.982     |
| arm/bitmap    | 10    | or_high_med        | B       | 2677.35    | 2709.50          | 0.988×  | 0.986/0.990     |
| arm/bitmap    | 10    | or_high_low        | B       | 601.92     | 591.19           | 1.018×  | 1.031/1.005     |
| arm/bitmap    | 10    | high_phrase        | B       | 17853.33   | 17627.50         | 1.013×  | 1.021/1.005     |
| arm/bitmap    | 10    | med_phrase         | B       | 26838.38   | 26425.67         | 1.016×  | 1.010/1.021     |
| arm/bitmap    | 10    | low_phrase         | B       | 2487.44    | 2535.02          | 0.981×  | 0.997/0.966     |
| arm/bitmap    | 10    | high_sloppy_phrase | B       | 1985.54    | 1926.94          | 1.030×  | 1.051/1.009     |
| arm/bitmap    | 10    | med_sloppy_phrase  | B       | 12154.85   | 12168.21         | 0.999×  | 0.998/1.000     |
| arm/bitmap    | 10    | low_sloppy_phrase  | B       | 17029.35   | 16828.14         | 1.012×  | 1.029/0.995     |
| arm/bitmap    | 10    | wildcard           | B       | 232.27     | 229.69           | 1.011×  | 1.009/1.013     |
| arm/bitmap    | 10    | prefix3            | B       | 31.92      | 32.40            | 0.985×  | 0.997/0.973     |
| arm/bitmap    | 10    | wildcard_scan      | B       | 967.21     | 966.46           | 1.001×  | 0.999/1.003     |
| arm/bitmap    | 10    | regex              | B       | 1347.25    | 1361.29          | 0.990×  | 0.977/1.002     |
| arm/bitmap    | 10    | and_multi          | B       | 23958.04   | 24984.62         | 0.959×  | 0.943/0.975     |
| arm/bitmap    | 100   | high_term          | B       | 740.60     | 731.56           | 1.012×  | 1.030/0.995     |
| arm/bitmap    | 100   | all                | B       | 240572.58  | 242770.00        | 0.991×  | 0.990/0.992     |
| arm/bitmap    | 100   | med_term           | B       | 450.29     | 442.29           | 1.018×  | 1.028/1.008     |
| arm/bitmap    | 100   | low_term           | B       | 160.23     | 160.96           | 0.995×  | 1.010/0.981     |
| arm/bitmap    | 100   | and_high_high      | B       | 8660.12    | 8698.15          | 0.996×  | 0.988/1.004     |
| arm/bitmap    | 100   | and_high_med       | B       | 3032.04    | 3085.10          | 0.983×  | 0.993/0.973     |
| arm/bitmap    | 100   | and_high_low       | B       | 233.35     | 231.61           | 1.008×  | 1.002/1.013     |
| arm/bitmap    | 100   | or_high_high       | B       | 14653.71   | 14667.31         | 0.999×  | 0.994/1.004     |
| arm/bitmap    | 100   | or_high_med        | B       | 4291.56    | 4249.42          | 1.010×  | 1.022/0.998     |
| arm/bitmap    | 100   | or_high_low        | B       | 2332.79    | 2343.46          | 0.995×  | 0.986/1.005     |
| arm/bitmap    | 100   | high_phrase        | B       | 72160.98   | 73464.60         | 0.982×  | 0.987/0.978     |
| arm/bitmap    | 100   | med_phrase         | B       | 54037.54   | 53854.42         | 1.003×  | 1.010/0.997     |
| arm/bitmap    | 100   | low_phrase         | B       | 6930.42    | 6607.75          | 1.049×  | 1.005/1.094     |
| arm/bitmap    | 100   | high_sloppy_phrase | B       | 5063.83    | 5034.71          | 1.006×  | 1.009/1.003     |
| arm/bitmap    | 100   | med_sloppy_phrase  | B       | 19986.31   | 19921.27         | 1.003×  | 0.991/1.015     |
| arm/bitmap    | 100   | low_sloppy_phrase  | B       | 21147.85   | 21257.48         | 0.995×  | 0.998/0.991     |
| arm/bitmap    | 100   | wildcard           | B       | 237.94     | 245.56           | 0.969×  | 0.962/0.976     |
| arm/bitmap    | 100   | prefix3            | B       | 36.23      | 37.42            | 0.968×  | 1.051/0.893     |
| arm/bitmap    | 100   | wildcard_scan      | B       | 964.31     | 960.77           | 1.004×  | 0.989/1.018     |
| arm/bitmap    | 100   | regex              | B       | 1342.96    | 1408.98          | 0.953×  | 0.933/0.974     |
| arm/bitmap    | 100   | and_multi          | B       | 24109.50   | 25367.19         | 0.950×  | 0.937/0.965     |
| arm/bitmap    | count | high_term          | B       | 3.08       | 3.06             | 1.007×  | 1.042/0.974     |
| arm/bitmap    | count | all                | B       | 502268.37  | 512290.52        | 0.980×  | 0.974/0.987     |
| arm/bitmap    | count | med_term           | B       | 3.17       | 2.77             | 1.143×  | 1.000/1.279     |
| arm/bitmap    | count | low_term           | B       | 3.00       | 3.00             | 1.000×  | 0.986/1.014     |
| arm/bitmap    | count | and_high_high      | B       | 3749.29    | 3693.71          | 1.015×  | 0.991/1.040     |
| arm/bitmap    | count | and_high_med       | B       | 1804.35    | 1794.17          | 1.006×  | 1.012/1.000     |
| arm/bitmap    | count | and_high_low       | B       | 108.31     | 108.81           | 0.995×  | 0.999/0.992     |
| arm/bitmap    | count | or_high_high       | B       | 3443.61    | 3452.73          | 0.997×  | 0.995/1.000     |
| arm/bitmap    | count | or_high_med        | B       | 1954.50    | 1947.44          | 1.004×  | 1.000/1.007     |
| arm/bitmap    | count | or_high_low        | B       | 367.25     | 368.27           | 0.997×  | 1.002/0.992     |
| arm/bitmap    | count | high_phrase        | B       | 270853.31  | 276958.54        | 0.978×  | 0.966/0.990     |
| arm/bitmap    | count | med_phrase         | B       | 97690.98   | 99775.21         | 0.979×  | 0.983/0.975     |
| arm/bitmap    | count | low_phrase         | B       | 15989.71   | 15932.50         | 1.004×  | 1.007/1.000     |
| arm/bitmap    | count | high_sloppy_phrase | B       | 39597.75   | 41232.44         | 0.960×  | 0.963/0.958     |
| arm/bitmap    | count | med_sloppy_phrase  | B       | 41675.33   | 41629.77         | 1.001×  | 0.996/1.006     |
| arm/bitmap    | count | low_sloppy_phrase  | B       | 21138.75   | 21394.33         | 0.988×  | 0.985/0.992     |
| arm/bitmap    | count | wildcard           | B       | 393.52     | 401.19           | 0.981×  | 0.990/0.971     |
| arm/bitmap    | count | prefix3            | B       | 262.58     | 266.52           | 0.985×  | 0.971/1.000     |
| arm/bitmap    | count | wildcard_scan      | B       | 1013.29    | 1020.46          | 0.993×  | 0.967/1.020     |
| arm/bitmap    | count | regex              | B       | 1489.94    | 1549.39          | 0.962×  | 0.893/1.038     |
| arm/bitmap    | count | and_multi          | B       | 726.65     | 756.21           | 0.961×  | 0.885/1.045     |
| arm/rgb       | 10    | high_term          | B       | 110.19     | 108.90           | 1.012×  | 1.006/1.018     |
| arm/rgb       | 10    | all                | B       | 105214.85  | 105753.52        | 0.995×  | 1.004/0.986     |
| arm/rgb       | 10    | med_term           | B       | 66.35      | 67.25            | 0.987×  | 0.977/0.997     |
| arm/rgb       | 10    | low_term           | B       | 41.58      | 40.96            | 1.015×  | 0.993/1.038     |
| arm/rgb       | 10    | and_high_high      | B       | 3119.25    | 3174.68          | 0.983×  | 0.989/0.976     |
| arm/rgb       | 10    | and_high_med       | B       | 1707.12    | 1734.44          | 0.984×  | 0.991/0.978     |
| arm/rgb       | 10    | and_high_low       | B       | 173.38     | 172.69           | 1.004×  | 0.997/1.011     |
| arm/rgb       | 10    | or_high_high       | B       | 4612.67    | 4164.63          | 1.108×  | 1.014/1.200     |
| arm/rgb       | 10    | or_high_med        | B       | 1719.56    | 1702.92          | 1.010×  | 0.974/1.046     |
| arm/rgb       | 10    | or_high_low        | B       | 361.85     | 371.12           | 0.975×  | 1.012/0.938     |
| arm/rgb       | 10    | high_phrase        | B       | 8880.77    | 8598.40          | 1.033×  | 1.007/1.058     |
| arm/rgb       | 10    | med_phrase         | B       | 25542.27   | 25627.92         | 0.997×  | 0.997/0.997     |
| arm/rgb       | 10    | low_phrase         | B       | 2074.71    | 2008.67          | 1.033×  | 1.093/0.974     |
| arm/rgb       | 10    | high_sloppy_phrase | B       | 1735.90    | 1693.33          | 1.025×  | 1.034/1.016     |
| arm/rgb       | 10    | med_sloppy_phrase  | B       | 11907.52   | 12023.44         | 0.990×  | 0.987/0.994     |
| arm/rgb       | 10    | low_sloppy_phrase  | B       | 15728.56   | 16450.50         | 0.956×  | 1.004/0.912     |
| arm/rgb       | 10    | wildcard           | B       | 220.56     | 229.08           | 0.963×  | 0.966/0.960     |
| arm/rgb       | 10    | prefix3            | B       | 39.25      | 41.19            | 0.953×  | 0.847/1.070     |
| arm/rgb       | 10    | wildcard_scan      | B       | 1134.94    | 1095.65          | 1.036×  | 0.997/1.072     |
| arm/rgb       | 10    | regex              | B       | 1503.75    | 1563.21          | 0.962×  | 0.988/0.937     |
| arm/rgb       | 10    | and_multi          | B       | 24534.67   | 24884.56         | 0.986×  | 1.014/0.959     |
| arm/rgb       | 100   | high_term          | B       | 506.71     | 494.85           | 1.024×  | 1.013/1.035     |
| arm/rgb       | 100   | all                | B       | 197567.87  | 198471.94        | 0.995×  | 0.996/0.995     |
| arm/rgb       | 100   | med_term           | B       | 266.98     | 263.62           | 1.013×  | 1.004/1.021     |
| arm/rgb       | 100   | low_term           | B       | 148.60     | 148.35           | 1.002×  | 1.008/0.995     |
| arm/rgb       | 100   | and_high_high      | B       | 5596.27    | 5831.42          | 0.960×  | 1.011/0.912     |
| arm/rgb       | 100   | and_high_med       | B       | 2404.06    | 2469.62          | 0.973×  | 0.996/0.952     |
| arm/rgb       | 100   | and_high_low       | B       | 176.48     | 178.86           | 0.987×  | 0.999/0.975     |
| arm/rgb       | 100   | or_high_high       | B       | 7994.12    | 8144.52          | 0.982×  | 1.003/0.961     |
| arm/rgb       | 100   | or_high_med        | B       | 3109.56    | 3071.02          | 1.013×  | 1.021/1.004     |
| arm/rgb       | 100   | or_high_low        | B       | 1411.92    | 1389.75          | 1.016×  | 1.038/0.994     |
| arm/rgb       | 100   | high_phrase        | B       | 48883.35   | 49661.71         | 0.984×  | 0.993/0.976     |
| arm/rgb       | 100   | med_phrase         | B       | 51541.69   | 51446.69         | 1.002×  | 1.008/0.995     |
| arm/rgb       | 100   | low_phrase         | B       | 5879.12    | 5917.92          | 0.993×  | 1.003/0.984     |
| arm/rgb       | 100   | high_sloppy_phrase | B       | 3992.17    | 3893.98          | 1.025×  | 0.997/1.053     |
| arm/rgb       | 100   | med_sloppy_phrase  | B       | 19473.98   | 18746.50         | 1.039×  | 0.995/1.083     |
| arm/rgb       | 100   | low_sloppy_phrase  | B       | 19093.92   | 19360.94         | 0.986×  | 0.980/0.993     |
| arm/rgb       | 100   | wildcard           | B       | 240.94     | 241.71           | 0.997×  | 1.012/0.983     |
| arm/rgb       | 100   | prefix3            | B       | 49.21      | 48.48            | 1.015×  | 0.975/1.057     |
| arm/rgb       | 100   | wildcard_scan      | B       | 1052.10    | 1084.90          | 0.970×  | 0.978/0.961     |
| arm/rgb       | 100   | regex              | B       | 1493.73    | 1550.23          | 0.964×  | 0.958/0.969     |
| arm/rgb       | 100   | and_multi          | B       | 24252.96   | 24526.87         | 0.989×  | 0.977/1.001     |
| arm/rgb       | count | high_term          | B       | 2.81       | 3.75             | 0.750×  | 0.736/0.764     |
| arm/rgb       | count | all                | B       | 477251.96  | 484564.99        | 0.985×  | 0.984/0.986     |
| arm/rgb       | count | med_term           | B       | 2.77       | 3.44             | 0.806×  | 0.810/0.803     |
| arm/rgb       | count | low_term           | B       | 2.40       | 2.40             | 1.000×  | 0.982/1.018     |
| arm/rgb       | count | and_high_high      | B       | 2537.29    | 2674.42          | 0.949×  | 0.999/0.903     |
| arm/rgb       | count | and_high_med       | B       | 1361.63    | 1395.48          | 0.976×  | 0.997/0.956     |
| arm/rgb       | count | and_high_low       | B       | 70.25      | 71.17            | 0.987×  | 0.977/0.997     |
| arm/rgb       | count | or_high_high       | B       | 2579.90    | 2655.71          | 0.971×  | 0.995/0.949     |
| arm/rgb       | count | or_high_med        | B       | 1467.67    | 1518.44          | 0.967×  | 0.992/0.943     |
| arm/rgb       | count | or_high_low        | B       | 285.00     | 290.00           | 0.983×  | 0.996/0.970     |
| arm/rgb       | count | high_phrase        | B       | 262257.60  | 266544.73        | 0.984×  | 0.984/0.984     |
| arm/rgb       | count | med_phrase         | B       | 92369.50   | 94415.81         | 0.978×  | 0.967/0.990     |
| arm/rgb       | count | low_phrase         | B       | 13895.92   | 13712.60         | 1.013×  | 1.024/1.003     |
| arm/rgb       | count | high_sloppy_phrase | B       | 38040.54   | 39210.39         | 0.970×  | 0.981/0.960     |
| arm/rgb       | count | med_sloppy_phrase  | B       | 39076.00   | 38681.77         | 1.010×  | 1.004/1.016     |
| arm/rgb       | count | low_sloppy_phrase  | B       | 19128.15   | 19142.83         | 0.999×  | 1.005/0.993     |
| arm/rgb       | count | wildcard           | B       | 327.27     | 325.62           | 1.005×  | 1.017/0.993     |
| arm/rgb       | count | prefix3            | B       | 277.02     | 285.00           | 0.972×  | 0.966/0.978     |
| arm/rgb       | count | wildcard_scan      | B       | 1109.06    | 1105.19          | 1.004×  | 1.009/0.998     |
| arm/rgb       | count | regex              | B       | 1704.10    | 1781.35          | 0.957×  | 1.022/0.899     |
| arm/rgb       | count | and_multi          | B       | 757.08     | 744.90           | 1.016×  | 1.014/1.019     |
| x86/default   | 10    | high_term          | B       | 835.39     | 817.64           | 1.022×  | 1.022/1.021     |
| x86/default   | 10    | all                | B       | 1332528.79 | 1335502.52       | 0.998×  | 1.001/0.994     |
| x86/default   | 10    | med_term           | B       | 259.78     | 258.84           | 1.004×  | 1.006/1.001     |
| x86/default   | 10    | low_term           | B       | 147.88     | 147.35           | 1.004×  | 1.004/1.003     |
| x86/default   | 10    | and_high_high      | B       | 71280.04   | 69884.31         | 1.020×  | 1.026/1.014     |
| x86/default   | 10    | and_high_med       | B       | 38646.56   | 38301.48         | 1.009×  | 1.028/0.991     |
| x86/default   | 10    | and_high_low       | B       | 2866.60    | 2857.09          | 1.003×  | 1.007/1.000     |
| x86/default   | 10    | or_high_high       | B       | 113413.95  | 114517.49        | 0.990×  | 0.994/0.987     |
| x86/default   | 10    | or_high_med        | B       | 41273.91   | 41829.24         | 0.987×  | 0.984/0.989     |
| x86/default   | 10    | or_high_low        | B       | 5701.55    | 5711.05          | 0.998×  | 1.008/0.989     |
| x86/default   | 10    | high_phrase        | B       | 59607.38   | 59360.60         | 1.004×  | 1.007/1.001     |
| x86/default   | 10    | med_phrase         | B       | 181739.58  | 181589.83        | 1.001×  | 1.002/0.999     |
| x86/default   | 10    | low_phrase         | B       | 25970.41   | 26003.41         | 0.999×  | 1.006/0.992     |
| x86/default   | 10    | high_sloppy_phrase | B       | 16080.11   | 16247.09         | 0.990×  | 0.995/0.985     |
| x86/default   | 10    | med_sloppy_phrase  | B       | 133123.23  | 133228.87        | 0.999×  | 1.004/0.994     |
| x86/default   | 10    | low_sloppy_phrase  | B       | 212896.19  | 214525.40        | 0.992×  | 0.991/0.993     |
| x86/default   | 10    | wildcard           | B       | 2162.70    | 2156.38          | 1.003×  | 1.019/0.987     |
| x86/default   | 10    | prefix3            | B       | 179.18     | 180.92           | 0.990×  | 1.002/0.979     |
| x86/default   | 10    | wildcard_scan      | B       | 4791.40    | 4922.54          | 0.973×  | 0.976/0.971     |
| x86/default   | 10    | regex              | B       | 11615.96   | 11768.97         | 0.987×  | 0.997/0.977     |
| x86/default   | 10    | and_multi          | B       | 409937.02  | 411194.00        | 0.997×  | 1.001/0.993     |
| x86/default   | 100   | high_term          | B       | 2420.84    | 2365.27          | 1.023×  | 1.028/1.019     |
| x86/default   | 100   | all                | B       | 2538714.58 | 2537753.90       | 1.000×  | 1.004/0.997     |
| x86/default   | 100   | med_term           | B       | 1342.35    | 1330.30          | 1.009×  | 1.010/1.008     |
| x86/default   | 100   | low_term           | B       | 816.06     | 787.28           | 1.037×  | 1.037/1.036     |
| x86/default   | 100   | and_high_high      | B       | 132004.95  | 129440.47        | 1.020×  | 1.022/1.018     |
| x86/default   | 100   | and_high_med       | B       | 53339.12   | 53987.60         | 0.988×  | 1.005/0.971     |
| x86/default   | 100   | and_high_low       | B       | 3248.66    | 3283.19          | 0.989×  | 0.983/0.996     |
| x86/default   | 100   | or_high_high       | B       | 162777.42  | 165153.68        | 0.986×  | 0.989/0.983     |
| x86/default   | 100   | or_high_med        | B       | 55744.39   | 58630.44         | 0.951×  | 0.959/0.943     |
| x86/default   | 100   | or_high_low        | B       | 11356.80   | 11275.81         | 1.007×  | 0.987/1.028     |
| x86/default   | 100   | high_phrase        | B       | 330275.35  | 332553.70        | 0.993×  | 0.996/0.990     |
| x86/default   | 100   | med_phrase         | B       | 621654.80  | 621626.08        | 1.000×  | 1.002/0.998     |
| x86/default   | 100   | low_phrase         | B       | 81019.29   | 84490.03         | 0.959×  | 0.988/0.931     |
| x86/default   | 100   | high_sloppy_phrase | B       | 43382.08   | 44421.35         | 0.977×  | 0.985/0.969     |
| x86/default   | 100   | med_sloppy_phrase  | B       | 250118.61  | 253380.82        | 0.987×  | 0.987/0.988     |
| x86/default   | 100   | low_sloppy_phrase  | B       | 329559.34  | 331094.07        | 0.995×  | 1.000/0.991     |
| x86/default   | 100   | wildcard           | B       | 1948.91    | 1920.66          | 1.015×  | 1.012/1.018     |
| x86/default   | 100   | prefix3            | B       | 190.16     | 190.27           | 0.999×  | 0.998/1.001     |
| x86/default   | 100   | wildcard_scan      | B       | 4832.31    | 4992.36          | 0.968×  | 0.990/0.947     |
| x86/default   | 100   | regex              | B       | 11620.55   | 11902.97         | 0.976×  | 0.963/0.990     |
| x86/default   | 100   | and_multi          | B       | 441062.60  | 424927.55        | 1.038×  | 1.039/1.037     |
| x86/default   | count | high_term          | B       | 4.44       | 7.13             | 0.623×  | 0.976/0.459     |
| x86/default   | count | all                | B       | 8570042.19 | 8657367.02       | 0.990×  | 0.993/0.987     |
| x86/default   | count | med_term           | B       | 5.61       | 9.27             | 0.605×  | 0.968/0.442     |
| x86/default   | count | low_term           | B       | 5.36       | 8.67             | 0.618×  | 1.013/0.444     |
| x86/default   | count | and_high_high      | B       | 56357.44   | 55709.75         | 1.012×  | 1.007/1.016     |
| x86/default   | count | and_high_med       | B       | 36892.27   | 36776.75         | 1.003×  | 0.994/1.012     |
| x86/default   | count | and_high_low       | B       | 1378.70    | 1289.66          | 1.069×  | 1.025/1.113     |
| x86/default   | count | or_high_high       | B       | 54421.21   | 53018.14         | 1.026×  | 1.029/1.024     |
| x86/default   | count | or_high_med        | B       | 34333.10   | 34244.64         | 1.003×  | 1.005/1.001     |
| x86/default   | count | or_high_low        | B       | 2061.60    | 2012.82          | 1.024×  | 1.028/1.020     |
| x86/default   | count | high_phrase        | B       | 4547015.35 | 4586313.94       | 0.991×  | 0.993/0.990     |
| x86/default   | count | med_phrase         | B       | 1726169.87 | 1755526.52       | 0.983×  | 0.992/0.975     |
| x86/default   | count | low_phrase         | B       | 319927.60  | 320870.37        | 0.997×  | 0.998/0.996     |
| x86/default   | count | high_sloppy_phrase | B       | 647229.31  | 654857.97        | 0.988×  | 0.994/0.982     |
| x86/default   | count | med_sloppy_phrase  | B       | 726191.69  | 735701.23        | 0.987×  | 0.987/0.987     |
| x86/default   | count | low_sloppy_phrase  | B       | 370003.67  | 371911.22        | 0.995×  | 0.990/0.999     |
| x86/default   | count | wildcard           | B       | 5646.40    | 5440.42          | 1.038×  | 1.077/0.998     |
| x86/default   | count | prefix3            | B       | 2806.90    | 2838.90          | 0.989×  | 0.980/0.998     |
| x86/default   | count | wildcard_scan      | B       | 5734.86    | 5685.95          | 1.009×  | 1.010/1.007     |
| x86/default   | count | regex              | B       | 14954.80   | 15252.82         | 0.980×  | 0.979/0.982     |
| x86/default   | count | and_multi          | B       | 18902.03   | 19890.84         | 0.950×  | 0.942/0.959     |
| x86/rgb-pairs | 10    | high_term          | B       | 948.24     | 922.76           | 1.028×  | 1.035/1.020     |
| x86/rgb-pairs | 10    | all                | B       | 935397.77  | 930702.05        | 1.005×  | 1.004/1.006     |
| x86/rgb-pairs | 10    | med_term           | B       | 304.64     | 302.49           | 1.007×  | 1.010/1.004     |
| x86/rgb-pairs | 10    | low_term           | B       | 147.21     | 147.13           | 1.001×  | 1.002/1.000     |
| x86/rgb-pairs | 10    | and_high_high      | B       | 40179.33   | 39886.37         | 1.007×  | 0.998/1.016     |
| x86/rgb-pairs | 10    | and_high_med       | B       | 19962.21   | 20047.99         | 0.996×  | 0.982/1.010     |
| x86/rgb-pairs | 10    | and_high_low       | B       | 1440.52    | 1503.83          | 0.958×  | 0.919/1.000     |
| x86/rgb-pairs | 10    | or_high_high       | B       | 48086.00   | 48053.13         | 1.001×  | 1.000/1.002     |
| x86/rgb-pairs | 10    | or_high_med        | B       | 20037.84   | 19570.26         | 1.024×  | 1.005/1.043     |
| x86/rgb-pairs | 10    | or_high_low        | B       | 2732.69    | 2758.37          | 0.991×  | 0.985/0.996     |
| x86/rgb-pairs | 10    | high_phrase        | B       | 1142.31    | 1130.31          | 1.011×  | 1.009/1.012     |
| x86/rgb-pairs | 10    | med_phrase         | B       | 2979.21    | 2972.91          | 1.002×  | 1.000/1.004     |
| x86/rgb-pairs | 10    | low_phrase         | B       | 6193.39    | 6070.65          | 1.020×  | 1.016/1.024     |
| x86/rgb-pairs | 10    | high_sloppy_phrase | B       | 17814.40   | 17735.22         | 1.004×  | 1.003/1.006     |
| x86/rgb-pairs | 10    | med_sloppy_phrase  | B       | 127694.18  | 127799.00        | 0.999×  | 1.000/0.999     |
| x86/rgb-pairs | 10    | low_sloppy_phrase  | B       | 215529.85  | 216551.84        | 0.995×  | 0.997/0.993     |
| x86/rgb-pairs | 10    | wildcard           | B       | 2145.18    | 1973.67          | 1.087×  | 1.024/1.150     |
| x86/rgb-pairs | 10    | prefix3            | B       | 177.42     | 181.29           | 0.979×  | 0.996/0.962     |
| x86/rgb-pairs | 10    | wildcard_scan      | B       | 5747.99    | 5667.47          | 1.014×  | 0.938/1.100     |
| x86/rgb-pairs | 10    | regex              | B       | 12520.91   | 12561.07         | 0.997×  | 1.007/0.987     |
| x86/rgb-pairs | 10    | and_multi          | B       | 409614.24  | 404866.29        | 1.012×  | 1.012/1.012     |
| x86/rgb-pairs | 100   | high_term          | B       | 2910.22    | 2814.69          | 1.034×  | 1.034/1.034     |
| x86/rgb-pairs | 100   | all                | B       | 1280133.91 | 1279649.80       | 1.000×  | 1.003/0.998     |
| x86/rgb-pairs | 100   | med_term           | B       | 1645.43    | 1627.62          | 1.011×  | 1.012/1.010     |
| x86/rgb-pairs | 100   | low_term           | B       | 612.87     | 608.87           | 1.007×  | 1.011/1.003     |
| x86/rgb-pairs | 100   | and_high_high      | B       | 78909.87   | 77320.34         | 1.021×  | 1.029/1.013     |
| x86/rgb-pairs | 100   | and_high_med       | B       | 31622.11   | 31179.32         | 1.014×  | 1.020/1.009     |
| x86/rgb-pairs | 100   | and_high_low       | B       | 1574.19    | 1578.97          | 0.997×  | 0.996/0.998     |
| x86/rgb-pairs | 100   | or_high_high       | B       | 93569.86   | 91569.21         | 1.022×  | 1.031/1.013     |
| x86/rgb-pairs | 100   | or_high_med        | B       | 34259.39   | 33519.92         | 1.022×  | 1.022/1.022     |
| x86/rgb-pairs | 100   | or_high_low        | B       | 8235.24    | 8149.40          | 1.011×  | 1.021/1.001     |
| x86/rgb-pairs | 100   | high_phrase        | B       | 5068.62    | 4981.05          | 1.018×  | 1.021/1.015     |
| x86/rgb-pairs | 100   | med_phrase         | B       | 11594.07   | 11744.04         | 0.987×  | 0.980/0.995     |
| x86/rgb-pairs | 100   | low_phrase         | B       | 19767.31   | 19190.53         | 1.030×  | 1.095/0.966     |
| x86/rgb-pairs | 100   | high_sloppy_phrase | B       | 43033.38   | 42580.29         | 1.011×  | 1.002/1.019     |
| x86/rgb-pairs | 100   | med_sloppy_phrase  | B       | 220448.44  | 221728.53        | 0.994×  | 0.991/0.997     |
| x86/rgb-pairs | 100   | low_sloppy_phrase  | B       | 289837.70  | 292652.22        | 0.990×  | 0.993/0.987     |
| x86/rgb-pairs | 100   | wildcard           | B       | 2099.17    | 2081.77          | 1.008×  | 1.013/1.004     |
| x86/rgb-pairs | 100   | prefix3            | B       | 189.87     | 187.03           | 1.015×  | 1.016/1.015     |
| x86/rgb-pairs | 100   | wildcard_scan      | B       | 5537.26    | 5774.21          | 0.959×  | 0.940/0.978     |
| x86/rgb-pairs | 100   | regex              | B       | 12668.64   | 12577.93         | 1.007×  | 1.005/1.010     |
| x86/rgb-pairs | 100   | and_multi          | B       | 416550.28  | 417783.85        | 0.997×  | 0.998/0.996     |
| x86/rgb-pairs | count | high_term          | B       | 4.44       | 6.07             | 0.731×  | 0.963/0.591     |
| x86/rgb-pairs | count | all                | B       | 2143683.97 | 2142139.75       | 1.001×  | 0.998/1.003     |
| x86/rgb-pairs | count | med_term           | B       | 5.59       | 5.79             | 0.966×  | 0.984/0.948     |
| x86/rgb-pairs | count | low_term           | B       | 5.34       | 5.31             | 1.005×  | 1.006/1.005     |
| x86/rgb-pairs | count | and_high_high      | B       | 40721.65   | 39893.98         | 1.021×  | 1.025/1.016     |
| x86/rgb-pairs | count | and_high_med       | B       | 26761.20   | 26495.21         | 1.010×  | 1.007/1.013     |
| x86/rgb-pairs | count | and_high_low       | B       | 539.97     | 539.12           | 1.002×  | 1.000/1.003     |
| x86/rgb-pairs | count | or_high_high       | B       | 42214.88   | 41445.16         | 1.019×  | 1.035/1.003     |
| x86/rgb-pairs | count | or_high_med        | B       | 27007.20   | 26827.27         | 1.007×  | 1.011/1.003     |
| x86/rgb-pairs | count | or_high_low        | B       | 820.65     | 812.37           | 1.010×  | 1.014/1.007     |
| x86/rgb-pairs | count | high_phrase        | B       | 22.12      | 18.72            | 1.182×  | 1.465/0.986     |
| x86/rgb-pairs | count | med_phrase         | B       | 210808.46  | 211397.77        | 0.997×  | 1.006/0.989     |
| x86/rgb-pairs | count | low_phrase         | B       | 126012.25  | 126681.80        | 0.995×  | 1.002/0.988     |
| x86/rgb-pairs | count | high_sloppy_phrase | B       | 609231.16  | 605632.17        | 1.006×  | 1.000/1.012     |
| x86/rgb-pairs | count | med_sloppy_phrase  | B       | 684837.59  | 687474.64        | 0.996×  | 0.989/1.003     |
| x86/rgb-pairs | count | low_sloppy_phrase  | B       | 323542.77  | 324268.33        | 0.998×  | 0.994/1.002     |
| x86/rgb-pairs | count | wildcard           | B       | 4263.89    | 4086.67          | 1.043×  | 1.086/1.001     |
| x86/rgb-pairs | count | prefix3            | B       | 2610.37    | 2579.11          | 1.012×  | 1.004/1.020     |
| x86/rgb-pairs | count | wildcard_scan      | B       | 6024.31    | 5884.74          | 1.024×  | 1.048/0.999     |
| x86/rgb-pairs | count | regex              | B       | 16300.29   | 16470.95         | 0.990×  | 0.992/0.987     |
| x86/rgb-pairs | count | and_multi          | B       | 21949.83   | 21614.56         | 1.016×  | 1.019/1.012     |

## Thirty multi-term conjunctions

| Host/index    | Limit | Family         | Variant | A sum µs   | Candidate sum µs | Speedup | Forward/reverse |
| ------------- | ----- | -------------- | ------- | ---------- | ---------------- | ------- | --------------- |
| arm/bitmap    | 10    | multi_balanced | B       | 41922.92   | 42433.75         | 0.988×  | 1.002/0.974     |
| arm/bitmap    | 10    | all            | B       | 142238.16  | 142009.56        | 1.002×  | 1.010/0.994     |
| arm/bitmap    | 10    | multi_skewed   | B       | 13906.42   | 13822.94         | 1.006×  | 1.009/1.003     |
| arm/bitmap    | 10    | multi_wide     | B       | 41821.33   | 41855.12         | 0.999×  | 1.017/0.982     |
| arm/bitmap    | 10    | multi_ordered  | B       | 16901.88   | 16754.29         | 1.009×  | 1.014/1.003     |
| arm/bitmap    | 10    | multi_boundary | B       | 27685.62   | 27143.46         | 1.020×  | 1.008/1.032     |
| arm/bitmap    | 10    | multi_balanced | C       | 41922.92   | 41635.19         | 1.007×  | 1.016/0.998     |
| arm/bitmap    | 10    | all            | C       | 142238.16  | 141256.75        | 1.007×  | 1.009/1.005     |
| arm/bitmap    | 10    | multi_skewed   | C       | 13906.42   | 13802.73         | 1.008×  | 1.005/1.010     |
| arm/bitmap    | 10    | multi_wide     | C       | 41821.33   | 42060.94         | 0.994×  | 0.995/0.994     |
| arm/bitmap    | 10    | multi_ordered  | C       | 16901.88   | 16627.94         | 1.016×  | 1.024/1.009     |
| arm/bitmap    | 10    | multi_boundary | C       | 27685.62   | 27129.96         | 1.020×  | 1.013/1.027     |
| arm/bitmap    | 100   | multi_balanced | B       | 42562.10   | 42574.13         | 1.000×  | 0.981/1.018     |
| arm/bitmap    | 100   | all            | B       | 143798.48  | 143217.04        | 1.004×  | 0.996/1.012     |
| arm/bitmap    | 100   | multi_skewed   | B       | 14316.56   | 14168.31         | 1.010×  | 0.986/1.035     |
| arm/bitmap    | 100   | multi_wide     | B       | 42515.98   | 42416.81         | 1.002×  | 1.006/0.999     |
| arm/bitmap    | 100   | multi_ordered  | B       | 16907.60   | 16855.58         | 1.003×  | 0.996/1.010     |
| arm/bitmap    | 100   | multi_boundary | B       | 27496.23   | 27202.21         | 1.011×  | 1.008/1.014     |
| arm/bitmap    | 100   | multi_balanced | C       | 42562.10   | 43133.41         | 0.987×  | 1.005/0.970     |
| arm/bitmap    | 100   | all            | C       | 143798.48  | 143470.96        | 1.002×  | 0.997/1.008     |
| arm/bitmap    | 100   | multi_skewed   | C       | 14316.56   | 14574.10         | 0.982×  | 0.999/0.967     |
| arm/bitmap    | 100   | multi_wide     | C       | 42515.98   | 41864.35         | 1.016×  | 0.986/1.046     |
| arm/bitmap    | 100   | multi_ordered  | C       | 16907.60   | 16885.40         | 1.001×  | 0.980/1.023     |
| arm/bitmap    | 100   | multi_boundary | C       | 27496.23   | 27013.69         | 1.018×  | 1.011/1.025     |
| arm/bitmap    | count | multi_balanced | B       | 3220.35    | 2869.17          | 1.122×  | 1.061/1.184     |
| arm/bitmap    | count | all            | B       | 10902.85   | 10322.69         | 1.056×  | 1.032/1.081     |
| arm/bitmap    | count | multi_skewed   | B       | 2293.27    | 2183.98          | 1.050×  | 1.041/1.059     |
| arm/bitmap    | count | multi_wide     | B       | 3306.29    | 3169.04          | 1.043×  | 1.027/1.060     |
| arm/bitmap    | count | multi_ordered  | B       | 1679.92    | 1692.04          | 0.993×  | 0.986/1.000     |
| arm/bitmap    | count | multi_boundary | B       | 403.02     | 408.46           | 0.987×  | 0.995/0.978     |
| arm/bitmap    | count | multi_balanced | C       | 3220.35    | 2894.52          | 1.113×  | 1.061/1.163     |
| arm/bitmap    | count | all            | C       | 10902.85   | 10300.02         | 1.059×  | 1.031/1.086     |
| arm/bitmap    | count | multi_skewed   | C       | 2293.27    | 2203.98          | 1.041×  | 1.024/1.058     |
| arm/bitmap    | count | multi_wide     | C       | 3306.29    | 3154.31          | 1.048×  | 1.032/1.064     |
| arm/bitmap    | count | multi_ordered  | C       | 1679.92    | 1647.48          | 1.020×  | 0.991/1.048     |
| arm/bitmap    | count | multi_boundary | C       | 403.02     | 399.73           | 1.008×  | 1.017/0.999     |
| arm/rgb       | 10    | multi_balanced | B       | 42365.87   | 41612.38         | 1.018×  | 1.024/1.012     |
| arm/rgb       | 10    | all            | B       | 142953.38  | 139700.94        | 1.023×  | 1.032/1.015     |
| arm/rgb       | 10    | multi_skewed   | B       | 13580.58   | 13624.08         | 0.997×  | 1.068/0.931     |
| arm/rgb       | 10    | multi_wide     | B       | 42264.65   | 41059.21         | 1.029×  | 1.029/1.030     |
| arm/rgb       | 10    | multi_ordered  | B       | 16490.63   | 16412.23         | 1.005×  | 1.009/1.000     |
| arm/rgb       | 10    | multi_boundary | B       | 28251.65   | 26993.04         | 1.047×  | 1.045/1.048     |
| arm/rgb       | 10    | multi_balanced | C       | 42365.87   | 41676.02         | 1.017×  | 1.026/1.008     |
| arm/rgb       | 10    | all            | C       | 142953.38  | 139450.23        | 1.025×  | 1.028/1.022     |
| arm/rgb       | 10    | multi_skewed   | C       | 13580.58   | 13097.85         | 1.037×  | 1.064/1.009     |
| arm/rgb       | 10    | multi_wide     | C       | 42264.65   | 41212.13         | 1.026×  | 1.023/1.028     |
| arm/rgb       | 10    | multi_ordered  | C       | 16490.63   | 16487.67         | 1.000×  | 0.997/1.003     |
| arm/rgb       | 10    | multi_boundary | C       | 28251.65   | 26976.56         | 1.047×  | 1.040/1.055     |
| arm/rgb       | 100   | multi_balanced | B       | 42408.81   | 42019.25         | 1.009×  | 1.005/1.014     |
| arm/rgb       | 100   | all            | B       | 142106.67  | 140025.15        | 1.015×  | 1.013/1.017     |
| arm/rgb       | 100   | multi_skewed   | B       | 13337.25   | 13097.60         | 1.018×  | 1.007/1.030     |
| arm/rgb       | 100   | multi_wide     | B       | 41551.92   | 41189.04         | 1.009×  | 1.015/1.002     |
| arm/rgb       | 100   | multi_ordered  | B       | 16801.60   | 16469.98         | 1.020×  | 1.026/1.014     |
| arm/rgb       | 100   | multi_boundary | B       | 28007.08   | 27249.27         | 1.028×  | 1.017/1.039     |
| arm/rgb       | 100   | multi_balanced | C       | 42408.81   | 42050.48         | 1.009×  | 1.016/1.001     |
| arm/rgb       | 100   | all            | C       | 142106.67  | 140595.85        | 1.011×  | 1.010/1.011     |
| arm/rgb       | 100   | multi_skewed   | C       | 13337.25   | 13226.31         | 1.008×  | 1.001/1.016     |
| arm/rgb       | 100   | multi_wide     | C       | 41551.92   | 41356.00         | 1.005×  | 1.007/1.003     |
| arm/rgb       | 100   | multi_ordered  | C       | 16801.60   | 16452.77         | 1.021×  | 1.025/1.017     |
| arm/rgb       | 100   | multi_boundary | C       | 28007.08   | 27510.29         | 1.018×  | 1.001/1.035     |
| arm/rgb       | count | multi_balanced | B       | 2818.21    | 2673.46          | 1.054×  | 1.110/0.998     |
| arm/rgb       | count | all            | B       | 10146.65   | 9823.23          | 1.033×  | 1.068/0.998     |
| arm/rgb       | count | multi_skewed   | B       | 2402.08    | 2256.33          | 1.065×  | 1.120/1.009     |
| arm/rgb       | count | multi_wide     | B       | 3084.17    | 3052.96          | 1.010×  | 1.034/0.986     |
| arm/rgb       | count | multi_ordered  | B       | 1411.60    | 1415.15          | 0.997×  | 0.995/1.000     |
| arm/rgb       | count | multi_boundary | B       | 430.58     | 425.33           | 1.012×  | 1.009/1.015     |
| arm/rgb       | count | multi_balanced | C       | 2818.21    | 2721.23          | 1.036×  | 1.109/0.965     |
| arm/rgb       | count | all            | C       | 10146.65   | 9972.35          | 1.017×  | 1.073/0.964     |
| arm/rgb       | count | multi_skewed   | C       | 2402.08    | 2287.92          | 1.050×  | 1.127/0.975     |
| arm/rgb       | count | multi_wide     | C       | 3084.17    | 3108.48          | 0.992×  | 1.042/0.945     |
| arm/rgb       | count | multi_ordered  | C       | 1411.60    | 1421.48          | 0.993×  | 1.007/0.979     |
| arm/rgb       | count | multi_boundary | C       | 430.58     | 433.25           | 0.994×  | 0.995/0.993     |
| x86/default   | 10    | multi_balanced | B       | 712414.45  | 696513.07        | 1.023×  | 1.024/1.022     |
| x86/default   | 10    | all            | B       | 2411293.88 | 2374337.69       | 1.016×  | 1.019/1.013     |
| x86/default   | 10    | multi_skewed   | B       | 245736.11  | 240686.97        | 1.021×  | 1.020/1.022     |
| x86/default   | 10    | multi_wide     | B       | 704392.91  | 697949.03        | 1.009×  | 1.014/1.005     |
| x86/default   | 10    | multi_ordered  | B       | 276549.89  | 269170.78        | 1.027×  | 1.026/1.028     |
| x86/default   | 10    | multi_boundary | B       | 472200.52  | 470017.85        | 1.005×  | 1.013/0.996     |
| x86/default   | 10    | multi_balanced | C       | 712414.45  | 648458.80        | 1.099×  | 1.100/1.097     |
| x86/default   | 10    | all            | C       | 2411293.88 | 2207621.62       | 1.092×  | 1.094/1.091     |
| x86/default   | 10    | multi_skewed   | C       | 245736.11  | 235335.04        | 1.044×  | 1.044/1.044     |
| x86/default   | 10    | multi_wide     | C       | 704392.91  | 649023.64        | 1.085×  | 1.086/1.085     |
| x86/default   | 10    | multi_ordered  | C       | 276549.89  | 252233.17        | 1.096×  | 1.096/1.097     |
| x86/default   | 10    | multi_boundary | C       | 472200.52  | 422570.96        | 1.117×  | 1.123/1.111     |
| x86/default   | 100   | multi_balanced | B       | 714559.57  | 697616.91        | 1.024×  | 1.025/1.024     |
| x86/default   | 100   | all            | B       | 2406276.46 | 2371958.92       | 1.014×  | 1.017/1.011     |
| x86/default   | 100   | multi_skewed   | B       | 245573.44  | 240100.93        | 1.023×  | 1.023/1.022     |
| x86/default   | 100   | multi_wide     | B       | 699635.57  | 692646.46        | 1.010×  | 1.017/1.003     |
| x86/default   | 100   | multi_ordered  | B       | 277212.16  | 270237.23        | 1.026×  | 1.030/1.022     |
| x86/default   | 100   | multi_boundary | B       | 469295.72  | 471357.39        | 0.996×  | 0.997/0.995     |
| x86/default   | 100   | multi_balanced | C       | 714559.57  | 649885.10        | 1.100×  | 1.100/1.099     |
| x86/default   | 100   | all            | C       | 2406276.46 | 2210501.79       | 1.089×  | 1.092/1.085     |
| x86/default   | 100   | multi_skewed   | C       | 245573.44  | 234924.25        | 1.045×  | 1.043/1.048     |
| x86/default   | 100   | multi_wide     | C       | 699635.57  | 650265.47        | 1.076×  | 1.088/1.064     |
| x86/default   | 100   | multi_ordered  | C       | 277212.16  | 252880.91        | 1.096×  | 1.098/1.094     |
| x86/default   | 100   | multi_boundary | C       | 469295.72  | 422546.06        | 1.111×  | 1.110/1.111     |
| x86/default   | count | multi_balanced | B       | 54431.62   | 54254.63         | 1.003×  | 0.997/1.010     |
| x86/default   | count | all            | B       | 202241.18  | 201982.15        | 1.001×  | 0.999/1.003     |
| x86/default   | count | multi_skewed   | B       | 41131.26   | 41177.70         | 0.999×  | 0.989/1.009     |
| x86/default   | count | multi_wide     | B       | 66018.60   | 65735.20         | 1.004×  | 1.005/1.003     |
| x86/default   | count | multi_ordered  | B       | 30926.06   | 30991.84         | 0.998×  | 1.007/0.989     |
| x86/default   | count | multi_boundary | B       | 9733.65    | 9822.78          | 0.991×  | 0.988/0.994     |
| x86/default   | count | multi_balanced | C       | 54431.62   | 54370.51         | 1.001×  | 0.993/1.009     |
| x86/default   | count | all            | C       | 202241.18  | 201098.30        | 1.006×  | 0.998/1.014     |
| x86/default   | count | multi_skewed   | C       | 41131.26   | 40394.23         | 1.018×  | 1.004/1.033     |
| x86/default   | count | multi_wide     | C       | 66018.60   | 65277.51         | 1.011×  | 1.006/1.016     |
| x86/default   | count | multi_ordered  | C       | 30926.06   | 30967.28         | 0.999×  | 0.994/1.003     |
| x86/default   | count | multi_boundary | C       | 9733.65    | 10088.77         | 0.965×  | 0.947/0.982     |
| x86/rgb-pairs | 10    | multi_balanced | B       | 710530.00  | 691601.71        | 1.027×  | 1.028/1.027     |
| x86/rgb-pairs | 10    | all            | B       | 2386936.33 | 2286275.97       | 1.044×  | 1.043/1.045     |
| x86/rgb-pairs | 10    | multi_skewed   | B       | 227287.09  | 223539.58        | 1.017×  | 1.022/1.012     |
| x86/rgb-pairs | 10    | multi_wide     | B       | 696835.95  | 679060.57        | 1.026×  | 1.032/1.020     |
| x86/rgb-pairs | 10    | multi_ordered  | B       | 284152.74  | 266430.02        | 1.067×  | 1.031/1.103     |
| x86/rgb-pairs | 10    | multi_boundary | B       | 468130.56  | 425644.08        | 1.100×  | 1.102/1.097     |
| x86/rgb-pairs | 10    | multi_balanced | C       | 710530.00  | 643469.04        | 1.104×  | 1.099/1.110     |
| x86/rgb-pairs | 10    | all            | C       | 2386936.33 | 2148892.42       | 1.111×  | 1.101/1.121     |
| x86/rgb-pairs | 10    | multi_skewed   | C       | 227287.09  | 212485.33        | 1.070×  | 1.071/1.068     |
| x86/rgb-pairs | 10    | multi_wide     | C       | 696835.95  | 637264.11        | 1.093×  | 1.086/1.101     |
| x86/rgb-pairs | 10    | multi_ordered  | C       | 284152.74  | 246094.64        | 1.155×  | 1.115/1.194     |
| x86/rgb-pairs | 10    | multi_boundary | C       | 468130.56  | 409579.30        | 1.143×  | 1.134/1.152     |
| x86/rgb-pairs | 100   | multi_balanced | B       | 709442.27  | 690861.40        | 1.027×  | 1.027/1.027     |
| x86/rgb-pairs | 100   | all            | B       | 2362910.72 | 2282604.30       | 1.035×  | 1.034/1.036     |
| x86/rgb-pairs | 100   | multi_skewed   | B       | 227889.03  | 224680.21        | 1.014×  | 1.013/1.016     |
| x86/rgb-pairs | 100   | multi_wide     | B       | 685169.51  | 675617.53        | 1.014×  | 1.015/1.014     |
| x86/rgb-pairs | 100   | multi_ordered  | B       | 276003.87  | 267461.86        | 1.032×  | 1.030/1.034     |
| x86/rgb-pairs | 100   | multi_boundary | B       | 464406.04  | 423983.31        | 1.095×  | 1.092/1.099     |
| x86/rgb-pairs | 100   | multi_balanced | C       | 709442.27  | 639926.97        | 1.109×  | 1.108/1.109     |
| x86/rgb-pairs | 100   | all            | C       | 2362910.72 | 2135835.08       | 1.106×  | 1.106/1.107     |
| x86/rgb-pairs | 100   | multi_skewed   | C       | 227889.03  | 213373.91        | 1.068×  | 1.066/1.070     |
| x86/rgb-pairs | 100   | multi_wide     | C       | 685169.51  | 629380.53        | 1.089×  | 1.088/1.089     |
| x86/rgb-pairs | 100   | multi_ordered  | C       | 276003.87  | 248636.32        | 1.110×  | 1.110/1.110     |
| x86/rgb-pairs | 100   | multi_boundary | C       | 464406.04  | 404517.35        | 1.148×  | 1.148/1.148     |
| x86/rgb-pairs | count | multi_balanced | B       | 52490.24   | 52062.53         | 1.008×  | 0.999/1.018     |
| x86/rgb-pairs | count | all            | B       | 201370.68  | 198531.76        | 1.014×  | 1.004/1.025     |
| x86/rgb-pairs | count | multi_skewed   | B       | 47524.40   | 46215.32         | 1.028×  | 1.016/1.041     |
| x86/rgb-pairs | count | multi_wide     | B       | 63269.95   | 62810.08         | 1.007×  | 1.001/1.014     |
| x86/rgb-pairs | count | multi_ordered  | B       | 26747.66   | 26067.19         | 1.026×  | 1.007/1.045     |
| x86/rgb-pairs | count | multi_boundary | B       | 11338.43   | 11376.64         | 0.997×  | 0.984/1.009     |
| x86/rgb-pairs | count | multi_balanced | C       | 52490.24   | 52288.73         | 1.004×  | 1.005/1.003     |
| x86/rgb-pairs | count | all            | C       | 201370.68  | 199639.22        | 1.009×  | 1.007/1.010     |
| x86/rgb-pairs | count | multi_skewed   | C       | 47524.40   | 46514.51         | 1.022×  | 1.017/1.026     |
| x86/rgb-pairs | count | multi_wide     | C       | 63269.95   | 62855.22         | 1.007×  | 1.007/1.006     |
| x86/rgb-pairs | count | multi_ordered  | C       | 26747.66   | 26654.01         | 1.004×  | 0.991/1.016     |
| x86/rgb-pairs | count | multi_boundary | C       | 11338.43   | 11326.75         | 1.001×  | 1.012/0.990     |

Process resources include startup, warmups, all queries and output. They
are not per-kernel heap measurements. RSS includes mapped index pages.
Raw caller-thread CPU clocks exclude search-pool workers and are not used
as total query CPU. The process user+system counters below include them.

## Whole-process resources (mean of paired phases)

| Suite      | Host/index    | Limit | Variant | A CPU s | Candidate CPU s | A RSS MiB | Candidate RSS MiB |
| ---------- | ------------- | ----- | ------- | ------- | --------------- | --------- | ----------------- |
| integrated | arm/bitmap    | 10    | B       | 2.665   | 2.655           | 222.15    | 221.89            |
| integrated | arm/bitmap    | 100   | B       | 4.955   | 5.015           | 226.05    | 223.95            |
| integrated | arm/bitmap    | count | B       | 10.165  | 10.405          | 221.70    | 221.44            |
| integrated | arm/rgb       | 10    | B       | 2.235   | 2.245           | 224.40    | 225.23            |
| integrated | arm/rgb       | 100   | B       | 4.100   | 4.145           | 231.49    | 230.51            |
| integrated | arm/rgb       | count | B       | 9.685   | 9.815           | 227.63    | 228.16            |
| integrated | x86/default   | 10    | B       | 27.815  | 27.855          | 2297.36   | 2297.41           |
| integrated | x86/default   | 100   | B       | 51.740  | 51.890          | 2326.87   | 2326.74           |
| integrated | x86/default   | count | B       | 172.505 | 174.300         | 2295.84   | 2295.81           |
| integrated | x86/rgb-pairs | 10    | B       | 19.980  | 19.905          | 2231.25   | 2231.18           |
| integrated | x86/rgb-pairs | 100   | B       | 26.945  | 26.920          | 2249.33   | 2249.58           |
| integrated | x86/rgb-pairs | count | B       | 44.010  | 44.065          | 2175.75   | 2175.48           |
| multi      | arm/bitmap    | 10    | B       | 3.770   | 3.750           | 113.02    | 112.62            |
| multi      | arm/bitmap    | 10    | C       | 3.770   | 3.750           | 113.02    | 112.85            |
| multi      | arm/bitmap    | 100   | B       | 3.815   | 3.795           | 112.55    | 113.02            |
| multi      | arm/bitmap    | 100   | C       | 3.815   | 3.820           | 112.55    | 113.31            |
| multi      | arm/bitmap    | count | B       | 0.345   | 0.315           | 111.42    | 110.65            |
| multi      | arm/bitmap    | count | C       | 0.345   | 0.315           | 111.42    | 110.59            |
| multi      | arm/rgb       | 10    | B       | 3.780   | 3.700           | 122.16    | 122.05            |
| multi      | arm/rgb       | 10    | C       | 3.780   | 3.700           | 122.16    | 121.80            |
| multi      | arm/rgb       | 100   | B       | 3.755   | 3.720           | 121.76    | 122.29            |
| multi      | arm/rgb       | 100   | C       | 3.755   | 3.735           | 121.76    | 122.16            |
| multi      | arm/rgb       | count | B       | 0.320   | 0.305           | 120.18    | 120.00            |
| multi      | arm/rgb       | count | C       | 0.320   | 0.310           | 120.18    | 119.99            |
| multi      | x86/default   | 10    | B       | 63.150  | 62.430          | 1268.73   | 1268.88           |
| multi      | x86/default   | 10    | C       | 63.150  | 58.150          | 1268.73   | 1268.67           |
| multi      | x86/default   | 100   | B       | 63.070  | 62.390          | 1269.14   | 1269.12           |
| multi      | x86/default   | 100   | C       | 63.070  | 58.185          | 1269.14   | 1268.83           |
| multi      | x86/default   | count | B       | 5.950   | 5.925           | 1252.32   | 1252.39           |
| multi      | x86/default   | count | C       | 5.950   | 5.915           | 1252.32   | 1252.19           |
| multi      | x86/rgb-pairs | 10    | B       | 62.735  | 60.305          | 1363.72   | 1363.96           |
| multi      | x86/rgb-pairs | 10    | C       | 62.735  | 56.650          | 1363.72   | 1363.60           |
| multi      | x86/rgb-pairs | 100   | B       | 62.260  | 60.220          | 1364.17   | 1364.15           |
| multi      | x86/rgb-pairs | 100   | C       | 62.260  | 56.440          | 1364.17   | 1363.71           |
| multi      | x86/rgb-pairs | count | B       | 6.110   | 6.010           | 1347.23   | 1347.40           |
| multi      | x86/rgb-pairs | count | C       | 6.110   | 6.070           | 1347.23   | 1347.17           |
| selected   | x86/default   | 10    | D       | 48.125  | 46.155          | 2306.11   | 2306.16           |
| selected   | x86/default   | 100   | D       | 62.650  | 60.830          | 2335.29   | 2335.19           |
| selected   | x86/default   | count | D       | 103.390 | 101.335         | 2312.09   | 2312.36           |
| selected   | x86/rgb-pairs | 10    | D       | 42.785  | 40.460          | 2241.36   | 2241.39           |
| selected   | x86/rgb-pairs | 100   | D       | 47.360  | 44.910          | 2258.99   | 2259.00           |
| selected   | x86/rgb-pairs | count | D       | 28.695  | 28.185          | 2192.12   | 2192.37           |
