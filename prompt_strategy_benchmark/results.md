Model: llama3.2   Tickets: 50   Shots: 6

| Strategy | Accuracy | Ambiguous acc. | Parse fails | Avg in tok | Avg out tok | Avg sec | Total min |
|---|---|---|---|---|---|---|---|
| zero_shot | 70.0% | 41.7% | 0 | 76 | 2 | 1.51 | 1.3 |
| few_shot | 78.0% | 75.0% | 0 | 179 | 2 | 1.36 | 1.1 |
| cot | 80.0% | 75.0% | 0 | 100 | 76 | 10.11 | 8.4 |

Confusion matrix: zero_shot (rows = true, columns = predicted)
| true \ pred | billing | bug | feature_request | other | none |
|---|---|---|---|---|---|
| billing | 6 | 3 | 2 | 2 | 0 |
| bug | 0 | 12 | 1 | 0 | 0 |
| feature_request | 0 | 1 | 11 | 0 | 0 |
| other | 1 | 1 | 4 | 6 | 0 |

Confusion matrix: few_shot (rows = true, columns = predicted)
| true \ pred | billing | bug | feature_request | other | none |
|---|---|---|---|---|---|
| billing | 8 | 1 | 2 | 2 | 0 |
| bug | 2 | 9 | 2 | 0 | 0 |
| feature_request | 0 | 0 | 12 | 0 | 0 |
| other | 0 | 0 | 2 | 10 | 0 |

Confusion matrix: cot (rows = true, columns = predicted)
| true \ pred | billing | bug | feature_request | other | none |
|---|---|---|---|---|---|
| billing | 10 | 1 | 1 | 1 | 0 |
| bug | 0 | 13 | 0 | 0 | 0 |
| feature_request | 0 | 0 | 12 | 0 | 0 |
| other | 1 | 0 | 6 | 5 | 0 |
