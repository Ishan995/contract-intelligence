# Improvement Experiment Log

Started: 2026-11-01T17:51:15.965958

## Iteration 0

- **Change:** baseline
- **Reason:** initial config
- **Proposer:** —
- **Accepted:** ✅ YES
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'wrong_doc': 3, 'right_doc_wrong_section': 109, 'partial_span': 7, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0069
  - precision@16: 0.0070
  - precision@2: 0.0035
  - precision@4: 0.0075
  - precision@8: 0.0095
  - real_abstain_rate: 0.0000
  - recall@1: 0.0047
  - recall@16: 0.1405
  - recall@2: 0.0047
  - recall@4: 0.0330
  - recall@8: 0.0937

---

## Iteration 1

- **Change:** clause-aware chunking
- **Reason:** top failure: right_doc_wrong_section ({'wrong_doc': 3, 'right_doc_wrong_section': 109, 'partial_span': 7, 'missed_abstain': 28})
- **Proposer:** right_doc_wrong_section → _variants_clause_chunking
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'wrong_doc': 3, 'right_doc_wrong_section': 114, 'partial_span': 8, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0091
  - precision@16: 0.0063
  - precision@2: 0.0051
  - precision@4: 0.0079
  - precision@8: 0.0067
  - real_abstain_rate: 0.0000
  - recall@1: 0.0059
  - recall@16: 0.1134
  - recall@2: 0.0059
  - recall@4: 0.0247
  - recall@8: 0.0622

---

## Iteration 2

- **Change:** chunk_size=256, overlap=64
- **Reason:** top failure: right_doc_wrong_section ({'wrong_doc': 3, 'right_doc_wrong_section': 109, 'partial_span': 7, 'missed_abstain': 28})
- **Proposer:** right_doc_wrong_section → _variants_chunk_size
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'wrong_doc': 5, 'right_doc_wrong_section': 114, 'partial_span': 7, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0000
  - precision@16: 0.0072
  - precision@2: 0.0024
  - precision@4: 0.0068
  - precision@8: 0.0084
  - real_abstain_rate: 0.0000
  - recall@1: 0.0000
  - recall@16: 0.0620
  - recall@2: 0.0020
  - recall@4: 0.0124
  - recall@8: 0.0373

---

## Iteration 3

- **Change:** chunk_size=384, overlap=96
- **Reason:** top failure: right_doc_wrong_section ({'wrong_doc': 3, 'right_doc_wrong_section': 109, 'partial_span': 7, 'missed_abstain': 28})
- **Proposer:** right_doc_wrong_section → _variants_chunk_size
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'right_doc_wrong_section': 113, 'wrong_doc': 2, 'partial_span': 6, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0075
  - precision@16: 0.0077
  - precision@2: 0.0038
  - precision@4: 0.0057
  - precision@8: 0.0108
  - real_abstain_rate: 0.0000
  - recall@1: 0.0039
  - recall@16: 0.1211
  - recall@2: 0.0039
  - recall@4: 0.0142
  - recall@8: 0.0917

---

## Iteration 4

- **Change:** chunk_size=768, overlap=192
- **Reason:** top failure: right_doc_wrong_section ({'wrong_doc': 3, 'right_doc_wrong_section': 109, 'partial_span': 7, 'missed_abstain': 28})
- **Proposer:** right_doc_wrong_section → _variants_chunk_size
- **Accepted:** ✅ YES
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'wrong_doc': 3, 'right_doc_wrong_section': 99, 'partial_span': 9, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0086
  - precision@16: 0.0079
  - precision@2: 0.0058
  - precision@4: 0.0089
  - precision@8: 0.0103
  - real_abstain_rate: 0.0000
  - recall@1: 0.0096
  - recall@16: 0.2155
  - recall@2: 0.0178
  - recall@4: 0.0761
  - recall@8: 0.1441

---

## Iteration 5

- **Change:** chunk_size=1024, overlap=256
- **Reason:** top failure: right_doc_wrong_section ({'wrong_doc': 3, 'right_doc_wrong_section': 99, 'partial_span': 9, 'missed_abstain': 28})
- **Proposer:** right_doc_wrong_section → _variants_chunk_size
- **Accepted:** ✅ YES
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'right_doc_wrong_section': 96, 'wrong_doc': 2, 'partial_span': 6, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0076
  - precision@16: 0.0085
  - precision@2: 0.0086
  - precision@4: 0.0113
  - precision@8: 0.0110
  - real_abstain_rate: 0.0000
  - recall@1: 0.0130
  - recall@16: 0.2949
  - recall@2: 0.0386
  - recall@4: 0.0998
  - recall@8: 0.1977

---

## Iteration 6

- **Change:** bm25 k1=0.5, b=0.3
- **Reason:** top failure: right_doc_wrong_section ({'right_doc_wrong_section': 96, 'wrong_doc': 2, 'partial_span': 6, 'missed_abstain': 28})
- **Proposer:** right_doc_wrong_section → _variants_bm25_params
- **Accepted:** ✅ YES
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'right_doc_wrong_section': 96, 'wrong_doc': 1, 'partial_span': 5, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0041
  - precision@16: 0.0095
  - precision@2: 0.0085
  - precision@4: 0.0109
  - precision@8: 0.0118
  - real_abstain_rate: 0.0000
  - recall@1: 0.0104
  - recall@16: 0.3328
  - recall@2: 0.0389
  - recall@4: 0.0985
  - recall@8: 0.2042

---

## Iteration 7

- **Change:** bm25 k1=0.5, b=0.5
- **Reason:** top failure: right_doc_wrong_section ({'right_doc_wrong_section': 96, 'wrong_doc': 1, 'partial_span': 5, 'missed_abstain': 28})
- **Proposer:** right_doc_wrong_section → _variants_bm25_params
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'right_doc_wrong_section': 97, 'wrong_doc': 1, 'partial_span': 5, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0041
  - precision@16: 0.0095
  - precision@2: 0.0085
  - precision@4: 0.0103
  - precision@8: 0.0116
  - real_abstain_rate: 0.0000
  - recall@1: 0.0104
  - recall@16: 0.3328
  - recall@2: 0.0389
  - recall@4: 0.0961
  - recall@8: 0.2002

---

## Iteration 8

- **Change:** bm25 k1=0.5, b=0.75
- **Reason:** top failure: right_doc_wrong_section ({'right_doc_wrong_section': 96, 'wrong_doc': 1, 'partial_span': 5, 'missed_abstain': 28})
- **Proposer:** right_doc_wrong_section → _variants_bm25_params
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.0000
  - failures: {'right_doc_wrong_section': 96, 'wrong_doc': 1, 'partial_span': 6, 'missed_abstain': 28}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0041
  - precision@16: 0.0092
  - precision@2: 0.0085
  - precision@4: 0.0103
  - precision@8: 0.0120
  - real_abstain_rate: 0.0000
  - recall@1: 0.0104
  - recall@16: 0.3294
  - recall@2: 0.0389
  - recall@4: 0.0961
  - recall@8: 0.2023

---

## Summary

| Metric | Baseline | Final | Delta |
|--------|----------|-------|-------|
| recall@8 | 0.0937 | 0.2042 | +0.1106 |
| precision@1 | 0.0069 | 0.0041 | -0.0029 |
| precision@8 | 0.0095 | 0.0118 | +0.0022 |
| recall@1 | 0.0047 | 0.0104 | +0.0056 |
| abstain_accuracy | 0.0000 | 0.0000 | +0.0000 |
| real_abstain_rate | 0.0000 | 0.0000 | +0.0000 |

Final config saved to `configs/final.yaml`

# Continued Optimization (2026-11-01T18:05:41.131422)


# Continued Optimization (2026-11-01T18:09:11.580898)

## Iteration 9

- **Change:** threshold sweep (bm25_abstain_threshold=32.19, obj=0.9774, unans_abs=1.0000, real_abs=0.0226)
- **Reason:** failures: {'right_doc_wrong_section': 96, 'wrong_doc': 1, 'partial_span': 5, 'missed_abstain': 28}
- **Proposer:** abstention_failure -> _sweep_threshold
- **Accepted:** ✅ YES
- **Metrics:**
  - abstain_accuracy: 1.0000
  - failures: {'right_doc_wrong_section': 94, 'wrong_doc': 1, 'wrong_abstain': 3, 'partial_span': 4}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0266
  - precision@16: 0.0317
  - precision@2: 0.0310
  - precision@4: 0.0334
  - precision@8: 0.0343
  - real_abstain_rate: 0.0226
  - recall@1: 0.0104
  - recall@16: 0.3281
  - recall@2: 0.0389
  - recall@4: 0.0983
  - recall@8: 0.2040

---

## Iteration 10

- **Change:** bm25 k1=0.5, b=1.0
- **Reason:** failures: {'right_doc_wrong_section': 94, 'wrong_doc': 1, 'wrong_abstain': 3, 'partial_span': 4}
- **Proposer:** right_doc_wrong_section -> _variants_bm25_params
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 1.0000
  - failures: {'right_doc_wrong_section': 95, 'wrong_doc': 1, 'wrong_abstain': 3, 'partial_span': 4}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0266
  - precision@16: 0.0314
  - precision@2: 0.0313
  - precision@4: 0.0335
  - precision@8: 0.0341
  - real_abstain_rate: 0.0226
  - recall@1: 0.0104
  - recall@16: 0.3130
  - recall@2: 0.0429
  - recall@4: 0.0985
  - recall@8: 0.2005

---

## Iteration 11

- **Change:** bm25 k1=1.0, b=0.3
- **Reason:** failures: {'right_doc_wrong_section': 94, 'wrong_doc': 1, 'wrong_abstain': 3, 'partial_span': 4}
- **Proposer:** right_doc_wrong_section -> _variants_bm25_params
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.9643
  - failures: {'right_doc_wrong_section': 96, 'wrong_doc': 1, 'wrong_abstain': 2, 'partial_span': 4, 'missed_abstain': 1}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0191
  - precision@16: 0.0241
  - precision@2: 0.0218
  - precision@4: 0.0271
  - precision@8: 0.0264
  - real_abstain_rate: 0.0150
  - recall@1: 0.0104
  - recall@16: 0.3135
  - recall@2: 0.0331
  - recall@4: 0.1039
  - recall@8: 0.1979

---

## Summary

| Metric | Baseline | Final | Delta |
|--------|----------|-------|-------|
| recall@8 | 0.0937 | 0.2040 | +0.1104 |
| precision@1 | 0.0069 | 0.0266 | +0.0197 |
| precision@8 | 0.0095 | 0.0343 | +0.0248 |
| recall@1 | 0.0047 | 0.0104 | +0.0056 |
| abstain_accuracy | 0.0000 | 1.0000 | +1.0000 |
| real_abstain_rate | 0.0000 | 0.0226 | +0.0226 |

Final config saved to `configs/final.yaml`

# Continued Optimization (2026-11-01T18:17:36.804362)

## Iteration 12

- **Change:** bm25 k1=1.0, b=0.5
- **Reason:** failures: {'right_doc_wrong_section': 94, 'wrong_doc': 1, 'wrong_abstain': 3, 'partial_span': 4}
- **Proposer:** right_doc_wrong_section -> _variants_bm25_params
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.9643
  - failures: {'right_doc_wrong_section': 97, 'wrong_doc': 1, 'wrong_abstain': 2, 'partial_span': 4, 'missed_abstain': 1}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0191
  - precision@16: 0.0241
  - precision@2: 0.0221
  - precision@4: 0.0259
  - precision@8: 0.0263
  - real_abstain_rate: 0.0150
  - recall@1: 0.0104
  - recall@16: 0.3135
  - recall@2: 0.0360
  - recall@4: 0.0985
  - recall@8: 0.1904

---

## Iteration 13

- **Change:** bm25 k1=1.0, b=0.75
- **Reason:** failures: {'right_doc_wrong_section': 94, 'wrong_doc': 1, 'wrong_abstain': 3, 'partial_span': 4}
- **Proposer:** right_doc_wrong_section -> _variants_bm25_params
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.9643
  - failures: {'right_doc_wrong_section': 97, 'wrong_doc': 1, 'wrong_abstain': 2, 'partial_span': 5, 'missed_abstain': 1}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0191
  - precision@16: 0.0241
  - precision@2: 0.0221
  - precision@4: 0.0257
  - precision@8: 0.0266
  - real_abstain_rate: 0.0150
  - recall@1: 0.0104
  - recall@16: 0.3135
  - recall@2: 0.0360
  - recall@4: 0.0965
  - recall@8: 0.1890

---

## Iteration 14

- **Change:** bm25 k1=1.0, b=1.0
- **Reason:** failures: {'right_doc_wrong_section': 94, 'wrong_doc': 1, 'wrong_abstain': 3, 'partial_span': 4}
- **Proposer:** right_doc_wrong_section -> _variants_bm25_params
- **Accepted:** ❌ NO
- **Metrics:**
  - abstain_accuracy: 0.9643
  - failures: {'wrong_doc': 2, 'right_doc_wrong_section': 96, 'wrong_abstain': 2, 'partial_span': 4, 'missed_abstain': 1}
  - n_train: 133.0000
  - n_unans_train: 28.0000
  - precision@1: 0.0227
  - precision@16: 0.0238
  - precision@2: 0.0239
  - precision@4: 0.0257
  - precision@8: 0.0263
  - real_abstain_rate: 0.0150
  - recall@1: 0.0130
  - recall@16: 0.2969
  - recall@2: 0.0426
  - recall@4: 0.0965
  - recall@8: 0.1950

---

## Summary

| Metric | Baseline | Final | Delta |
|--------|----------|-------|-------|
| recall@8 | 0.0937 | 0.2040 | +0.1104 |
| precision@1 | 0.0069 | 0.0266 | +0.0197 |
| precision@8 | 0.0095 | 0.0343 | +0.0248 |
| recall@1 | 0.0047 | 0.0104 | +0.0056 |
| abstain_accuracy | 0.0000 | 1.0000 | +1.0000 |
| real_abstain_rate | 0.0000 | 0.0226 | +0.0226 |

Final config saved to `configs/final.yaml`
