# Classifier Results Summary

| Label mode | FS | Temporal | Margin | Features | Best | Acc | Macro-F1 |
|---|---:|---:|---:|---:|---|---:|---:|
| 3class | 250 | True | 4.0 | 32 | RF | 0.705 | 0.643 |
| 3class | 250 | LSTM(seq=6) | 4.0 | 8 | LSTM | 0.684 | 0.638 |
| 3class | 250 | LSTM(seq=6),predict-next | 4.0 | 8 | LSTM | 0.684 | 0.632 |
| 3class | 250 | False | 0.0 | 8 | RF | 0.659 | 0.593 |
| 3class | 1259 | True | 4.0 | 32 | RF | 0.731 | 0.675 |
| binary_drop_transition | 250 | True | 4.0 | 32 | RF | 0.896 | 0.854 |
| binary_drop_transition | 250 | LSTM(seq=6) | 4.0 | 8 | LSTM | 0.887 | 0.843 |
| binary_drop_transition | 1259 | True | 4.0 | 32 | RF | 0.910 | 0.876 |
