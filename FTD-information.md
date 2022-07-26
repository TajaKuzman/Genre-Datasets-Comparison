# The FTD dataset

The dataset is available here: https://github.com/ssharoff/genre-keras/blob/master/en.csv

To simplify experiments, we will perform single-label classification. The original dataset allows multiple-label annotation using a scale from 0 to 2 for each label. To simplify experiments, we will regard texts annotated with 2 at a certain category as belonging to this label.

The distribution of the labels:

| Labels         | Count | % (of single labels)|
|--------------------|-------------|--------------|
| **single labels**      | **1547**        |      |
| A1 (argumentative) | 297         | 19,20%       |
| A11 (personal)     | 79          | 5,11%        |
| A12 (promotion)    | 259         | 16,74%       |
| A14 (academic)     | 81          | 5,24%        |
| A16 (information)  | 168         | 10,86%       |
| A17 (review)       | 70          | 4,52%        |
| A22 (non-text)     | 125         | 8,08%        |
| A4 (fiction)       | 94          | 6,08%        |
| A7 (instruction)   | 165         | 10,67%       |
| A8 (news)          | 136         | 8,79%        |
| A9 (legal)         | 73          | 4,72%        |
| **multiple labels**    | **139**         |      |
| **Grand Total**        | **1686**        |              |

Although the FTD article mentions 12 principal categories and 6 optional, it seems that the final FTD dataset is annotated only with 11 categories, out of which one is an "unsuitable" category.

For the experiments, we removed:
* unsuitable texts, marked as "non-text" (125)
* texts, annotated with multiple labels (139)
* duplicated texts (7)

The final number of instances, used for the experiments: 1415.

Dataset that is used for the ML experiments is split into the train-dev-test split according to the label distribution.

|        | train              | test               | dev                |
|:-------|:-------------------|:-------------------|:-------------------|
| count (texts) | 849                | 283                | 283                |


The splits are located in the *data* folder.

Dataset with all texts and labels: *extension-to-the-FTD/data/FTD-dataset-with-all-information.txt*

Text length:

|       |    length |
|:------|----------:|
| mean  |   1445.29 |
| std   |   4987.81 |
| min   |     31    |
| 25%   |    224    |
| 50%   |    495    |
| 75%   |   1147    |
| max   | 146922    |

There are 215 texts that are longer than 2000 words, 71 of them are longer than 5000 words and 10 of them are longer than 20,000 words. The analysis shows that the corpus contains very big parts of literary works (e.g., __id__47-FictBalzacH_Goriot_Ia_EN.txt - 22.3k words) and very long UN documents (e.g., __id__214-un - 35.6k words).