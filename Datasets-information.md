# Information on the datasets

Content:
* [CORE](core)
* [FTD](ftd)
* [GINCO](ginco)

## CORE

When preparing the dataset, we:
* discarded instances with no texts (17)
* discarded duplicates (12)

The dataset has 48,420 texts with 459 different main and sub-category label combinations. Regarding main labels, it has 35 different combinations and 297 different sub-category combinations.

### CORE-main

CORE-main is the CORE dataset, annotated with main categories only (9 categories). For these experiments, we discarded all text that are annotated with more than 1 main category (5686 texts).

| main labels                  | count | percentage |
|-----------------------------------------|-------|------------|
| Narrative                               | 17659 | 41.3231    |
| Informational Description/Explanation | 9314  | 21.7953    |
| Opinion                                 | 7862  | 18.3975    |
| Interactive Discussion                | 3272  | 7.65667    |
| How-To/Instructional                    | 1493  | 3.49371    |
| Informational Persuasion              | 1330  | 3.11228    |
| Lyrical                                 | 639   | 1.4953     |
| Spoken                                  | 583   | 1.36425    |
| Other                                   | 582   | 1.36191    |


|       |   text_length |
|:------|--------------:|
| text count |      42734    |
| mean  |       1236.27 |
| std   |       3167.36 |
| min   |         52    |
| 25%   |        333    |
| 50%   |        630.5  |
| 75%   |       1152    |
| max   |     118278    |

Total number of texts: 42734, distributed in train split (25640 texts), test and dev split (8547 each), stratified according to the label.

### CORE-sub

CORE-sub is the CORE dataset, annotated with subcategories only. For the experiments, we:
* discarded all texts that are annotated with multiple subcategories (3622)
* discarded all texts that are not annotated with any subcategory (4932)
* discarded instances belonging to categories with less than 10 instances (17)

| subcategories                     | count | percentage |
|-----------------------------------|-------|------------|
| News Report/Blog                  | 10503 | 26.3458    |
| Opinion Blog                      | 4135  | 10.3722    |
| Description of a   Thing          | 3508  | 8.79948    |
| Sports Report                     | 2820  | 7.0737     |
| Personal Blog                     | 2769  | 6.94577    |
| Discussion Forum                  | 1950  | 4.89139    |
| Reviews                           | 1803  | 4.52265    |
| Information Blog                  | 1526  | 3.82782    |
| How-to                            | 1318  | 3.30608    |
| Description with   Intent to Sell | 1093  | 2.74168    |
| Question/Answer   Forum           | 1052  | 2.63884    |
| Advice                            | 933   | 2.34034    |
| Research Article                  | 822   | 2.06191    |
| Description of a   Person         | 764   | 1.91642    |
| Religious   Blogs/Sermons         | 697   | 1.74836    |
| Song Lyrics                       | 543   | 1.36206    |
| Encyclopedia   Article            | 522   | 1.30939    |
| Interview                         | 468   | 1.17393    |
| Historical Article                | 422   | 1.05855    |
| Travel Blog                       | 283   | 0.709878   |
| Short Story                       | 283   | 0.709878   |
| FAQ about   Information           | 252   | 0.632118   |
| Legal terms                       | 186   | 0.466563   |
| Recipe                            | 171   | 0.428937   |
| Other Information                 | 137   | 0.343651   |
| Persuasive Article   or Essay     | 120   | 0.301008   |
| Course Materials                  | 118   | 0.295992   |
| Poem                              | 73    | 0.183113   |
| Magazine Article                  | 72    | 0.180605   |
| Editorial                         | 66    | 0.165555   |
| Transcript of   Video/Audio       | 62    | 0.155521   |
| Reader/Viewer   Responses         | 50    | 0.12542    |
| FAQ about How-to                  | 47    | 0.117895   |
| Letter to Editor                  | 43    | 0.107861   |
| Formal Speech                     | 43    | 0.107861   |
| Technical Report                  | 41    | 0.102845   |
| Technical Support                 | 39    | 0.0978277  |
| TV/Movie Script                   | 22    | 0.0551849  |
| Other Opinion                     | 18    | 0.0451513  |
| Other Forum                       | 18    | 0.0451513  |
| Other Spoken                      | 17    | 0.0426429  |
| Prayer                            | 16    | 0.0401345  |
| Advertisement                     | 13    | 0.0326092  |
| Other   Informational Persuasion  | 11    | 0.0275924  |
| Other Narrative*                   | 9     | 0.0225756  |
| Other Lyrical*                     | 6     | 0.0150504  |
| Other How-to*                      | 2     | 0.00501681 |

Categories (marked with a *) with less than 10 instances were discarded.

The final dataset contains 39,849 texts, annotated with 44 labels. It was split into train (23,909), test and dev split (7970 each), stratified based on the labels.

## FTD

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

## GINCO

We will use paragraphs of texts that are marked as "keep". As labels, we used the primary_level_1 labels (the original set without downcasting).

Text length:

|       |   text_length |
|:------|--------------:|
| count |      1002     |
| mean  |       362.159 |
| std   |       483.747 |
| min   |        12     |
| 25%   |        98     |
| 50%   |       208     |
| 75%   |       418.75  |
| max   |      4364     |

Like in experiments with CORE, we discarded instances of categories with less than 10 instances (marked with a * in the table below).

|                            |   Count |   Percentage |
|:---------------------------|--------:|-------------:|
| Information/Explanation    |     130 |  0.129741    |
| News/Reporting             |     115 |  0.11477     |
| Promotion of a Product     |     115 |  0.11477     |
| Opinion/Argumentation      |     114 |  0.113772    |
| List of Summaries/Excerpts |     106 |  0.105788    |
| Opinionated News           |      89 |  0.0888224   |
| Forum                      |      52 |  0.0518962   |
| Instruction                |      38 |  0.0379242   |
| Other                      |      34 |  0.0339321   |
| Invitation                 |      32 |  0.0319361   |
| Promotion of Services      |      32 |  0.0319361   |
| Promotion                  |      30 |  0.0299401   |
| Legal/Regulation           |      17 |  0.0169661   |
| Announcement               |      17 |  0.0169661   |
| Review                     |      17 |  0.0169661   |
| Correspondence             |      16 |  0.0159681   |
| Call                       |      11 |  0.010978    |
| Research Article*           |       9 |  0.00898204  |
| Interview*                  |       8 |  0.00798403  |
| Recipe*                     |       6 |  0.00598802  |
| Prose*                      |       6 |  0.00598802  |
| Lyrical*                    |       4 |  0.00399202  |
| FAQ*                        |       3 |  0.00299401  |
| Script/Drama*               |       1 |  0.000998004 |

The final dataset has 965 texts with 17 different labels. A stratified split was performed in a 60:20:20 manner into a train (579), dev and test spli (each 193). The splits are saved as *data/GINCO-full-set-{train, test, dev}.csv*

The spreadsheet with information on the splits is saved as *final_data/GINCO-MT-GINCO-keeptext-split-file-with-all-information.csv*.