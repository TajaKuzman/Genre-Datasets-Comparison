# Information on the CORE dataset

When preparing the dataset, we:
* discarded instances with no texts (17)
* discarded duplicates (12)

The dataset has 48,420 texts with 459 different main and sub-category label combinations. Regarding main labels, it has 35 different combinations and 297 different sub-category combinations.

## CORE-main

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

## CORE-sub

CORE-sub is the CORE dataset, annotated with subcategories only. For the experiments, we:
* discarded all texts that are annotated with multiple subcategories (3622)
* discarded all texts that are not annotated with any subcategory (4932)

The dataset contains 39,866 instances, annotated with 47 different labels.

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