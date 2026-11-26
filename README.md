# Detecting Spoilers in IMDb Reviews

This project trains a TextCNN to predict user-provided spoiler flags from movie and TV review text. It examines input length and class weighting on previously unseen titles, compares the neural model with a TF-IDF logistic regression baseline, and demonstrates automatic spoiler-warning recommendations.

The recorded results favor the full-review linear baseline. Increasing the TextCNN input limit from 128 to 256 tokens improved validation performance, while class weighting produced smaller, inconsistent changes. A manual audit of potential missing user flags has been exported but remains incomplete.

## Dataset and preparation

The [IMDb Spoiler Dataset by Rishabh Misra](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset) supplies review text, spoiler flags and title identifiers. The loader found 573,913 eligible source reviews and skipped none. A uniform reservoir sample of 30,000 reviews was selected. This executed version uses ordinary random sampling, not proportionate stratified sampling.

Cleanup removed five sampled rows, including rows associated with a duplicated text that had conflicting labels. The cleaned sample contains 29,995 reviews: 22,195 non-spoiler reviews (74.0%) and 7,800 spoiler reviews (26.0%). Median review length is 170 whitespace-separated words for non-spoiler reviews and 260 for spoiler reviews. Model tokens use a separate regex word/punctuation tokenizer.

The model input is `review_text`, and the target is `is_spoiler`. Ratings, user IDs, title IDs, review summaries and plot synopses are excluded from model inputs. Movie metadata is used for genre analysis only. User spoiler flags are imperfect reference labels rather than independently verified annotations.

## Data splits

Splitting by `movie_id` keeps all reviews of each sampled title in one partition. Development/test and training/validation-within-development both use 80/20 group splits with seed 42. The approximate overall group allocation is 64/16/20; review counts need not follow those fractions exactly.

| Partition | Reviews | Titles | Spoiler rate |
|---|---:|---:|---:|
| Training | 19,175 | 993 | 25.8% |
| Validation | 4,828 | 249 | 26.1% |
| Test | 5,992 | 311 | 26.6% |

Title and exact cleaned-text overlap checks passed. However, 1,065 known user IDs occur in both training and test. The evaluation therefore concerns unseen titles rather than entirely unseen authors. Exact deduplication covers the selected sample; near-duplicate detection was not performed.

## Models and training

The baseline uses full-review TF-IDF unigram/bigram features and logistic regression. The TextCNN uses a training-only vocabulary capped at 20,000 entries, trainable 100-dimensional embeddings, 64 filters for each convolution width 3, 4 and 5, max pooling, dropout 0.4 and a single output logit. Padding-only windows are excluded from pooling.

Training uses Adam with learning rate 0.001, batch size 64, and binary cross-entropy with logits. Weighted conditions set the positive-class weight to the training negative/positive ratio. The maximum is six epochs with early-stopping patience two. Checkpoints and CNN configuration are selected using validation average precision (AP); classification thresholds maximize validation spoiler F1. AP is not trapezoidal PR-AUC.

The executed notebook contains four head-only conditions: 128 and 256 tokens, each with and without class weighting. The optional higher-dropout follow-up was disabled. No head–tail experiment was run in this version. The baseline receives full reviews while the CNN receives truncated prefixes, so their comparison does not isolate architecture alone.

## Validation results

Precision, recall and F1 below use each condition's validation-selected threshold.

| TextCNN input | Class weighting | Threshold | Precision | Recall | F1 | AP |
|---|---|---:|---:|---:|---:|---:|
| First 128 tokens | No | 0.1712 | 0.3107 | 0.7683 | 0.4424 | 0.4186 |
| First 128 tokens | Yes | 0.5444 | 0.3407 | 0.6373 | 0.4440 | 0.4146 |
| First 256 tokens | No | 0.1839 | 0.3608 | 0.7238 | 0.4815 | 0.4799 |
| First 256 tokens | Yes | 0.4528 | 0.3895 | 0.6516 | 0.4875 | 0.4861 |

Longer input improved AP under both weighting settings. Class weighting slightly reduced AP at 128 tokens and increased it by 0.0063 at 256 tokens. For the 256-token conditions, weighting increased precision and reduced recall at their separately optimized thresholds; it did not simply increase recall.

The selected CNN uses 256 tokens and weighted loss, with threshold 0.4528. The TF-IDF baseline still achieved higher validation F1 (0.5217) and AP (0.5211). Its accuracy was 68.93%, below the all-negative reference's 73.90%, because the F1-selected operating point trades false warnings for identifying spoiler reviews. The all-negative reference has zero spoiler recall and F1.

Training loss decreased in all four runs. Validation AP plateaued for the 128-token models, whereas the selected 256-token weighted model achieved its highest AP at epoch six. The outputs do not show a sustained validation decline for the selected model. Recorded CPU run times ranged from approximately 6.6 to 18.7 minutes and are not a controlled hardware benchmark. All conditions use one seed, so small differences are not evidence of statistical significance.

## Held-out test results

| Model | Accuracy | Spoiler precision | Spoiler recall | Spoiler F1 | Macro-F1 | AP |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.6973 | 0.4526 | 0.6629 | 0.5380 | 0.6564 | 0.5324 |
| TextCNN: 256 tokens, weighted loss | 0.6484 | 0.4015 | 0.6579 | 0.4987 | 0.6140 | 0.4893 |

The baseline outperformed the selected CNN on every reported test metric. Its F1 advantage was 0.0393 and its AP advantage was 0.0431. These results provide no evidence that this TextCNN configuration adds predictive value over the full-review linear baseline.

The CNN identified 1,048 of 1,593 spoiler-marked reviews and missed 545. Counts consistent with the saved metrics are 1,562 false positives and 2,837 true negatives. Approximately 59.8% of CNN warnings disagreed with non-spoiler reference labels. Some disagreements could involve missing user flags, but their true proportion has not been established.

The selected CNN's test F1 and AP were close to its validation values, with no large observed performance collapse on unseen titles. This observation is limited to one split and seed.

## Sensitivity and genre analysis

Removing explicit occurrences of “spoiler” and “spoilers” at validation inference reduced CNN AP from 0.4861 to 0.4410, F1 from 0.4875 to 0.4738, and recall from 0.6516 to 0.6214 at the unchanged threshold. The model is sensitive to these terms. This perturbation does not prove exclusive reliance on them; deletion can also change which later tokens fit inside the input limit.

Family reviews had the highest observed spoiler F1 (0.5621; 456 reviews from 28 titles), while Comedy had the lowest (0.4219; 1,273 reviews from 95 titles). Horror had the highest AP (0.5760; 488 reviews from 24 titles), while Comedy had the lowest AP (0.3782). Fantasy had the highest recall (0.7323). Genre groups overlap and differ in sample sizes, spoiler prevalence and title composition, so these are descriptive comparisons rather than established genre effects.

## Manual audit of potential missing flags

The notebook exported 20 random test reviews labeled non-spoiler but predicted as spoiler, mixed with six controls. The reader sheet hides labels, scores and group membership. All 26 rows still lacked a judgment or rationale in the recorded output, so there are no completed manual-audit findings. Exporting a sample does not verify that the model detected user omissions.

To complete the audit, fill `human_judgment` with `spoiler`, `not_spoiler` or `uncertain` in `spoiler_outputs/manual_audit_reader.csv`, add a `rationale`, and optionally record plot verification in `verification_source`. Preserve case IDs and text, save UTF-8 CSV, and rerun the aggregation cell. Keep the key hidden until judgments are complete. Existing audit sheets are preserved.

A consequential revelation, such as a major twist, culprit identity or important character fate, qualifies under the audit's operational definition. Generic evaluations and vague references to endings do not alone establish a spoiler. Use `uncertain` when context is insufficient. Results describe selected disagreement cases and cannot estimate overall accuracy, missed-spoiler recall or dataset-wide label error. Test-audit findings must not be used to tune the model.

## Spoiler-warning demonstration

After training and loading the selected model, call `spoiler_warning('Your review text')`. The function accepts text without a user-supplied spoiler label and returns a score, threshold, processed-token count and suggested warning action.

| Synthetic example | Score | Warning |
|---|---:|---|
| General comment about acting and pacing | 0.0853 | No |
| Revelation that the detective's partner was the murderer | 0.5964 | Yes |
| Vague comment about a surprising ending | 0.1682 | No |

These constructed examples illustrate inference and do not constitute evaluation evidence. Scores are uncalibrated outputs. Only the first 256 regex tokens are processed by the selected model, so later revelations can be missed. The classifier does not locate spoiler sentences or establish understanding of a film's plot.

## Running the notebook

Use `MGSC673_IMDb_Spoiler_Detection_Interpreted.ipynb`. Place `IMDB_reviews.json` and, optionally, `IMDB_movie_details.json` beside the notebook, matching `DATA_DIR = Path('.')`. Relative paths resolve against the kernel's current working directory. JSON arrays and JSON Lines are supported. Raw data is acquired separately under the provider's terms.

Install the project dependencies into the notebook's Python environment:

```bash
python -m pip install torch pandas numpy scikit-learn matplotlib seaborn ijson joblib jupyterlab
python -m jupyter lab
```

Select the matching kernel and run cells in order. The executed configuration selects CUDA if available, otherwise CPU; the recorded run used CPU. Lowering the sample size or epochs provides a pipeline check but will not reproduce the reported experiment. Re-running training regenerates results; exact reproducibility can vary across platforms and package versions.

## Outputs and limitations

`spoiler_outputs/` stores split manifests, vocabulary, checkpoints, training histories, validation comparisons, final test metrics, test predictions, genre summaries, audit files and run configuration. This executed version does not generate stratified-sampling counts or a head–tail comparison. The saved configuration records package versions. Match checkpoints to their vocabulary and settings.

The experiment is limited by uncertain user flags, lack of explicit plot context, prefix truncation, repeated authors across partitions, a single seed and split, and English IMDb sampling. The optional follow-up and manual verification remain incomplete. The model is an exploratory warning prototype; reliable automatic deployment has not been established.

When publishing, exclude raw data, local environments, checkpoints and audit exports unless their inclusion is deliberate and permitted. If JSON files are stored at repository root, add their exact filenames to `.gitignore`; a `data/` exclusion alone will not cover them. Review saved notebook outputs before committing. No third-party data license is granted by this repository.

## Reference

Rishabh Misra. *IMDB Spoiler Dataset*. [Paper](https://arxiv.org/abs/2212.06034). Record the dataset acquisition date and version in the completed project and follow applicable course AI-assistance requirements.
