# Content and Review Notes

Prepared on 2026-10-06 from the supplied final paper, `2609.19916v1(1).pdf`.

## Source mapping

| Website content | Source |
|---|---|
| Paper title; 11 authors; seven numbered affiliations; equal contribution; correspondence | Paper title page |
| Collection and annotated entry overview | Figure 2, original crop |
| 오하운 entry | Table 3; English paraphrases for readability |
| Dataset distributions | Figure 3, reproduced unchanged |
| Contextual Identification and Definition Generation examples | Figure 4 |
| Specialized-domain odd-one-out | Table 5 |
| 헬스닥 gold source words and GPT-5.4 prediction | Table 8 |
| 49.96 F1; definition scores 5.93 → 7.18; nine-model table | Table 1 |
| Survey-year comparison for Task 4 | Figure 6(b), original plot |
| Linguistics majors 63.40 vs. GPT-5.4 51.01; all participants 75.40 vs. GPT-5.4 82.00 | Appendix D, Table 10 |
| Funding acknowledgments | Paper acknowledgments |

Full affiliations are available under “Full affiliations & author notes”; the hero institution line is only a compact overview. No personal author homepage URLs were inferred.

## Deliberately preserved source discrepancies

The user requested that Figure 3 remain unchanged. Its visible category totals include 986 general-vocabulary entries and 801 terminology entries, which sum to 1,787 rather than the stated benchmark total of 1,785. Its word-formation panel reports 1,058 MWEs and 58.9%, which is not an exact match to the headline denominator of 1,785. The website therefore states only the paper's overall total, 1,785, next to the unmodified figure and does not reconstruct or recalculate its category distribution.

Table 1 labels Solar Pro3 with a parameter count, while the appendix leaves that count undisclosed. The compact website table uses “Solar Pro3” without asserting a parameter count.

## Scope safeguards

- “2020–2024” refers to survey years, not necessarily calendar-year coinage dates. For example, the 2020 survey spans July 2019–June 2020.
- The 396-item source-recovery task uses selected blends and abbreviations. The website does not claim every entry has source-word annotations.
- The 헬스닥 interaction presents two alternatives for readers. The original model task is free-form, and this distinction is stated.
- English translations and glosses are reading aids, not benchmark model inputs.
- The Figure 6(b) survey-year claim applies only to Task 4. It is not a claim that every later neologism or every task is harder.
- Full-set source-recovery results and the matched 50-item human-comparison subsets are separate evaluations. The GPT-5.4 values 49.96 and 51.01 are not contradictory.
- The two human-comparison rows use different human groups, explicitly labeled. The page does not claim uniform human superiority.
- Task 4 uses LLM-judge scores, not percentage accuracy.
- No new model outputs or numerical experimental findings were generated.

## Figures

The images are crops of the supplied paper, not redrawn figures. WebP files are lossless display versions; PNG files are available for enlargement. `static/images/figure-sources.json` records source page numbers, crop bounds, dimensions and source PDF hash. Figure 6(b)'s standalone crop retains both plot panels, axes, legends and plotted data; its outer panel marker is identified in the HTML caption.

## Publication and citation

The venue badge follows the user's confirmed “Findings of EMNLP 2026” wording. Citation metadata and the downloadable BibTeX refer to the arXiv preprint; no unverified proceedings DOI, page range, or anthology URL has been added. A final proceedings citation can replace the arXiv entry when confirmed.

## Verification status

- Independent paper-to-HTML audit completed for author list, affiliations, examples, labels and numeric results.
- JavaScript syntax checked with Node.
- HTML parsed and checked for duplicate IDs, broken local paths, internal links, image alternative text and JSON-LD validity.
- BibTeX download checked against the displayed citation.
- Figure PNG/WebP dimensions and decoded pixel equivalence checked.
- Responsive layout rules included for desktop, tablet and mobile. Keyboard-friendly native details, radio inputs and dialog controls are used.
- Browser-based visual and interaction QA was unavailable in the managed execution environment. No browser pass is claimed.

## Distribution

This is a local, reviewable website package. No repository was created, no external repository was modified, and no public deployment was made. A self-contained preview is provided separately for easier review.
