# Course submission checklist

Status updated: October 8, 2026. This checklist separates items present in the repository from items still needed for the course submission.

## Git and folder status

The course implementation, EOS evaluator fix, tests, executed notebook, and current evaluation documentation are maintained on `runqi/sft-work`. Branch integration into `main` is handled through a separate pull request. `docs/final-report/` is Git-ignored at the user’s request and remains on this computer. `docs/report-notes-memo.md` is also ignored and private. `data/`, `models/`, and `runs/` are ignored local data and run artifacts. The Overleaf ZIP is a report-source package, not the full course-submission ZIP.

## In the project

| Item | Current state | Location |
|---|---|---|
| Source code and tests | Present; course code is primarily trained with TensorFlow/Keras | `src/`, `tests/` |
| Executed course notebook | Present with saved tables and plots; review mode does not rerun training | `Final_Project.ipynb` |
| Notebook dependencies | Present | `requirements-notebook.txt` |
| Training and evaluation records | Present locally; data and model files are excluded from Git | `runs/`, `data/`, `models/` |
| Research plans and evidence notes | Present; internal working material | `docs/final-report/plan/`, `docs/final-report/refs/` |
| LaTeX report source | Revised for final submission review; available locally and excluded from Git at the user’s request | local `docs/final-report/` |
| Overleaf source package | Available locally; contains the report sources and figures only | local `docs/final-report/DAT5565_Group21_Overleaf.zip` |
| Compiled report PDF | Local 20-page reading copy; source compiled with Tectonic and pages inspected | local `docs/final-report/DAT5565_Group21_Final_Report.pdf` |
| Readable report preview | Available locally; generated from the LaTeX source | local `docs/final-report/report-preview.html` |
| Result dashboard | Present; reads saved run records | `runs/final_results.html` |

## Still required or unverified

- **Word report:** The course asks for a Word document of at least five pages. The LaTeX source now compiles locally into a reviewed 20-page PDF, including the cover, references, and appendices. Conversion to Word and its final pagination remain incomplete. The local build does not verify Overleaf compilation.
- **Human answer review:** The report includes two selected Base/CPT+SFT/LSTM answer comparisons (source lines 823 and 7); these are illustrative. A documented review of a fixed sample for relevance, missing key information, repetition, and unsupported claims remains to be completed by the team.
- **Explainability:** The course instructions require an explainable-AI component. Training curves and example answers provide diagnostics but do not complete a dedicated explainability analysis.
- **Cross-validation:** The course instructions include cross-validation. The experiments use holdout validation and test splits; k-fold cross-validation was not run. Record the reason or confirm the instructor's accepted alternative.
- **Deployment:** No model service or API was deployed. The report discusses a staff-reviewed draft workflow, but a deployment guide or instructor-approved scope decision remains open.
- **Presentation:** Record a 10-minute video with all team members participating.
- **Team survey and contribution record:** Each of the three members must complete the collaboration survey individually. Confirm the report and presentation reflect the team's actual shared work; no unverified contribution percentages are assigned in the report.
- **LLM-use record:** The report names Codex assistance and describes checks performed. The team should add any other tools it used and verify the listed corrections reflect its experience.
- **Final course ZIP:** Not assembled. The Overleaf ZIP is only a LaTeX-source package. After the Word report, video, surveys, and final review are ready, assemble the required organized course-submission archive with its README, code, notebook, environment files, results, contribution documentation, and setup instructions. Keep large model weights separate if the course allows this.

## Evaluation scope to preserve in the report

Final Qwen Base/CPT+SFT metrics use the EOS-corrected WSL run in `runs/qa_eval_cpt_sft_test_200_eosfix_wsl/`; it uses the same 200-question sample but an RTX 2080 Ti. The earlier RTX 4090 Infra run is retained as a historical run with the old stop configuration. CPT-only and LSTM scores come from earlier runs. QA covers 200 of 1,573 test questions; MMLU covers all 272 professional_medicine questions with five-shot prompts. There is no SFT-only control, repeated outputs still occur, and neither text metrics nor the exam result establish clinical correctness.
