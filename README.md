# Inter-Case Dynamics Aware Remaining Time Predictions
This python scripts were made as a part of my master thesis work which seeks to build a process-aware remaining time prediction model in the area of predictive process monitoring.
We specifically leverage inter-play of individual process instances (cases), i.e., inter-case dynamics caused by bathcing and non-batching patterns while making predictions. This allows us to lift the assumption of cases proceeding in a process in isolation. The proposed approach creates new inter-case features to make the existing remaining time prediction model aware of the intre-case dynamics in relevant process segments.
<br />
To create the intre-case features we need to first detect potential process segments with inter-case dynamics that cause high prediction errors. To do so, follow the steps:
* Make remaining time predictions using https://github.com/verenich/time-prediction-benchmark and calculate Relative Abslolute Error for each prediction. Save results and provide path to relevant directories in helper.py
* Run notebook Detect_uncertain_segments.ipynb to analyze predictions and detect process segments with inter-case dynamics.
