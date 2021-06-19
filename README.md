# Inter-Case Dynamics Aware Remaining Time Predictions
This python scripts were made as a part of my master thesis work which seeks to build a process-aware remaining time prediction model in the area of predictive process monitoring.
We specifically leverage inter-play of individual process instances (cases), i.e., inter-case dynamics caused by bathcing and non-batching patterns while making predictions. This allows us to lift the assumption of cases proceeding in a process in isolation. The proposed approach creates new inter-case features to make the existing remaining time prediction model aware of the intre-case dynamics in relevant process segments.
<br />
To create the inter-case features we need to first detect potential process segments with inter-case dynamics that cause high prediction errors. To do so, follow the steps:
* Make remaining time predictions using https://github.com/verenich/time-prediction-benchmark and calculate Relative Abslolute Error for each prediction. Save results and provide path to relevant directories in helper.py.
* Run notebook Detect_uncertain_segments.ipynb to analyze predictions and detect process segments with inter-case dynamics. Use the performance spectrum with error progression [<cite>1</cite>]
along with pattern taxonomy provided in [<cite>2</cite>] to identify inter-case patterns in selected uncertain segments.
* Save the identified process segments with their respective inter-case pattern in segments.json.

To create the inter-case features:
* run train_waiting_time_pred_model.py by providing the right file name and path to relevant directories. This script will create the prediction models for inter-case feature waiting time.
* run prepare_event_log_with_inter_case_features.py to add predicted inter-case features to event log.
* retrain remaining time prediction models using https://github.com/verenich/time-prediction-benchmark with the new feature enriched inter-case event log.





[1] E. L. Klijn and D. Fahland, "Identifying and Reducing Errors in Remaining Time Prediction due to Inter-Case Dynamics," 2020 2nd International Conference on Process Mining (ICPM), 2020, pp. 25-32, doi: 10.1109/ICPM49681.2020.00015
  <br />
[2] Denisov, Vadim & Belkina, Elena & Fahland, Dirk & Aalst, Wil. (2018). The Performance Spectrum Miner: Visual Analytics for Fine-Grained Performance Analysis of Processes. 

