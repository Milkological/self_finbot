"""
feedback/ — Prediction tracking and accuracy feedback loop for FinBot.

Modules
-------
tracker   : save_prediction()  — write a new row to the ticker's CSV after each run
resolver  : resolve_pending()  — fetch actual prices and fill outcome columns
accuracy  : get_context()      — compute stats and return calibration data for analysis
"""
