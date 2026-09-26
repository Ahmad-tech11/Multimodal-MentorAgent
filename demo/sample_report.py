"""Default demo scenario: a student practicum report on an Edge-AI IoT Fall
Detection System. The accompanying diagram (see flowchart_generator.py)
intentionally includes a Redis Cache block that this text never mentions —
the discrepancy the Agent_VisionAlign stage is meant to catch.
"""

DEFAULT_REPORT_TEXT = """\
Problem Statement:
Elderly patients living alone face high risk from unwitnessed falls, where delayed \
detection significantly worsens medical outcomes. Existing camera-based fall \
detection systems raise privacy concerns and require constant cloud connectivity, \
making them unsuitable for many home deployments. This project addresses the need \
for a privacy-preserving, low-latency fall detection system that operates primarily \
at the network edge.

Proposed Architecture & Technical Route:
The system uses a wearable Sensor Node equipped with a 3-axis accelerometer and \
gyroscope to continuously sample motion data. Raw sensor readings are streamed over \
BLE to an Edge Gateway (Raspberry Pi 4), which performs local preprocessing including \
noise filtering and windowing. A Lightweight CNN Inference module, exported to TFLite \
and running directly on the Edge Gateway, classifies each motion window as \
"fall" or "normal activity" using a model trained on the SisFall public dataset. \
When a fall is classified with confidence above a fixed threshold, the Alert Module \
immediately triggers a local audible alarm and sends a push notification to a \
caregiver's phone. Aggregated (non-raw) event logs are periodically synced to a \
Cloud Dashboard for longitudinal monitoring by family members or clinicians.

System Implementation & Feasibility:
The CNN was quantized to INT8 to fit within the Raspberry Pi's compute budget, \
achieving an average inference latency of 42ms per window, well within the \
real-time requirement for timely alerting. The BLE link between the sensor node and \
the edge gateway was benchmarked at a stable 150ms round-trip under typical home \
conditions. Power consumption of the sensor node was measured at 18mA average draw, \
giving an estimated battery life of approximately 5 days on a 200mAh coin cell.

Experimental Evaluation & Results:
The system was evaluated using the SisFall dataset (38 subjects, 19 fall types) \
plus 3 days of in-home pilot testing with 2 volunteers. It achieved 94.2% \
sensitivity and 91.7% specificity on the held-out test split, with a false alarm \
rate of 1.3 events per day during the in-home pilot, which the team considers \
acceptable but still improvable for wider deployment.
"""
