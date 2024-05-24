# Lived Time Disruption in Borderline Personality Disorder: A Deep Learning and NLP Approach to Subjective Time Perception

[![DOI](https://zenodo.org/badge/773448922.svg)](https://zenodo.org/doi/10.5281/zenodo.11276478)

 The current repository contains all of the computations performed for the Master's thesis writen by Marta Sokol, 
a student of Cognitive Science at Osnabrueck University in the summer semester 2024.

*Note*: The directory **does not** contain the trained models nor any sensitive data (training data, evaluation data).

 ## Abstract
<p align="justify">
Emerging evidence from phenomenological research suggests that borderline personality disorder (BPD) is associated with altered time perception; however, empirical support for this claim is sparse. This study used Polish autobiographical texts to compare differences in subjective time perception between individuals with BPD (<i>n</i> = 5) and a sex-matched control group (<i>n</i> = 5). A computational model based on a fine-tuned transformer neural network was trained for sentiment analysis and supplemented with classical natural language processing techniques. This model extracted objective language-based proxy variables such as word length and frequency and predicted subjective sentiments to estimate time perception in real-time. Outputs were analyzed as a time series where each sentence in a single autobiographical corpus was assigned a time perception value representing the perception of time by an individual participant. The resulting time series were compared across the groups using sample entropy and autocorrelation as indicators of time perception's short- and long-term predictability and orderliness. The results reveal significantly lower sample entropy (<i>p</i> < .001) and significantly decreased autocorrelation (<i>p</i> = .032) in the BPD group compared to the control group, indicating altered time perception in BPD individuals. These findings suggest that the temporal experience of BPD patients is characterized by repetitive, predictable motifs accompanied by relatively high short-term temporal state variability. This study not only provides a more accurate understanding of how BPD affects time perception but also highlights the value of computational methods in exploring the subjective experiences associated with psychiatric conditions.
 </p>

 ## Directory structure
```
.
├───arousal_prediction
│   ├───data
│   ├───output_model
│   └───__pycache__
├───time_perception_analyses
└───valence_prediction
    ├───data
    ├───output_model
    └───__pycache__
```
