# Lived Time Disruption in Borderline Personality Disorder: A Deep Learning and NLP Approach to Subjective Time Perception

 The current repository contains all of the computations performed for the Master's thesis writen by Marta Sokol, 
a student of Cognitive Science at Osnabrueck University in the summer semester 2024.

*Note*: The directory **does not** contain the trained models nor any sensitive data (training data, evaluation data).

 ## Abstract
<p align="justify">
Emerging evidence from phenomenological research suggests that borderline personality disorder (BPD) is associated with altered time perception, yet empirical support for this claim is sparse. This study compared differences in subjective time perception between individuals with BPD and an age- and sex-matched control group, as expressed in Polish autobiographical texts (N=10). A computational model based on a fine-tuned transformer neural network was trained for sentiment analysis and supplemented with classical natural language processing techniques. This model extracted objective language-based proxy variables, such as word length and frequency, as well as it predicted subjective sentiments to reflect various aspects of time perception. Outputs were analyzed as a time series, where each sentence in a single autobiographical corpus was assigned a time perception value, representing the perception of time by an individual participant. The resulting time series were compared across the groups utilizing sample entropy and autocorrelation as indicators of the short- and long-term predictability and orderliness of time perception. The results reveal a significantly lower sample entropy (p < .001) and a non-significant, yet marked decrease in autocorrelation (p = .120) in the BPD group compared to the control group, indicating altered time perception in BPD individuals. This suggests that the temporal experience of BPD patients is characterized by the presence of repetitive, predictable motifs, accompanied by relatively high short-term temporal state variability. This study not only provides more accurate understanding of how BPD affects time perception but also highlights the value of computational methods in exploring the subjective experiences associated with psychiatric conditions.
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
