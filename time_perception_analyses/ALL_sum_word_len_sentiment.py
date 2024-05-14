import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import spacy
from collections import Counter

# Function to get lengths and frequencies
def get_lengths_and_frequencies(df_raw):
    nlp = spacy.load("pl_core_news_sm")
    df_raw.dropna(subset=['sentence'], inplace=True)
    text = df_raw['sentence'].values

    punctuations = ['(', ')', ';', ':', '[', ']', ',', '...', '…', '.', '„', '”', '!', '-', '?']
    cleaned_text = []

    for sentence in text:
        cleaned_words_in_sentence = [word.lower() for word in word_tokenize(sentence, language='polish') if word.lower() not in punctuations and word.isalpha()]
        lemmatized_words = [token.lemma_ for token in nlp(" ".join(cleaned_words_in_sentence))]
        cleaned_text.append(lemmatized_words)

    all_words = [word for sublist in cleaned_text for word in sublist]
    word_frequency = Counter(all_words)
    max_frequency = max(word_frequency.values())
    normalized_scores = {word: freq / max_frequency for word, freq in word_frequency.items()}

    sentence_word_frequencies = [sum(normalized_scores.get(word, 0) for word in sentence) / len(sentence) if sentence else 0 for sentence in cleaned_text]
    sentence_lengths = [sum(len(word) for word in sentence) / len(sentence) if sentence else 0 for sentence in cleaned_text]

    return pd.DataFrame({'sentence': df_raw['sentence'].values, 'length': sentence_lengths, 'frequency': sentence_word_frequencies})

# Function to calculate adjustments
def calculate_adjustments(new_df):
    high_length_quantile = new_df['length'].quantile(0.98)
    low_length_quantile = new_df['length'].quantile(0.02)
    high_frequency_quantile = new_df['frequency'].quantile(0.98)
    low_frequency_quantile = new_df['frequency'].quantile(0.02)

    top_length = new_df[new_df['length'] >= high_length_quantile]
    bottom_length = new_df[new_df['length'] <= low_length_quantile]
    top_frequency = new_df[new_df['frequency'] >= high_frequency_quantile]
    bottom_frequency = new_df[new_df['frequency'] <= low_frequency_quantile]

    length_list = [0.127 if word in top_length['sentence'].values else (-0.127 if word in bottom_length['sentence'].values else 0) for word in new_df['sentence']]
    frequency_list = [-0.267 if word in top_frequency['sentence'].values else (0.267 if word in bottom_frequency['sentence'].values else 0) for word in new_df['sentence']]

    return [x + y for x, y in zip(length_list, frequency_list)]

# Function to process data
def process_data(df_valence, df_arousal):
    df = pd.merge(df_valence, df_arousal, on='sentence', how='inner')
    df.dropna(subset=['sentence'], inplace=True)
    df['combined'] = 0

    for idx, row in df.iterrows():
        valence_step = row['valence']
        arousal_step = row['arousal']
        combined = -0.099 if valence_step == 2 and arousal_step == 2 else (-0.202 if valence_step == 0 and arousal_step in [0, 2] else (0.022 if valence_step == 2 and arousal_step == 0 else 0))
        df.at[idx, 'combined'] = combined

    return df['combined']

# Main analysis function
def main_analysis(df_raw, df_valence, df_arousal):
    old_data = process_data(df_valence, df_arousal)
    new_df = get_lengths_and_frequencies(df_raw)
    adjustments = calculate_adjustments(new_df)
    return [x + y for x, y in zip(old_data, adjustments)]

# Exponential smoothing function
def exponential_smoothing(x, window_size):
    weights = np.exp(np.linspace(-1., 0., window_size))
    weights /= weights.sum()
    return np.convolve(x, weights, mode='valid')

# # Plotting function
# def plot_data(smoothed_x, raw_data):
#     x = range(len(smoothed_x))
#     fig, ax = plt.subplots()
#     y2 = raw_data[len(raw_data) - len(smoothed_x):]
#     ax.plot(x, y2, label='Subjective time perception in computed effect sizes', color='gray')
#     ax.axhline(np.mean(smoothed_x), color='red', linestyle='--', label='Mean')
#     ax.plot(x, smoothed_x, label='Subjective time perception in mean effect sizes', color='blue')
#     ax.legend(fontsize=9)
#     ax.set_title('Subjective time perception in BPD patients in a single corpus')
#     ax.set_xlabel('Sentence index')
#     ax.set_ylabel('Effect size (Hedges\' g)')
#     #plt.show()

# Main loop to process files
num_files = 8
base_paths = {
    "raw": "input_corpora/controls/",
    "valence": "valence_prediction/output_corpora/",
    "arousal": "arousal_prediction/output_corpora/"
}

output_df = pd.DataFrame()

for i in range(1, num_files + 1):
    df_raw = pd.read_excel(os.path.join(base_paths['raw'], f"control_input{i}.xlsx"))
    df_valence = pd.read_csv(os.path.join(base_paths['valence'], f"control_input{i}_valence.csv"))
    df_arousal = pd.read_csv(os.path.join(base_paths['arousal'], f"control_input{i}_arousal.csv"))
    
    final_result = main_analysis(df_raw, df_valence, df_arousal)
    smoothed_result = exponential_smoothing(final_result, window_size=40)
    output_df[f'Corpus no. {i}'] = pd.Series(smoothed_result)
    #plot_data(smoothed_result, final_result)
    print(f"Corpus {i} of {num_files} done.")

print(output_df)

output_df.to_excel("control_time_output.xlsx", index=False)