from nltk.tokenize import word_tokenize
import spacy
import spacy.lang.pl
import pandas as pd
import numpy as np
import pandas as pd
from collections import Counter


df_raw_dir = pd.read_excel("input_corpora")
df_valence_dir = pd.read_excel('output_corpora/valence')
df_arousal_dir = pd.read_excel('output_corpora/arousal')


df_raw = pd.read_excel("C:/Users/marta/OneDrive/Desktop/Dane/Clean corpora/Borderline/corpus_for_cleanup1.xlsx")
df_valence = pd.read_excel('C:/Users/marta/OneDrive/Desktop/Dane/RoBERTa output/valence_output_file.xlsx')
df_arousal = pd.read_excel('C:/Users/marta/OneDrive/Desktop/Dane/RoBERTa output/arousal_output_file.xlsx')

# For time series modelling:
window_size = 40


def get_lengths_and_frequencies(df_raw):

    # Load the Polish language model for spaCy
    nlp = spacy.load("pl_core_news_sm")

    # Drop any missing values to avoid processing errors
    df_raw.dropna(subset=['sentence'], inplace=True)

    # Flatten the array to process each sentence
    text = df_raw.values.flatten()

    # Define the set of punctuations to be cleaned from the text
    punctuations = ['(', ')', ';', ':', '[', ']', ',', '...', '…', '.', '„', '”', '!', '-', '?']

    # Initialize a list to hold the cleaned text
    cleaned_text = []

    # Process each sentence in the text
    for sentence in text:
        cleaned_words_in_sentence = []
        for word in word_tokenize(sentence, language='polish'):
            cleaned_word = word.lower()
            if cleaned_word not in punctuations and cleaned_word.isalpha():
                cleaned_words_in_sentence.append(cleaned_word)

        lemmatized_words = [token.lemma_ for token in nlp(" ".join(cleaned_words_in_sentence))]
        cleaned_text.append(lemmatized_words)

    # Flatten the list of lists into a single list of words
    all_words = [word for sublist in cleaned_text for word in sublist]

    # Calculate the frequency of each word
    word_frequency = Counter(all_words)

    # Find the maximum frequency to normalize word frequencies
    max_frequency = max(word_frequency.values())

    # Normalize the frequencies to a scale from 0 to 1
    normalized_scores = {word: freq / max_frequency for word, freq in word_frequency.items()}

    # Compute the average normalized frequency per sentence
    sentence_word_frequencies = []
    for sentence in cleaned_text:
        sentence_frequency = sum(normalized_scores.get(word, 0) for word in sentence)
        average_popularity = sentence_frequency / len(sentence) if sentence else 0
        sentence_word_frequencies.append(average_popularity)

    # Compute the average word length per sentence
    sentence_lengths = []
    for sentence in cleaned_text:
        total_word_length = sum(len(word) for word in sentence)
        num_words = len(sentence)
        average_word_length = total_word_length / num_words if num_words else 0
        sentence_lengths.append(average_word_length)

    new_df = pd.DataFrame({'sentence': df_raw.values.flatten(), 'length': sentence_lengths, 'frequency': sentence_word_frequencies})

    return new_df



def calculate_adjustments(new_df, df_raw):

    new_df.dropna(subset=['sentence'], inplace=True)
    
    # Calculate the quantiles for lengths and frequencies
    high_length_quantile = new_df['length'].quantile(0.98)
    low_length_quantile = new_df['length'].quantile(0.02)
    high_frequency_quantile = new_df['frequency'].quantile(0.98)
    low_frequency_quantile = new_df['frequency'].quantile(0.02)

    # Filter data to get top and bottom quantiles for length and frequency
    top_length = new_df[new_df['length'] >= high_length_quantile].sort_values(by='length', ascending=False)
    bottom_length = new_df[new_df['length'] <= low_length_quantile].sort_values(by='length', ascending=True)
    top_frequency = new_df[new_df['frequency'] >= high_frequency_quantile].sort_values(by='frequency', ascending=False)
    bottom_frequency = new_df[new_df['frequency'] <= low_frequency_quantile].sort_values(by='frequency', ascending=True)

    # Initialize lists to store adjustments
    frequency_list = []
    length_list = []

    # Calculate frequency adjustments
    for word in df_raw['sentence']:
        if word in top_frequency['sentence'].values:
            frequency_list.append(-0.267)  # Frequent words adjustment
        elif word in bottom_frequency['sentence'].values:
            frequency_list.append(0.267)   # Infrequent words adjustment
        else:
            frequency_list.append(0)

    # Calculate length adjustments
    for word in df_raw['sentence']:
        if word in top_length['sentence'].values:
            length_list.append(0.127)      # Longer words adjustment
        elif word in bottom_length['sentence'].values:
            length_list.append(-0.127)     # Shorter words adjustment
        else:
            length_list.append(0)

    # Combine frequency and length adjustments
    result = [x + y for x, y in zip(length_list, frequency_list)]
    return result



def process_data(df_valence, df_arousal):
    # Merge the valence and arousal dataframes
    df = pd.merge(df_valence, df_arousal, on='sentence', how='inner')

    # Clean the data
    df.dropna(subset=['sentence'], inplace=True)

    # Initialize a column for combined scores
    df['combined'] = 0

    # Calculate combined scores based on valence and arousal
    for idx, row in df.iterrows():
        valence_step = row['valence']
        arousal_step = row['arousal']
        if valence_step == 2 and arousal_step == 2:
            combined = -0.099
        elif valence_step == 0 and arousal_step in [0, 2]:
            combined = -0.202
        elif valence_step == 2 and arousal_step == 0:
            combined = 0.022
        else:
            combined = 0
        df.at[idx, 'combined'] = combined

    return df['combined']



def main_analysis(df_valence, df_arousal):

    # Process the valence and arousal data to get combined scores
    old_data = process_data(df_valence, df_arousal)

    # Calculate adjustments based on length and frequency
    adjustments = calculate_adjustments(get_lengths_and_frequencies(df_raw), df_raw)

    # Combine the old data scores with the new adjustments
    final_result = [x + y for x, y in zip(old_data, adjustments)]

    return final_result



import matplotlib.pyplot as plt

def exponential_smoothing(x, window_size):
    weights = np.exp(np.linspace(-1, 0, window_size))
    weights /= weights.sum()
    smoothed_x = np.convolve(x, weights, mode='valid')
    return smoothed_x

def calculate_autocorrelation(data, lag=1):
        # Ensure data is a 1D array
        data = np.array(data).flatten()
        
        # Calculate autocorrelation
        acorr = np.corrcoef(np.array([data[:-lag], data[lag:]]))
        print(f'Autocorrelation for corpus X: {acorr[0, 1]}')


smoothed_x = exponential_smoothing(main_analysis(df_valence, df_arousal), window_size)

# Pad the beginning of smoothed_x with zeros to match the length of x
smoothed_x = np.concatenate((np.zeros(window_size - 1), smoothed_x))

#plt.plot(final_result, label='Gaussian smoothed data', color='black')
plt.plot(smoothed_x[window_size-1:], label=f'Smoothed Data with window size={window_size}', color='blue')
plt.axhline(np.mean(main_analysis(df_valence, df_arousal)))
plt.legend()
#plt.show()
plt.savefig('foo.png')




