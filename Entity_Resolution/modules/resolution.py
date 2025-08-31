# modules/resolution.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)
    latent = Dense(16, activation="relu")(encoded)
    decoded = Dense(32, activation="relu")(latent)
    decoded = Dense(64, activation="relu")(decoded)
    output_layer = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer=Adam(0.001), loss="mse")
    return autoencoder

def run_entity_resolution(dfs, mse_weight=0.5, tfidf_weight=0.5):
    df1, df2 = dfs["base"], dfs["incoming"]

    # Text Similarity
    common_cols = list(set(df1.columns) & set(df2.columns))
    text_cols = [col for col in common_cols if df1[col].dtype == 'object' and df2[col].dtype == 'object']

    df1['text'] = df1[text_cols].fillna('').agg(' '.join, axis=1)
    df2['text'] = df2[text_cols].fillna('').agg(' '.join, axis=1)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix1 = vectorizer.fit_transform(df1['text'])
    tfidf_matrix2 = vectorizer.transform(df2['text'])

    text_sim_matrix = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

    # Autoencoder MSE Similarity
    encoder_input_cols = [
        col for col in common_cols
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col])
    ]

    if encoder_input_cols:
        scaler = MinMaxScaler()
        df1_encoded = scaler.fit_transform(df1[encoder_input_cols])
        df2_encoded = scaler.transform(df2[encoder_input_cols])

        autoencoder1 = build_autoencoder(df1_encoded.shape[1])
        autoencoder2 = build_autoencoder(df2_encoded.shape[1])

        autoencoder1.fit(df1_encoded, df1_encoded, epochs=10, batch_size=16, verbose=0)
        autoencoder2.fit(df2_encoded, df2_encoded, epochs=10, batch_size=16, verbose=0)

        recon1 = autoencoder1.predict(df1_encoded)
        recon2 = autoencoder2.predict(df2_encoded)

        mse1 = np.mean(np.square(df1_encoded - recon1), axis=1)
        mse2 = np.mean(np.square(df2_encoded - recon2), axis=1)

        min_len = min(len(mse1), len(mse2))
        mse_diff = np.abs(mse1[:min_len] - mse2[:min_len])
        mse_score = 1 - MinMaxScaler().fit_transform(mse_diff.reshape(-1, 1)).flatten()
    else:
        min_len = min(tfidf_matrix1.shape[0], tfidf_matrix2.shape[0])
        mse_score = np.ones(min_len)

    matches = []
    max_matches = min(10, tfidf_matrix1.shape[0], tfidf_matrix2.shape[0])
    for i in range(max_matches):
        j = np.argmax(text_sim_matrix[i])
        hybrid_score = tfidf_weight * text_sim_matrix[i, j] + mse_weight * mse_score[i]
        matches.append({
            "Row_Dataset1": i,
            "Row_Dataset2": j,
            "Text_Similarity": text_sim_matrix[i, j],
            "MSE_Score": mse_score[i],
            "Final_Similarity": hybrid_score,
            "Text1": df1.iloc[i]['text'],
            "Text2": df2.iloc[j]['text']
        })

    return pd.DataFrame(matches)