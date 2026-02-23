from typing import List, Dict, Tuple
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import uuid
import pathlib
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from sklearn.cluster import KMeans

keras = tf.keras
layers = tf.keras.layers
# PREPARATION ----------------------------------------------------------------------------------------------------------
def get_feature_metadata(demo_cols, continuous_cols):
    """
    Creates a dictionary mapping logical feature names to their
    one-hot encoded column names.

    Example output:
    {
        'AGEGRP': {'type': 'categorical', 'columns': ['AGEGRP_1', ...]},
        'EMPIN':  {'type': 'continuous',  'columns': ['EMPIN']}
    }
    """
    feature_meta = {}

    # 1. Group Categorical Features (based on prefixes like 'AGEGRP_')
    # We find prefixes by splitting on the last underscore
    prefixes = set()
    for col in demo_cols:
        if col not in continuous_cols:
            # Assuming format "NAME_VALUE", extract "NAME"
            # We use rsplit to handle names that might contain underscores
            prefix = col.rsplit('_', 1)[0]
            prefixes.add(prefix)

    for prefix in prefixes:
        cols = [c for c in demo_cols if c.startswith(prefix + '_')]
        # Sort to ensure consistent order
        cols.sort()
        if cols:
            feature_meta[prefix] = {'type': 'categorical', 'columns': cols}

    # 2. Add Continuous Features
    for col in continuous_cols:
        if col in demo_cols:
            feature_meta[col] = {'type': 'continuous', 'columns': [col]}

    return feature_meta
def prepare_data_for_generative_model(file_paths_dict: Dict[int, str], sample_frac: float = 1.0, random_state: int = 42
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, MinMaxScaler]]:
    """
    MASTER FUNCTION: Loads, combines, samples, and preprocesses census data.
    Used for both Training (Script 1) and Forecasting (Script 2).
    """

    # --- 1. Load and Combine All Datasets ---
    print("--- 1. Loading and combining all datasets... ---")
    all_dfs = []
    for year, path in file_paths_dict.items():
        try:
            df = pd.read_csv(path, dtype=str)
            df['YEAR'] = str(year)  # Create the YEAR column
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found {path}. Skipping.")

    if not all_dfs:
        raise ValueError("No data files were loaded. Check your file paths.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # --- 2. Household-level Sampling ---
    if sample_frac < 1.0:
        print(f"--- 2. Sampling {sample_frac * 100}% of households... ---")
        full_df['GLOBAL_HH_ID'] = full_df['YEAR'].astype(str) + '_' + full_df['HH_ID'].astype(str)
        unique_hh_ids = full_df['GLOBAL_HH_ID'].unique()
        sample_size = int(len(unique_hh_ids) * sample_frac)

        rng = np.random.RandomState(random_state)
        sampled_hh_ids = rng.choice(unique_hh_ids, size=sample_size, replace=False)

        full_df = full_df[full_df['GLOBAL_HH_ID'].isin(sampled_hh_ids)].copy()
        full_df = full_df.drop(columns=['GLOBAL_HH_ID'])
        print(f"   Sampled {sample_size} unique households.")
    else:
        print("--- 2. Using 100% of data. ---")

    # --- 3. Define Feature Lists ---

    # ID Columns to remove (Noise)
    ID_COLS_TO_DROP = ['HH_ID', 'EF_ID', 'CF_ID', 'PP_ID']

    # DEMOGRAPHICS (Outputs the model will Generate)
    DEMOGRAPHIC_FEATURES = [
        'YEAR', 'MARSTH', 'EMPIN', 'TOTINC', 'KOL', 'ATTSCH', 'CIP', 'NOCS',
        'GENSTAT', 'POWST', 'CITIZEN', 'LFTAG', 'CF_RP', 'COW', 'CMA',
        'AGEGRP', 'SEX', 'CFSTAT', 'INCTAX', 'HHSIZE', 'EFSIZE', 'CFSIZE',
        'PR', 'HRSWRK', 'MODE'
    ]

    # BUILDINGS (Inputs/Conditions the model uses to predict)
    # VALUE is here because it is a property of the building stock
    BUILDING_FEATURES = [
        'BUILTH', 'CONDO', 'BEDRM', 'ROOM', 'DTYPE', 'REPAIR', 'VALUE'
    ]

    # CONTINUOUS (To be scaled 0-1)
    CONTINUOUS_COLS = ['EMPIN', 'TOTINC', 'INCTAX', 'VALUE']

    # --- 4. Filter and Preprocess ---
    print("--- 3. Filtering to relevant columns... ---")
    all_features = list(set(DEMOGRAPHIC_FEATURES + BUILDING_FEATURES))
    all_cols_to_use = [
        col for col in all_features
        if col in full_df.columns and col not in ID_COLS_TO_DROP
    ]
    df_filtered = full_df[all_cols_to_use].copy()
    print(f"--- 4. Identified {len(all_cols_to_use)} total columns. ---")

    # --- 5. Scale and Encode ---
    print("--- 5. Scaling continuous data (MinMax) and one-hot encoding... ---")

    continuous_to_scale = [col for col in CONTINUOUS_COLS if col in df_filtered.columns]
    categorical_to_encode = [col for col in all_cols_to_use if col not in continuous_to_scale]

    # Fill missing values
    df_filtered[categorical_to_encode] = df_filtered[categorical_to_encode].fillna('Missing')

    scalers = {}
    for col in continuous_to_scale:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(0)
        scaler = MinMaxScaler()
        df_filtered[col] = scaler.fit_transform(df_filtered[[col]])
        scalers[col] = scaler

    # One-Hot Encode
    df_processed = pd.get_dummies(df_filtered, columns=categorical_to_encode, dtype=int)

    # --- 6. Get Final Column Lists ---
    def get_final_col_names(feature_list, continuous_list, processed_cols):
        final_cols = []
        for col in feature_list:
            if col in continuous_list:
                if col in processed_cols:
                    final_cols.append(col)
            elif col in all_cols_to_use:
                # Find all one-hot columns starting with this feature name
                # We sort them to ensure the order is always 1, 10, 11... same as training
                cols = [c for c in processed_cols if c.startswith(f"{col}_")]
                final_cols.extend(sorted(cols))
        return sorted(list(set(final_cols)))

    processed_columns_set = set(df_processed.columns)

    final_demographic_cols = get_final_col_names(
        DEMOGRAPHIC_FEATURES,
        continuous_to_scale,
        processed_columns_set
    )

    final_building_cols = get_final_col_names(
        BUILDING_FEATURES,
        continuous_to_scale,
        processed_columns_set
    )

    print("Data preparation complete.")
    return df_processed, final_demographic_cols, final_building_cols, scalers
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
# ARCHITECTURE ---------------------------------------------------------------------------------------------------------
def build_encoder(n_demo_features, n_bldg_features, latent_dim):
    """Builds the Encoder model with Batch Normalization."""
    demo_input = keras.Input(shape=(n_demo_features,), name="demo_input")
    bldg_input = keras.Input(shape=(n_bldg_features,), name="bldg_input")

    merged_input = layers.concatenate([demo_input, bldg_input])

    # --- UPDATED BLOCK ---
    x = layers.Dense(512)(merged_input)  # 1. Linear part
    x = layers.BatchNormalization()(x)  # 2. Normalize
    x = layers.ReLU()(x)  # 3. Activate

    x = layers.Dense(256)(x)  # 1. Linear part
    x = layers.BatchNormalization()(x)  # 2. Normalize
    x = layers.ReLU()(x)  # 3. Activate
    # --- END UPDATE ---

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(
        [demo_input, bldg_input],
        [z_mean, z_log_var, z],
        name="encoder"
    )
    return encoder
def build_decoder(n_bldg_features, latent_dim, feature_meta):
    """
    Builds a Multi-Head Decoder.
    Outputs a LIST of tensors (one per feature).
    """
    latent_input = keras.Input(shape=(latent_dim,), name="z_input")
    bldg_input = keras.Input(shape=(n_bldg_features,), name="bldg_input")

    merged_input = layers.concatenate([latent_input, bldg_input])

    # --- Shared Hidden Layers ---
    x = layers.Dense(256)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # --- Multi-Head Outputs ---
    outputs = []

    # We sort keys to ensure deterministic order
    for name in sorted(feature_meta.keys()):
        info = feature_meta[name]
        output_dim = len(info['columns'])

        if info['type'] == 'categorical':
            # Head for Categorical (Softmax for probability distribution)
            head = layers.Dense(output_dim, activation='softmax', name=f"out_{name}")(x)
        else:
            # Head for Continuous (Sigmoid for 0-1 scaling)
            head = layers.Dense(output_dim, activation='sigmoid', name=f"out_{name}")(x)

        outputs.append(head)

    # Define model
    decoder = keras.Model(
        [latent_input, bldg_input],
        outputs,  # Returns a list of outputs
        name="decoder"
    )
    return decoder
class MultiHeadCVAE(keras.Model):
    def __init__(self, encoder, decoder, feature_meta, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.feature_meta = feature_meta  # Store metadata to slice inputs
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        # Inputs are (demo_data, bldg_data)
        # We need demo_data BOTH as input to encoder AND as ground truth
        inputs, _ = data
        demo_input, bldg_input = inputs

        with tf.GradientTape() as tape:
            # 1. Encode
            z_mean, z_log_var, z = self.encoder([demo_input, bldg_input])

            # 2. Decode (returns a list of outputs)
            reconstructions = self.decoder([z, bldg_input])

            # 3. Calculate Reconstruction Loss (Sum of all heads)
            total_recon_loss = 0.0

            # We need to slice the 'demo_input' to match each head
            # We iterate in the SAME sorted order as the decoder
            sorted_features = sorted(self.feature_meta.keys())

            current_idx = 0
            for i, name in enumerate(sorted_features):
                info = self.feature_meta[name]
                dim = len(info['columns'])

                # Slice the true data for this feature
                y_true = demo_input[:, current_idx: current_idx + dim]
                y_pred = reconstructions[i]

                # Calculate appropriate loss
                if info['type'] == 'categorical':
                    # Categorical Crossentropy
                    loss = keras.losses.categorical_crossentropy(y_true, y_pred)
                else:
                    # Mean Squared Error (or Binary Crossentropy for 0-1)
                    loss = keras.losses.binary_crossentropy(y_true, y_pred)

                # Sum over batch and add to total
                total_recon_loss += tf.reduce_mean(loss)

                current_idx += dim

            # 4. KL Divergence
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # 5. Total Loss
            total_loss = total_recon_loss + (self.beta * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(total_recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
# TRAINING -------------------------------------------------------------------------------------------------------------
def train_cvae(df_processed, demo_cols, bldg_cols, continuous_cols=None, latent_dim=48, epochs=100, batch_size=4096):
    print("--- Preparing data for TensorFlow ---")

    feature_meta = get_feature_metadata(demo_cols, continuous_cols)
    n_demo_features = len(demo_cols)
    n_bldg_features = len(bldg_cols)

    demo_data = df_processed[demo_cols].values.astype(np.float32)
    bldg_data = df_processed[bldg_cols].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(((demo_data, bldg_data), demo_data))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("--- Building Multi-Head C-VAE ---")

    encoder = build_encoder(n_demo_features, n_bldg_features, latent_dim)
    decoder = build_decoder(n_bldg_features, latent_dim, feature_meta)

    # --- IMPROVEMENT 2: Lower Beta Manually ---
    # We force Beta to 0.1 to prioritize Reconstruction accuracy
    beta_weight = 0.1
    print(f"Using Manual KL Loss Beta weight: {beta_weight}")

    cvae = MultiHeadCVAE(encoder, decoder, feature_meta, beta=beta_weight)

    # --- IMPROVEMENT 3: Aggressive Learning Rate ---
    # Increased to 1e-3 to unstuck the model
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipvalue=1.0)

    cvae.compile(optimizer=optimizer)

    print("--- Starting Training ---")
    history = cvae.fit(dataset, epochs=epochs)

    return encoder, decoder, cvae, history

# FORECASTING -------------------------------------------------------------------------------------------------------
class ClusterMomentumModel:
    """
    Advanced Latent Space Forecaster.
    Instead of moving the entire population average (which kills diversity),
    this groups the population into clusters and moves each cluster
    along its own unique trajectory.
    """

    def __init__(self, n_clusters=5, decay_factor=0.95, recent_weight=0.5):
        self.n_clusters = n_clusters
        self.decay = decay_factor
        self.alpha = recent_weight
        self.kmeans = None
        self.cluster_velocities = {}  # {cluster_id: velocity_vector}
        self.last_year = None

    def fit(self, data_dict):
        """
        data_dict: {year: latent_matrix_of_shape (N, latent_dim)}
        """
        years = sorted(data_dict.keys())
        self.last_year = years[-1]

        print(f"   [ClusterMomentum] Fitting {self.n_clusters} clusters on base year {years[0]}...")

        # 1. Fit Clusters on the earliest year to define "Cohort Types"
        # (We assume these archetypes exist across years, e.g., "Students", "Retirees")
        base_data = data_dict[years[0]]
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(base_data)

        # 2. Calculate Velocity for EACH cluster
        print(f"   [ClusterMomentum] Calculating trajectories for {self.n_clusters} unique demographic groups...")

        for k in range(self.n_clusters):
            centroids = {}
            for year in years:
                # Extract data for this year
                z_data = data_dict[year]
                # Predict which points belong to cluster k (using the fixed kmeans model)
                labels = self.kmeans.predict(z_data)

                # If cluster is empty in a year (rare), use global mean, else cluster mean
                if np.sum(labels == k) == 0:
                    centroids[year] = np.mean(z_data, axis=0)
                else:
                    centroids[year] = np.mean(z_data[labels == k], axis=0)

            # Calculate Velocity Vectors (2006->2011 and 2011->2016)
            # Assuming 5-year steps. Adjust denominator if years differ.
            v_old = (centroids[2011] - centroids[2006]) / 5.0
            v_recent = (centroids[2016] - centroids[2011]) / 5.0

            # Weighted Momentum
            v_final = (self.alpha * v_recent) + ((1 - self.alpha) * v_old)
            self.cluster_velocities[k] = v_final

    def predict(self, z_source, source_year, target_year):
        """
        Projects a set of source points (z_source) into the future
        by applying the momentum of the cluster they belong to.
        """
        dt = target_year - source_year

        # 1. Assign source points to clusters
        labels = self.kmeans.predict(z_source)

        # 2. Apply specific momentum
        z_projected = np.zeros_like(z_source)

        # Effective velocity with decay
        # decay applied per 5-year step
        steps = dt / 5.0
        decay_mult = self.decay ** steps

        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.sum(mask) > 0:
                # v * dt * decay
                shift = self.cluster_velocities[k] * dt * decay_mult
                z_projected[mask] = z_source[mask] + shift

        return z_projected
# --- 2. Train Function ---
def train_temporal_model(encoder, df_processed, demo_cols, bldg_cols):
    print("\n--- Step 3: Temporal Modeling (Cluster-Based) ---")

    # Identify years
    year_cols = sorted([col for col in demo_cols if col.startswith('YEAR_')])
    years = [int(col.split('_')[1]) for col in year_cols]

    # Store FULL latent populations, not just means
    latent_history = {}

    print(f"   Extracting latent populations for: {years}")
    for year, col_name in zip(years, year_cols):
        year_df = df_processed[df_processed[col_name] == 1]

        # Encode chunks to avoid OOM if huge
        demo_data = year_df[demo_cols].values.astype(np.float32)
        bldg_data = year_df[bldg_cols].values.astype(np.float32)

        # Get Deterministic Z (Mean) for trajectory calculation
        z_mean, z_log_var, z = encoder.predict([demo_data, bldg_data], verbose=0)
        latent_history[year] = z_mean

    # Fit the Cluster Model
    temporal_model = ClusterMomentumModel(n_clusters=8, decay_factor=0.95)  # 8 Clusters for better granularity
    temporal_model.fit(latent_history)

    # Return the model and the MOST RECENT population (to serve as seed for future)
    last_year = years[-1]
    last_population_z = latent_history[last_year]

    return temporal_model, last_population_z, last_year
# --- 3. Generation Function ---
def generate_future_population(decoder, temporal_model, last_population_z, last_year,
                               df_processed, bldg_cols, target_year, n_samples, variance_factor=1.15):
    """
    Generates future population by resampling from the last known population
    and projecting individuals forward.
    """
    print(f"--- Starting Step 4: Forecasting {target_year} (Base: {last_year}) ---")

    # 1. Resample from the Last Known Population (Preserves structural diversity)
    # Instead of generating from a generic Gaussian, we pick real 'agents' from 2016/2021
    # and evolve them.
    indices = np.random.choice(len(last_population_z), size=n_samples, replace=True)
    z_source = last_population_z[indices]

    # 2. Project Forward (Cluster Momentum)
    print(f"   Projecting {n_samples} agents from {last_year} to {target_year}...")
    z_projected = temporal_model.predict(z_source, last_year, target_year)

    # 3. Apply Variance Inflation (Diffusion)
    # Add small noise to prevent exact duplicates and simulate increasing uncertainty
    # We calculate the inherent noise in the source and scale it
    std_dev = np.std(last_population_z, axis=0)
    # We add 15% of the natural variation as "drift noise"
    noise_scale = std_dev * (variance_factor - 1.0)
    noise = np.random.normal(0, noise_scale, size=z_projected.shape)
    z_final = z_projected + noise

    # 4. Get Building Conditions (Scenario-based)
    # Currently assuming 2021 building stock distribution persists.
    # (Can be modified to simulate "New Construction" scenarios)
    year_cols = sorted([col for col in df_processed.columns if col.startswith('YEAR_')])
    year_last_col = year_cols[-1]
    bldg_conditions = df_processed[df_processed[year_last_col] == 1][bldg_cols]
    bldg_future = bldg_conditions.sample(n_samples, replace=True).values.astype(np.float32)

    # 5. Decode
    print(f"   Decoding into demographic variables...")
    generated_list = decoder.predict([z_final, bldg_future], verbose=0)

    # Concatenate outputs (assuming decoder returns [demo_output, ...])
    # Adjust this concatenation based on your specific decoder structure.
    # If decoder outputs a single tensor, remove the concatenation.
    if isinstance(generated_list, list):
        generated_raw = np.concatenate(generated_list, axis=1)
    else:
        generated_raw = generated_list

    return generated_raw, bldg_future, z_final
# --- 4. Advanced Post-Processing ---
def quantile_mapping(predicted_values, reference_values):
    """
    Forces the predicted distribution to match the shape of the reference (historical) distribution
    while preserving the rank order of the predictions.
    Fixes 'Regression to the Mean' in continuous variables (Income, etc).
    """
    # Sort predictions to establish rank
    pred_sorted_indices = np.argsort(predicted_values)
    pred_sorted = predicted_values[pred_sorted_indices]

    # Generate target quantiles from reference data
    n_samples = len(predicted_values)
    reference_quantiles = np.percentile(reference_values, np.linspace(0, 100, n_samples))

    # Map back
    mapped_values = np.zeros_like(predicted_values)
    mapped_values[pred_sorted_indices] = reference_quantiles

    # Optional: Apply the MEAN SHIFT predicted by the VAE
    # (If VAE predicted the whole population got richer, we want to keep that shift
    # but use the 2016 'shape' including the high-income tail)
    vae_shift = np.mean(predicted_values) - np.mean(reference_values)
    final_values = mapped_values + vae_shift

    return final_values
def sample_categorical(logits, temperature=0.8):
    """
    Samples from logits with temperature scaling.
    Lower Temp (<1.0) = Sharper, more decisive (fixes 'muddy' categories).
    Higher Temp (>1.0) = More random.
    """
    # 1. Scale by temperature
    logits = np.array(logits) / temperature

    # 2. Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 3. Sample
    # Argmax is temperature -> 0. Here we use probabilistic sampling
    # Ideally, for speed in huge arrays, we might just use argmax if temp is very low
    # But let's do a weighted choice for correctness
    n, k = probs.shape
    choices = np.zeros(n, dtype=int)

    # Vectorized random choice is hard in numpy, using a loop or optimization
    # Fast approximation: Gumbel-Max trick
    u = np.random.uniform(0, 1, size=(n, k))
    gumbel = -np.log(-np.log(u))
    choices = np.argmax(logits + gumbel, axis=1)

    return choices
def post_process_generated_data(generated_raw_data, demo_cols, generated_bldg_data, bldg_cols, scalers, ref_df=None):
    """
    ref_df: The DataFrame of the last historical year (e.g. 2016). Used for Quantile Mapping.
    """
    print("--- Starting Post-Processing (Quantile Mapping + Temp Sampling) ---")

    # We need to know which columns in generated_raw_data correspond to which demo_cols
    # Assuming generated_raw_data columns align 1:1 with demo_cols

    # Create temporary DF
    df_gen = pd.DataFrame(generated_raw_data, columns=demo_cols)
    df_bldg = pd.DataFrame(generated_bldg_data, columns=bldg_cols)
    df_final = pd.DataFrame()

    # 1. Handle Continuous Variables (Inverse Scale + Quantile Mapping)
    for col_name, scaler in scalers.items():
        if col_name in df_gen.columns:
            raw_vals = df_gen[col_name].values.reshape(-1, 1)

            # Inverse Transform
            inv_vals = scaler.inverse_transform(raw_vals).flatten()

            # Apply Quantile Mapping if reference data exists and is continuous
            if ref_df is not None and col_name in ref_df.columns:
                if col_name in ['TOTINC', 'EMPIN', 'INCTAX', 'WAGES']:  # Target specific cols
                    print(f"   -> Applying Quantile Mapping to fix tails: {col_name}")
                    # We must assume ref_df[col_name] is already inverse_scaled?
                    # Usually ref_df is the processed (scaled) data passed in.
                    # Let's unscale the reference data first to get the real distribution shape
                    ref_vals_scaled = ref_df[col_name].values.reshape(-1, 1)
                    ref_vals = scaler.inverse_transform(ref_vals_scaled).flatten()

                    inv_vals = quantile_mapping(inv_vals, ref_vals)

            df_final[col_name] = inv_vals

    # 2. Handle Categorical Variables (One-Hot Decode with Temperature)
    # Group columns by prefix
    all_prefixes = set()
    for col in demo_cols:
        if '_' in col and col not in scalers:
            all_prefixes.add(col.rsplit('_', 1)[0])

    for prefix in all_prefixes:
        cat_cols = [col for col in demo_cols if col.startswith(f"{prefix}_")]

        if cat_cols:
            # Extract Logits/Probs for this feature
            logits = df_gen[cat_cols].values

            # Apply Temperature Sampling (Temperature 0.7 for sharpness)
            # Use argmax if you just want the most likely, but sample_categorical preserves diversity
            # For strict consistency, we use Argmax (Temp -> 0) as user requested improvement on "muddy"
            # But Gumbel-Max (Temp 0.8) is better for population synthesis.

            # Let's use simple Argmax for robustness unless the user specifically enables sampling
            # Using simple Argmax for now as it's safest for "muddy" outputs
            indices = np.argmax(logits, axis=1)

            # Map indices back to column names (e.g. "AGEGRP_1" -> "1")
            suffixes = [c.replace(f"{prefix}_", "") for c in cat_cols]
            df_final[prefix] = [suffixes[i] for i in indices]

    # Add Building columns
    bldg_prefixes = set()
    for col in bldg_cols:
        if '_' in col and col not in scalers:
            bldg_prefixes.add(col.rsplit('_', 1)[0])

    for prefix in bldg_prefixes:
        cat_cols = [col for col in bldg_cols if col.startswith(f"{prefix}_")]
        if cat_cols:
            indices = np.argmax(df_bldg[cat_cols].values, axis=1)
            suffixes = [c.replace(f"{prefix}_", "") for c in cat_cols]
            df_final[prefix] = [suffixes[i] for i in indices]

    return df_final
#VISUALIZATION FOR FORECASTING -----------------------------------------------------------------------------------------
def plot_latent_trajectory(encoder, temporal_model, df_processed, demo_cols, bldg_cols):
    print("--- Generating Latent Space Trajectory Plot ---")

    # 1. Get Historical Means (2006-2021)
    years = [2006, 2011, 2016, 2021]
    history_vectors = []

    for year in years:
        col_name = f"YEAR_{year}"
        if col_name in df_processed.columns:
            year_df = df_processed[df_processed[col_name] == 1]
            demo_data = year_df[demo_cols].values.astype(np.float32)
            bldg_data = year_df[bldg_cols].values.astype(np.float32)

            # Get latent positions
            z_mean, _, _ = encoder.predict([demo_data, bldg_data], verbose=0)

            # Calculate average center
            history_vectors.append(np.mean(z_mean, axis=0))

    # 2. Get Forecasted Means (2025, 2030)
    future_years = [2025, 2030]
    future_vectors = []
    for year in future_years:
        pred_z = temporal_model.predict([[year]])[0]
        future_vectors.append(pred_z)

    # 3. Fit PCA on History + Future to find the best 2D plane
    all_vectors = np.array(history_vectors + future_vectors)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_vectors)

    # 4. Plot
    plt.figure(figsize=(10, 8))

    # Plot History (Blue line)
    plt.plot(pca_result[:4, 0], pca_result[:4, 1], 'o-', label='Historical (06-21)', color='blue', markersize=8)

    # Plot Future (Red dashed line)
    # Connect 2021 to 2025
    plt.plot(pca_result[3:, 0], pca_result[3:, 1], 'o--', label='Forecast (25-30)', color='red', markersize=8)

    # Annotate points
    labels = years + future_years
    for i, txt in enumerate(labels):
        plt.annotate(txt, (pca_result[i, 0], pca_result[i, 1]), xytext=(5, 5), textcoords='offset points')

    plt.title(
        f"Demographic Drift in Latent Space (PCA Projection)\nExplained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    save_path = VALIDATION_FORECASTVIS_DIR /"Latent_Trajectory_Plot.png"
    plt.savefig(save_path)
    print(f"   Plot saved to {save_path}")
#VALIDATION OF FORECASTING ---------------------------------------------------------------------------------------------
def validate_forecast_trajectory(encoder, df_processed, demo_cols, bldg_cols, output_dir, n_clusters=8):
    """
    Validates the forecasting logic by visualizing the latent space trajectory of distinct CLUSTERS.
    Shows how different demographic groups evolve differently over time.
    """
    print(f"\n{'=' * 60}")
    print(f"🔮 VALIDATING FORECAST TRAJECTORY (Cluster-Based Momentum)")
    print(f"{'=' * 60}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract Full Historical Latent Populations
    print("1. Extracting Historical Latent Data...")
    year_cols = sorted([col for col in demo_cols if col.startswith('YEAR_')])
    years_hist = [int(col.split('_')[1]) for col in year_cols]

    latent_data = {}  # {year: matrix}

    for year, col_name in zip(years_hist, year_cols):
        year_df = df_processed[df_processed[col_name] == 1]
        if len(year_df) == 0: continue

        demo_data = year_df[demo_cols].values.astype(np.float32)
        bldg_data = year_df[bldg_cols].values.astype(np.float32)

        # Predict Mean Z
        z_mean, _, _ = encoder.predict([demo_data, bldg_data], verbose=0)
        latent_data[year] = z_mean

    # 2. Fit Cluster Model
    print(f"2. Fitting {n_clusters} Clusters & Calculating Trajectories...")
    model = ClusterMomentumModel(n_clusters=n_clusters, decay_factor=0.95)
    model.fit(latent_data)

    # 3. Prepare Trajectory Points for Plotting
    # We want to plot the CENTROID of each cluster over time
    trajectory_data = {k: {'years': [], 'points': []} for k in range(n_clusters)}

    # A) Historical Centroids (2006, 2011, 2016)
    for year in years_hist:
        z_data = latent_data[year]
        labels = model.kmeans.predict(z_data)
        for k in range(n_clusters):
            if np.sum(labels == k) > 0:
                centroid = np.mean(z_data[labels == k], axis=0)
                trajectory_data[k]['years'].append(year)
                trajectory_data[k]['points'].append(centroid)

    # B) Future Projections (2021, 2025, 2030)
    # We project the *last known centroid* forward
    years_future = [2021, 2025, 2030]
    last_year = years_hist[-1]

    for k in range(n_clusters):
        # Get the 2016 centroid for this cluster
        last_centroid = trajectory_data[k]['points'][-1]
        # Treat this centroid as a 'point' source for projection
        z_source = np.array([last_centroid])

        # We need to manually set the label for this source point to 'k' so the model applies 'k' momentum
        # Hack: The model's predict function re-predicts labels.
        # Since we are passing the centroid, it *should* fall into its own cluster.

        for yr in years_future:
            pred = model.predict(z_source, last_year, yr)[0]
            trajectory_data[k]['years'].append(yr)
            trajectory_data[k]['points'].append(pred)

    # 4. PCA Projection for Visualization
    print("3. Generating Plots...")
    # Gather all points to fit PCA
    all_points = []
    for k in trajectory_data:
        all_points.extend(trajectory_data[k]['points'])
    all_points = np.array(all_points)

    pca = PCA(n_components=2)
    pca.fit(all_points)  # Fit PCA on the centroids

    # 5. PLOTTING
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.get_cmap('tab10', n_clusters)

    for k in range(n_clusters):
        pts = np.array(trajectory_data[k]['points'])
        yrs = trajectory_data[k]['years']

        # Project to 2D
        pts_2d = pca.transform(pts)

        # Plot Line
        ax.plot(pts_2d[:, 0], pts_2d[:, 1], 'o-', color=colors(k), linewidth=2, label=f'Cluster {k}')

        # Annotate Start (06) and Forecast (30)
        ax.text(pts_2d[0, 0], pts_2d[0, 1], f"06", fontsize=8, color=colors(k))
        ax.text(pts_2d[-1, 0], pts_2d[-1, 1], f"30", fontsize=9, fontweight='bold', color=colors(k))

        # Mark the history/future split (2016)
        split_idx = yrs.index(2016)
        ax.scatter(pts_2d[split_idx, 0], pts_2d[split_idx, 1], s=100, facecolors='none', edgecolors=colors(k),
                   linestyle='--')

    ax.set_title(
        f"Latent Space Trajectories by Demographic Cluster (PCA)\nSplitting population reveals distinct evolution paths")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title="Demographic Archetypes")

    plt.tight_layout()
    plot_path = output_dir / "Validation_Forecast_Trajectory_Clusters.png"
    plt.savefig(plot_path)
    print(f"   ✅ Cluster Trajectory plot saved to: {plot_path}")
# --- 2. Validation: Distribution Hindcast (Resampling + Quantile Mapping) ---
def validate_forecast_distributions(encoder, decoder, df_processed, demo_cols, bldg_cols, scalers, output_dir):
    """
    Performs 'Hindcasting':
    1. Trains Cluster Model on 2006-2016.
    2. Takes REAL 2016 agents.
    3. Projects them to 2021.
    4. Applies Post-Processing (Quantile Mapping).
    5. Compares against REAL 2021.
    """
    print(f"\n{'=' * 60}")
    print(f"📊 VALIDATING FORECAST DISTRIBUTIONS (Hindcast 2021)")
    print(f"{'=' * 60}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train Model on History (06, 11, 16)
    print("1. Training Cluster Model on History (2006-2016)...")
    years_train = [2006, 2011, 2016]
    latent_history = {}

    # Also need 2016 log_var for variance calc
    last_log_var = None

    for year in years_train:
        col_name = f"YEAR_{year}"
        if col_name not in df_processed.columns: continue

        year_df = df_processed[df_processed[col_name] == 1]
        demo_data = year_df[demo_cols].values.astype(np.float32)
        bldg_data = year_df[bldg_cols].values.astype(np.float32)

        z_mean, z_log_var, _ = encoder.predict([demo_data, bldg_data], verbose=0)
        latent_history[year] = z_mean

        if year == 2016:
            last_log_var = z_log_var

    # Fit Model
    temporal_model = ClusterMomentumModel(n_clusters=8, decay_factor=0.95)
    temporal_model.fit(latent_history)

    # 2. Generate Synthetic 2021 (Resampling Strategy)
    print("2. Resampling & Projecting 2016 Agents to 2021...")

    # Source: 2016 Population
    z_source = latent_history[2016]

    # Target: 2021 Building Conditions (for fair comparison)
    target_year = 2021
    real_2021_df = df_processed[df_processed[f"YEAR_{target_year}"] == 1]
    if len(real_2021_df) == 0:
        print("❌ Error: No Real 2021 data found.")
        return

    # Match sample size to real data for density plotting
    n_samples = len(real_2021_df)

    # Resample 2016 agents to match N
    indices = np.random.choice(len(z_source), size=n_samples, replace=True)
    z_sampled = z_source[indices]

    # Project Forward (2016 -> 2021)
    z_projected = temporal_model.predict(z_sampled, 2016, 2021)

    # Apply Variance Inflation
    # (Using std dev of 2016 population as base noise scale)
    variance_factor = 1.15
    std_dev = np.std(z_source, axis=0)
    noise_scale = std_dev * (variance_factor - 1.0)
    z_final = z_projected + np.random.normal(0, noise_scale, size=z_projected.shape)

    # Decode
    bldg_conditions = real_2021_df[bldg_cols].values.astype(np.float32)
    gen_list = decoder.predict([z_final, bldg_conditions], verbose=0)
    if isinstance(gen_list, list):
        gen_matrix = np.concatenate(gen_list, axis=1)
    else:
        gen_matrix = gen_list

    # 3. Post-Process (Apply Quantile Mapping)
    print("3. Post-Processing & Quantile Mapping...")
    # Map generated columns back to DF structure
    df_gen = pd.DataFrame(gen_matrix, columns=demo_cols)

    # Prepare Plotting Data
    # Continuous
    cont_cols = [c for c in scalers.keys() if c in demo_cols]

    # Categorical Prefixes
    cat_prefixes = set()
    for col in demo_cols:
        if '_' in col and col not in scalers and not col.startswith('YEAR_'):
            cat_prefixes.add(col.rsplit('_', 1)[0])
    cat_prefixes = sorted(list(cat_prefixes))

    total_plots = len(cont_cols) + len(cat_prefixes)
    cols_grid = 6
    rows_grid = math.ceil(total_plots / cols_grid)

    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(18, 3.5 * rows_grid))
    axes = axes.flatten()
    plot_idx = 0

    # --- A) Plot Continuous (With Quantile Mapping) ---
    for col_name in cont_cols:
        ax = axes[plot_idx]

        # Real Data (Inverse Scaled)
        real_val_scaled = real_2021_df[col_name].values.reshape(-1, 1)
        real_val = scalers[col_name].inverse_transform(real_val_scaled).flatten()

        # Gen Data (Inverse Scaled)
        gen_val_scaled = df_gen[col_name].values.reshape(-1, 1)
        gen_val = scalers[col_name].inverse_transform(gen_val_scaled).flatten()

        # APPLY QUANTILE MAPPING (The fix!)
        # We map the generated distribution to match the shape of the PREVIOUS year (2016)
        # to simulate the forecasting constraint.
        # But for validation against 2021, if we map to 2021 Real, it's cheating.
        # So we map to 2016 Real (History) + Mean Shift.

        # Get 2016 History for mapping reference
        year_16_df = df_processed[df_processed["YEAR_2016"] == 1]
        hist_val_scaled = year_16_df[col_name].values.reshape(-1, 1)
        hist_val = scalers[col_name].inverse_transform(hist_val_scaled).flatten()

        gen_val_mapped = quantile_mapping(gen_val, hist_val)

        # KDE Plot
        sns.kdeplot(real_val, label='Real 2021', fill=True, color='skyblue', alpha=0.3, ax=ax)
        sns.kdeplot(gen_val_mapped, label='Forecast (QM)', fill=False, color='red', linestyle='--', linewidth=2, ax=ax)
        # Optional: Show Raw VAE output to show improvement
        # sns.kdeplot(gen_val, label='Raw VAE', color='gray', linestyle=':', ax=ax)

        ax.set_title(f"{col_name}")
        ax.legend(fontsize=8)
        plot_idx += 1

    # --- B) Plot Categorical ---
    for prefix in cat_prefixes:
        ax = axes[plot_idx]

        cat_cols = [c for c in demo_cols if c.startswith(f"{prefix}_")]
        indices = [demo_cols.index(c) for c in cat_cols]

        # Real Dist
        real_counts = real_2021_df[cat_cols].sum().values
        real_dist = real_counts / real_counts.sum()

        # Gen Dist (Summing probabilities directly is safer than argmax for aggregate plots)
        gen_probs = gen_matrix[:, indices]
        gen_dist = gen_probs.sum(axis=0) / gen_probs.sum()

        labels = [c.replace(f"{prefix}_", "") for c in cat_cols]
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, real_dist, width, label='Real 2021', color='skyblue')
        ax.bar(x + width / 2, gen_dist, width, label='Forecast', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(f"{prefix}")
        plot_idx += 1

    # Hide unused
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plot_path = output_dir / "Validation_Forecast_Distributions_Resampled.png"
    plt.savefig(plot_path)
    print(f"   ✅ Distribution Subplots saved to: {plot_path}")
# VISUALIZATION & TESTING ----------------------------------------------------------------------------------------------
def plot_training_history(history):
    """
    Plots the total, reconstruction, and KL loss from the Multi-Head C-VAE
    training history.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- FIX IS HERE ---
    # Switched back to 'loss' because Keras standardizes the main loss key
    ax1.plot(history.history['loss'], label='Total Loss')
    # --- END FIX ---

    ax1.set_title('Total Training Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Loss Components ---
    ax2.plot(history.history['recon_loss'], label='Reconstruction Loss')
    ax2.plot(history.history['kl_loss'], label='KL Loss')
    ax2.set_title('Loss Components over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Value')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
def check_reconstruction_quality(encoder, decoder, df_processed, demo_cols, bldg_cols, n_samples=3):
    """
    Picks a few real samples and compares them to the model's
    reconstruction. Handles Multi-Head Decoder output.
    """
    # 1. Select a few random samples
    data_sample = df_processed.sample(n_samples)

    # 2. Prepare the data
    demo_data = data_sample[demo_cols].values.astype(np.float32)
    bldg_data = data_sample[bldg_cols].values.astype(np.float32)

    # 3. Run data through the full VAE (Encode -> Decode)
    z_mean, z_log_var, z = encoder.predict([demo_data, bldg_data])
    reconstructed_data_list = decoder.predict([z, bldg_data])

    # --- FIX IS HERE ---
    # The multi-head decoder returns a LIST of arrays.
    # We must concatenate them horizontally (axis=1) to form one big matrix.
    # Since 'demo_cols' is sorted alphabetically, and the decoder builds heads
    # alphabetically, the order will match perfectly.
    reconstructed_matrix = np.concatenate(reconstructed_data_list, axis=1)
    # --- END FIX ---

    # 4. Convert back to DataFrames for easy comparison
    original_df = pd.DataFrame(demo_data, columns=demo_cols)

    # Use the concatenated matrix here
    reconstructed_df = pd.DataFrame(reconstructed_matrix, columns=demo_cols)

    print("--- Checking Reconstruction Quality (Sample 1) ---")
    print("\n--- ORIGINAL ---")
    # Show the 10 most "active" features for the first person
    print(original_df.iloc[0].nlargest(10))

    print("\n--- RECONSTRUCTED ---")
    # Show the same 10 features from the reconstructed data
    print(reconstructed_df.iloc[0][original_df.iloc[0].nlargest(10).index])
def validate_vae_reconstruction(encoder, decoder, df_processed, demo_cols, bldg_cols, continuous_cols, output_dir, n_samples=5):
    """
    Expanded evaluation: Checks samples, compares Original vs. Reconstructed,
    prints to console, AND saves the detailed report to a CSV file.
    """
    print(f"\n{'=' * 60}")
    print(f"  EXPANDED RECONSTRUCTION CHECK ({n_samples} Samples)")
    print(f"{'=' * 60}")

    # 1. Select random samples
    data_sample = df_processed.sample(n_samples)

    # 2. Prepare data
    demo_data = data_sample[demo_cols].values.astype(np.float32)
    bldg_data = data_sample[bldg_cols].values.astype(np.float32)

    # 3. Predict
    z_mean, z_log_var, z = encoder.predict([demo_data, bldg_data], verbose=0)
    reconstructed_list = decoder.predict([z, bldg_data], verbose=0)

    # Concatenate multi-head output into one matrix
    reconstructed_matrix = np.concatenate(reconstructed_list, axis=1)

    # 4. Create DataFrames
    df_orig = pd.DataFrame(demo_data, columns=demo_cols)
    df_recon = pd.DataFrame(reconstructed_matrix, columns=demo_cols)

    # 5. Identify Feature Groups
    categorical_prefixes = set()
    for col in demo_cols:
        if col not in continuous_cols:
            prefix = col.rsplit('_', 1)[0]
            categorical_prefixes.add(prefix)

    sorted_prefixes = sorted(list(categorical_prefixes))

    # --- LIST TO STORE RESULTS FOR CSV ---
    results_list = []

    # 6. Loop through each sample
    for i in range(n_samples):
        print(f"\n--- Sample {i + 1} / {n_samples} ---")
        print(f"{'FEATURE':<15} | {'ORIGINAL':<15} | {'PREDICTED':<15} | {'CONFIDENCE':<10} | {'STATUS'}")
        print("-" * 75)

        # A) Check Categorical Features
        for prefix in sorted_prefixes:
            cols = [c for c in demo_cols if c.startswith(prefix + '_')]

            # Original
            orig_row = df_orig.iloc[i][cols]
            orig_cat = orig_row.idxmax().replace(f"{prefix}_", "")

            # Reconstructed
            recon_row = df_recon.iloc[i][cols]
            pred_cat = recon_row.idxmax().replace(f"{prefix}_", "")
            confidence = recon_row.max()

            status = "Pass" if orig_cat == pred_cat else "Fail"
            status_icon = "✅" if status == "Pass" else "❌"

            print(f"{prefix:<15} | {orig_cat:<15} | {pred_cat:<15} | {confidence:.4f}     | {status_icon}")

            # Add to results list
            results_list.append({
                "Sample_ID": i + 1,
                "Feature": prefix,
                "Type": "Categorical",
                "Original": orig_cat,
                "Predicted": pred_cat,
                "Confidence/Diff": confidence,
                "Status": status
            })

        # B) Check Continuous Features
        for col in continuous_cols:
            if col in demo_cols:
                val_orig = df_orig.iloc[i][col]
                val_pred = df_recon.iloc[i][col]
                diff = abs(val_orig - val_pred)

                status = "Pass" if diff < 0.05 else "Fail"  # Threshold can be adjusted
                status_icon = "✅" if status == "Pass" else "⚠️"

                print(
                    f"{col:<15} | {val_orig:.4f}          | {val_pred:.4f}          | Diff: {diff:.3f}  | {status_icon}")

                # Add to results list
                results_list.append({
                    "Sample_ID": i + 1,
                    "Feature": col,
                    "Type": "Continuous",
                    "Original": val_orig,
                    "Predicted": val_pred,
                    "Confidence/Diff": diff,  # Storing Diff here for continuous
                    "Status": status
                })

    # --- 7. Save to CSV ---
    output_path = pathlib.Path(output_dir) / "Validation_VAE_Reconstruction/validation_vae_reconstruction.csv"
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Detailed reconstruction report saved to: {output_path}")
#ASSEMBLE HOUSEHOLD ----------------------------------------------------------------------------------------------------
def assemble_households(csv_file_path, target_year, output_dir, start_id=100):
    """
    Reads forecasted CSV, reconstructs households, saves the LINKED CSV.
    Uses simple sequential IDs (100, 101, 102...) for households.
    """
    print(f"\n--- Assembling Households for {target_year} ---")
    print(f"   Loading data from: {csv_file_path}")

    # 1. LOAD DATA
    df_population = pd.read_csv(csv_file_path)

    # Generate PIDs (Personal IDs) - We keep these as UUIDs or Random strings
    # to distinguish people within the house.
    df_population['PID'] = [str(uuid.uuid4())[:8] for _ in range(len(df_population))]

    # Ensure Types
    df_population['HHSIZE'] = pd.to_numeric(df_population['HHSIZE'], errors='coerce').fillna(1).astype(int)
    df_population['CF_RP'] = df_population['CF_RP'].astype(str).str.replace('.0', '', regex=False)

    final_households = []

    # --- INITIALIZE COUNTER ---
    current_hh_id = start_id

    # --- PHASE 1: SINGLES (HHSIZE = 1) ---
    singles_mask = df_population['HHSIZE'] == 1
    df_singles = df_population[singles_mask].copy()

    if not df_singles.empty:
        # Assign a range of IDs to singles all at once
        num_singles = len(df_singles)
        df_singles['SIM_HH_ID'] = range(current_hh_id, current_hh_id + num_singles)
        current_hh_id += num_singles  # Increment counter
        final_households.append(df_singles)

    print(f"   Processed {len(df_singles)} Single-Person Households.")

    # Remove singles from the pool
    df_remaining = df_population[~singles_mask].copy()

    # --- PHASE 2: FAMILIES (Heads = 1) ---
    df_family_heads = df_remaining[df_remaining['CF_RP'] == '1'].copy()
    df_members_2 = df_remaining[df_remaining['CF_RP'] == '2'].copy()
    df_members_3 = df_remaining[df_remaining['CF_RP'] == '3'].copy()

    pool_family_mem = df_members_2.sample(frac=1.0).to_dict('records')
    pool_non_family = df_members_3.sample(frac=1.0).to_dict('records')

    print(f"   Assembling {len(df_family_heads)} Family Households (Heads)...")

    family_batch = []
    for _, head_series in df_family_heads.iterrows():
        head = head_series.to_dict()

        # Assign Simple ID
        house_id = current_hh_id
        current_hh_id += 1

        head['SIM_HH_ID'] = house_id
        family_batch.append(head)

        slots_needed = head['HHSIZE'] - 1

        for _ in range(slots_needed):
            if pool_family_mem:
                member = pool_family_mem.pop()
            elif pool_non_family:
                member = pool_non_family.pop()
            else:
                # Clone fallback
                if not df_members_2.empty:
                    member = df_members_2.sample(1).to_dict('records')[0]
                else:
                    member = df_remaining.sample(1).to_dict('records')[0]
                member['PID'] = str(uuid.uuid4())[:8]

            member['SIM_HH_ID'] = house_id
            family_batch.append(member)

    if family_batch:
        final_households.append(pd.DataFrame(family_batch))

    # --- PHASE 3: ROOMMATES (Leftover CF_RP 3s) ---
    leftover_roommates = pd.DataFrame(pool_non_family)

    if not leftover_roommates.empty:
        print(f"   Assembling {len(leftover_roommates)} Roommate/Non-Family Agents...")

        for size in sorted(leftover_roommates['HHSIZE'].unique()):
            if size == 1: continue

            mates_of_size = leftover_roommates[leftover_roommates['HHSIZE'] == size]
            mate_list = mates_of_size.to_dict('records')

            roommate_batch = []
            while mate_list:
                head = mate_list.pop()

                # Assign Simple ID
                house_id = current_hh_id
                current_hh_id += 1

                head['SIM_HH_ID'] = house_id
                roommate_batch.append(head)

                slots_needed = size - 1
                for _ in range(slots_needed):
                    if mate_list:
                        member = mate_list.pop()
                    else:
                        member = mates_of_size.sample(1).to_dict('records')[0]
                        member['PID'] = str(uuid.uuid4())[:8]

                    member['SIM_HH_ID'] = house_id
                    roommate_batch.append(member)

            if roommate_batch:
                final_households.append(pd.DataFrame(roommate_batch))

    # --- Combine All ---
    if final_households:
        df_assembled = pd.concat(final_households, ignore_index=True)
    else:
        df_assembled = pd.DataFrame()

    print(f"--- Assembly Complete. Last ID used: {current_hh_id - 1} ---")

    # Final Validation
    if not df_assembled.empty:
        size_counts = df_assembled.groupby('SIM_HH_ID')['PID'].count()
        target_sizes = df_assembled.groupby('SIM_HH_ID')['HHSIZE'].first()
        mismatches = size_counts != target_sizes
        if mismatches.any():
            print(f"⚠️ WARNING: {mismatches.sum()} households have mismatched sizes!")
        else:
            print("✅ VALIDATION SUCCESS: All households have correct member counts.")

    # --- SAVE TO CSV ---
    save_filename = f"forecasted_population_{target_year}_LINKED.csv"
    save_path = pathlib.Path(output_dir) / save_filename
    df_assembled.to_csv(save_path, index=False)
    print(f"✅ Saved linked {target_year} data to: {save_path}")

    return df_assembled
#PROFILE MATCHER -------------------------------------------------------------------------------------------------------
# =============================================================================
# CLASS 1: MatchProfiler (The Linker)
# =============================================================================
class MatchProfiler:
    """
    Phase 2 & 3: Assigns GSS Schedule IDs to Census Agents.
    Updated to include Residential Variables (DTYPE, BEDRM, etc.) in matching logic.
    """

    def __init__(self, df_census, df_gss, dday_col="DDAY", id_col="occID", cols_match_t1=None):
        print(f"\n{'=' * 60}")
        print(f"⚙️  INITIALIZING PHASE 2: MATCH PROFILER")
        print(f"{'=' * 60}")

        self.df_census = df_census.copy()
        self.id_col = id_col
        self.dday_col = dday_col

        # --- UPDATED TIERS WITH RESIDENTIAL VARIABLES ---
        # Tier 1: Perfect Match
        if cols_match_t1 is None:
            self.cols_t1 = [
                "HHSIZE", "HRSWRK", "AGEGRP", "MARSTH", "SEX",
                "KOL", "NOCS", "PR", "COW", "MODE",
                "DTYPE", "BEDRM", "CONDO", "ROOM", "REPAIR"
            ]
        else:
            self.cols_t1 = cols_match_t1

        # Tier 2: Energy Drivers (Physical Dwelling Attributes + Key Drivers)
        self.cols_t2 = [
            "HHSIZE", "HRSWRK", "AGEGRP", "SEX", "COW",
            "DTYPE", "BEDRM", "CONDO", "ROOM", "REPAIR"
        ]

        # Tier 3: Constraints (Occupancy physics only)
        self.cols_t3 = ["HHSIZE", "HRSWRK", "AGEGRP"]

        # Tier 4: Fail-safe
        self.cols_t4 = ["HHSIZE"]

        # Split & Flatten GSS to create "Catalogs"
        print(f"   Splitting GSS by Day Type ({dday_col})...")

        # --- FIX: Only include columns that actually exist in GSS ---
        # This prevents KeyError if residential variables (DTYPE, etc.) are missing in GSS
        available_t1 = [c for c in self.cols_t1 if c in df_gss.columns]
        missing_t1 = list(set(self.cols_t1) - set(available_t1))

        if missing_t1:
            print(f"⚠️  Warning: The following match columns are MISSING in GSS and will be ignored in matching:")
            print(f"    {missing_t1}")

        # We must include all AVAILABLE match columns in the catalog
        catalog_cols = list(set([self.id_col] + available_t1 + ["HHSIZE"]))

        # Weekday Catalog (Unique Profiles)
        raw_wd = df_gss[df_gss[self.dday_col].isin([2, 3, 4, 5, 6])]
        self.catalog_wd = raw_wd[catalog_cols].drop_duplicates(subset=[self.id_col])

        # Weekend Catalog (Unique Profiles)
        raw_we = df_gss[df_gss[self.dday_col].isin([1, 7])]
        self.catalog_we = raw_we[catalog_cols].drop_duplicates(subset=[self.id_col])

        print(f"   ✅ Catalogs Created: WD={len(self.catalog_wd):,}, WE={len(self.catalog_we):,}")

    def run_matching(self):
        print(f"\n🚀 Starting Matching Loop...")
        results = []
        for idx, agent in tqdm(self.df_census.iterrows(), total=len(self.df_census), desc="Matching"):
            # 1. Find Weekday Match
            wd_id, wd_tier = self._find_best_match(agent, self.catalog_wd)
            # 2. Find Weekend Match
            we_id, we_tier = self._find_best_match(agent, self.catalog_we)

            row = agent.to_dict()
            row['MATCH_ID_WD'] = wd_id
            row['MATCH_TIER_WD'] = wd_tier
            row['MATCH_ID_WE'] = we_id
            row['MATCH_TIER_WE'] = we_tier
            results.append(row)

        return pd.DataFrame(results)

    def _find_best_match(self, agent, catalog):
        # Tier 1: Perfect Match
        mask = np.ones(len(catalog), dtype=bool)
        for col in self.cols_t1:
            if col in catalog.columns and col in agent:
                mask &= (catalog[col] == agent[col])
        matches = catalog[mask]
        if not matches.empty: return matches.sample(1)[self.id_col].values[0], "1_Perfect"

        # Tier 2: Energy Drivers
        mask = np.ones(len(catalog), dtype=bool)
        for col in self.cols_t2:
            if col in catalog.columns and col in agent:
                mask &= (catalog[col] == agent[col])
        matches = catalog[mask]
        if not matches.empty: return matches.sample(1)[self.id_col].values[0], "2_Drivers"

        # Tier 3: Constraints
        mask = np.ones(len(catalog), dtype=bool)
        for col in self.cols_t3:
            if col in catalog.columns and col in agent:
                mask &= (catalog[col] == agent[col])
        matches = catalog[mask]
        if not matches.empty: return matches.sample(1)[self.id_col].values[0], "3_Constraints"

        # Tier 4: FailSafe
        mask = (catalog["HHSIZE"] == agent["HHSIZE"])
        matches = catalog[mask]
        if not matches.empty: return matches.sample(1)[self.id_col].values[0], "4_FailSafe"

        return catalog.sample(1)[self.id_col].values[0], "5_Random"
# =============================================================================
# CLASS 2: ScheduleExpander (The Retriever)
# =============================================================================
class ScheduleExpander:
    """
    Phase 4: Retrieval & Expansion.
    Takes the Matched Census DF and the Raw GSS DF.
    Retrieves the original variable-length episode lists.
    """

    def __init__(self, df_gss_raw, id_col="occID"):
        print(f"\n{'=' * 60}")
        print(f"📂 INITIALIZING PHASE 4: SCHEDULE EXPANDER")
        print(f"{'=' * 60}")

        self.df_gss_raw = df_gss_raw
        self.id_col = id_col

        # Indexing the Raw GSS by occID for instant retrieval
        print("   Indexing GSS Episodes for fast retrieval...")
        self.gss_indexed = self.df_gss_raw.set_index(self.id_col).sort_index()
        print("   ✅ Indexing complete.")

    def get_episodes(self, matched_id):
        """
        Directly retrieves episodes based on the Schedule ID.
        Used by generate_full_expansion.
        """
        try:
            # .loc[[id]] ensures we return a DataFrame, not a Series
            return self.gss_indexed.loc[[matched_id]].copy()
        except KeyError:
            # If ID is missing (shouldn't happen if matching worked), return None
            return None
# =============================================================================
# HELPER FUNCTIONS & MAIN EXECUTION
# =============================================================================
def verify_sample(df_matched, expander, n=3):
    print(f"\n🔎 VERIFYING EXPANSION (Sample of {n})")
    for i, agent in df_matched.head(n).iterrows():
        id_wd = agent['MATCH_ID_WD']
        id_we = agent['MATCH_ID_WE']
        ep_wd = expander.get_episodes(id_wd)
        ep_we = expander.get_episodes(id_we)
        count_wd = len(ep_wd) if ep_wd is not None else 0
        count_we = len(ep_we) if ep_we is not None else 0
        print(f"   User {i}: WD={count_wd} rows | WE={count_we} rows")
def generate_full_expansion(df_matched, expander, output_path):
    print(f"\n💾 Expanding Schedules for {len(df_matched)} agents...")
    all_episodes = []

    # Use 'idx' as the Unique Agent ID
    for idx, agent in tqdm(df_matched.iterrows(), total=len(df_matched), desc="Expanding"):

        # List of residential variables to carry over
        res_vars = ["DTYPE", "BEDRM", "CONDO", "ROOM", "REPAIR", "PR"]

        # Expand Weekday
        ep_wd = expander.get_episodes(agent['MATCH_ID_WD'])
        if ep_wd is not None:
            ep_wd = ep_wd.copy()
            ep_wd['SIM_HH_ID'] = agent['SIM_HH_ID']
            ep_wd['Day_Type'] = 'Weekday'
            ep_wd['AgentID'] = idx  # Unique ID

            # --- CRITICAL: Ensure Residential Variables are carried over ---
            for var in res_vars:
                if var in agent:
                    ep_wd[var] = agent[var]

            all_episodes.append(ep_wd)

        # Expand Weekend
        ep_we = expander.get_episodes(agent['MATCH_ID_WE'])
        if ep_we is not None:
            ep_we = ep_we.copy()
            ep_we['SIM_HH_ID'] = agent['SIM_HH_ID']
            ep_we['Day_Type'] = 'Weekend'
            ep_we['AgentID'] = idx  # Unique ID

            # --- CRITICAL: Ensure Residential Variables are carried over ---
            for var in res_vars:
                if var in agent:
                    ep_we[var] = agent[var]

            all_episodes.append(ep_we)

    if all_episodes:
        full_df = pd.concat(all_episodes)
        print(f"   Sorting expanded data...")
        full_df = full_df.sort_values(by=['SIM_HH_ID', 'Day_Type', 'AgentID'])
        full_df.to_csv(output_path, index=False)
        print(f"✅ Saved Expanded File: {len(full_df):,} rows to {output_path.name}")
#-----------------------------------------------------------------------------------------------------------------------

#VALIDATION: PROFILE MATCHER -------------------------------------------------------------------------------------------
def validate_matching_quality(df_matched, expander, save_path=None):
    """
    Calculates validation metrics and saves the report to a text file.
    """
    # --- Buffer to capture output ---
    report_buffer = []

    def log(message):
        """Helper to print to console AND append to buffer."""
        print(message)
        report_buffer.append(message)

    log(f"\n{'=' * 60}")
    log(f"📊 VALIDATION REPORT (CORRECTED)")
    log(f"{'=' * 60}")

    # --- METHOD 1: TIER DISTRIBUTION ---
    log(f"\n1. MATCH QUALITY (TIER DISTRIBUTION)")
    log("-" * 40)

    for day_type in ['WD', 'WE']:
        col = f'MATCH_TIER_{day_type}'
        if col in df_matched.columns:
            counts = df_matched[col].value_counts(normalize=True) * 100
            log(f"\n   [{day_type} Matching Tiers]")
            for tier, pct in counts.items():
                log(f"      - {tier}: {pct:.1f}%")

    # --- METHOD 2: BEHAVIORAL CONSISTENCY ---
    log(f"\n2. BEHAVIORAL CONSISTENCY (Workers vs. Non-Workers)")
    log("-" * 40)

    # Filter for Employees (COW 1 or 2)
    # We take a larger sample (up to 500) for better accuracy
    sample_size = min(500, len(df_matched))
    workers = df_matched[df_matched['COW'].isin([1, 2])].sample(sample_size)

    work_minutes = []

    for _, agent in workers.iterrows():
        # Get Weekday episodes
        ep_wd = expander.get_episodes(agent['MATCH_ID_WD'])

        if ep_wd is not None:
            # Filter for Work Activities
            # Standard GSS Work Codes often start with '1' or '0'. Adjust if needed.
            work_acts = ep_wd[ep_wd['occACT'].astype(str).str.startswith(('1', '0', '8'))]

            total_duration = 0
            for _, row in work_acts.iterrows():
                s = row['start']
                e = row['end']

                # --- FIX FOR MIDNIGHT WRAP ---
                # If end time is smaller than start time (e.g. 02:00 < 23:00), adds 24h (1440 min)
                if e < s:
                    duration = (e + 1440) - s
                else:
                    duration = e - s

                total_duration += duration

            work_minutes.append(total_duration)

    avg_work = np.mean(work_minutes) if work_minutes else 0
    log(f"   👉 Average Work Duration for 'Employees' (n={sample_size}): {avg_work:.0f} minutes/day")

    if avg_work < 60:
        log("      ⚠️ WARNING: Low work duration. Check if 'occACT' filter matches your GSS codes.")
    elif avg_work > 300:
        log("      ✅ Success: Employees are performing ~5-8 hours of work.")
    else:
        log("      ℹ️ Note: Work duration is moderate. Verify part-time vs full-time mix.")

    # --- STEP 3: SAVE TO FILE ---
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_buffer))
            print(f"\n✅ Validation Report saved to: {save_path}")
        except Exception as e:
            print(f"\n❌ Error saving report: {e}")
#PROFILE MATCHER: POST-PROCESSING --------------------------------------------------------------------------------------
def merge_keys_into_forecast(df_forecast, df_keys):
    """
    Merges authoritative Census columns (CFSIZE, TOTINC) from the Keys file
    into the Forecast file using 'AgentID'.
    """
    print("\n🔗 Merging Keys (CFSIZE, TOTINC) into Forecast...")

    # Validation
    if 'AgentID' not in df_forecast.columns:
        print("❌ Error: Forecast file missing 'AgentID'. Cannot merge keys.")
        return df_forecast

    # Select columns to retrieve
    # We retrieve CFSIZE and TOTINC. We can also retrieve others if needed.
    cols_to_retrieve = ['CFSIZE', 'TOTINC', 'CF_RP']
    available_cols = [c for c in cols_to_retrieve if c in df_keys.columns]

    if not available_cols:
        print("⚠️ Warning: Keys file missing CFSIZE/TOTINC columns.")
        return df_forecast

    print(f"   Retrieved columns: {available_cols}")

    # Prepare Lookup DataFrame
    # df_keys is indexed 0..N. We assume AgentID in forecast corresponds to this index.
    df_lookup = df_keys[available_cols].copy()
    df_lookup['AgentID'] = df_lookup.index

    # Merge
    # We use 'left' merge to preserve all schedule rows
    # We suffix existing columns in forecast with '_old' to verify overwrite
    df_merged = df_forecast.merge(df_lookup, on='AgentID', how='left', suffixes=('_old', ''))

    # Cleanup: If the merge created duplicate columns (e.g. TOTINC_old vs TOTINC),
    # The new 'TOTINC' (from keys) is the one we want. We drop the _old version.
    for col in available_cols:
        old_col = f"{col}_old"
        if old_col in df_merged.columns:
            # If the Key value is NaN (rare), fallback to old value
            df_merged[col] = df_merged[col].fillna(df_merged[old_col])
            df_merged.drop(columns=[old_col], inplace=True)

    print(f"   ✅ Merge complete. Forecast now has accurate {available_cols}")
    return df_merged
class DTypeRefiner:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}

        self.base_features = ['BEDRM', 'ROOM', 'PR', 'HHSIZE', 'CONDO', 'REPAIR', 'TOTINC', 'CFSIZE']
        self.train_features = self.base_features + ['ROOM_PER_PERSON', 'BEDRM_RATIO', 'INCOME_PER_PERSON']

    def _ensure_consistent_scaling(self, df, is_training=False):
        if 'TOTINC' not in df.columns: return df

        mean_inc = df['TOTINC'].mean()
        max_inc = df['TOTINC'].max()

        status = "Training" if is_training else "Forecast"
        if mean_inc < 50 and max_inc < 50:
            print(f"   ⚠️ [{status}] DETECTED LOG/SCALED INCOME (Mean={mean_inc:.2f}).")
            print(f"      🔄 Converting Log -> Dollars (exp(x) - 1)...")
            df['TOTINC'] = np.expm1(df['TOTINC'])

        return df

    def _add_derived_features(self, df):
        df = df.copy()
        cols_to_numeric = ['HHSIZE', 'ROOM', 'BEDRM', 'TOTINC', 'CFSIZE']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # Fallbacks
        if 'CFSIZE' not in df.columns:
            df['CFSIZE'] = df['HHSIZE'] if 'HHSIZE' in df.columns else 1

        # Ratios
        df['HHSIZE'] = df['HHSIZE'].replace(0, 1)
        df['ROOM'] = df['ROOM'].replace(0, 1)

        df['ROOM_PER_PERSON'] = df['ROOM'] / df['HHSIZE']
        df['BEDRM_RATIO'] = df['BEDRM'] / df['ROOM']

        if 'TOTINC' in df.columns:
            df['INCOME_PER_PERSON'] = df['TOTINC'] / df['HHSIZE']
        else:
            df['INCOME_PER_PERSON'] = 0

        return df.fillna(0)

    def train_models(self, df_historic):
        print(f"\n🧠 Training DTYPE Refinement Models...")
        df_historic = self._ensure_consistent_scaling(df_historic, is_training=True)
        df_historic = self._add_derived_features(df_historic)

        # --- MODEL A: APARTMENTS (Coarse 2 -> 5, 6) ---
        subset_apt = df_historic[df_historic['DTYPE'].isin([5, 6])]
        if len(subset_apt) > 100:
            clf_apt = RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_leaf=4,
                random_state=42, class_weight='balanced', n_jobs=-1
            )
            clf_apt.fit(subset_apt[self.train_features], subset_apt['DTYPE'])
            self.models['Apt'] = clf_apt
            print(f"   ✅ Trained Apartment Splitter (n={len(subset_apt):,})")

        # --- MODEL B: OTHER DWELLINGS (Coarse 3 -> 2, 3, 4, 7, 8) ---
        subset_other = df_historic[df_historic['DTYPE'].isin([2, 3, 4, 7, 8])]
        if len(subset_other) > 100:
            custom_weights = {2: 2.0, 3: 2.0, 4: 1.0, 7: 1.0, 8: 1.0}
            clf_other = RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_leaf=2,
                random_state=42, class_weight=custom_weights, n_jobs=-1
            )
            clf_other.fit(subset_other[self.train_features], subset_other['DTYPE'])
            self.models['Other'] = clf_other
            print(f"   ✅ Trained 'Other' Decoder (n={len(subset_other):,})")

    def apply_refinement(self, df_forecast):
        print(f"\n✨ Applying Refinement with Quota Calibration...")

        df_forecast = self._ensure_consistent_scaling(df_forecast, is_training=False)
        df_enhanced = self._add_derived_features(df_forecast)

        # Ensure features exist
        missing = [c for c in self.train_features if c not in df_enhanced.columns]
        if missing:
            print(f"   ⚠️ Warning: Still missing features: {missing}. Filling 0.")
            for c in missing: df_enhanced[c] = 0

        X = df_enhanced[self.train_features].fillna(0)
        refined_dtype = df_forecast['DTYPE'].copy()

        # --- HELPER: Quota Sampling ---
        def apply_quota_sampling(model, X_subset, mask_subset, target_ratios):
            if mask_subset.sum() == 0: return

            probs = model.predict_proba(X_subset)
            classes = model.classes_
            df_probs = pd.DataFrame(probs, columns=classes, index=X_subset.index)

            final_assignments = pd.Series(index=X_subset.index, dtype=int)
            total_n = len(X_subset)
            available_indices = set(X_subset.index)

            # Iterate through classes
            for cls, ratio in target_ratios.items():
                if cls not in df_probs.columns: continue
                target_count = int(total_n * ratio)

                if target_count > 0 and available_indices:
                    # Pick top N most likely candidates
                    candidates = df_probs.loc[list(available_indices), cls].sort_values(ascending=False)
                    selected = candidates.head(target_count).index
                    final_assignments.loc[selected] = cls
                    available_indices -= set(selected)

            # Fill remainder
            if available_indices:
                remaining = list(available_indices)
                fallback = df_probs.loc[remaining].idxmax(axis=1)
                final_assignments.loc[remaining] = fallback

            return final_assignments

        # --- APPLY MODEL A: APARTMENTS ---
        if 'Apt' in self.models:
            mask = (df_forecast['DTYPE'] == 2)
            if mask.sum() > 0:
                # Historic Split: 34% High Rise, 66% Low Rise
                ratios_apt = {5: 0.34, 6: 0.66}
                assignments = apply_quota_sampling(self.models['Apt'], X[mask], mask, ratios_apt)
                refined_dtype.loc[mask] = assignments
                print(f"   Refined {mask.sum():,} Apartments (Calibrated)")

        # --- APPLY MODEL B: OTHER DWELLINGS ---
        if 'Other' in self.models:
            mask = (df_forecast['DTYPE'] == 3)
            if mask.sum() > 0:
                # Historic Split: Row(33%), Semi(29%), Duplex(29%), Mobile(7%), Other(2%)
                ratios_other = {3: 0.33, 2: 0.29, 4: 0.29, 8: 0.07, 7: 0.02}
                assignments = apply_quota_sampling(self.models['Other'], X[mask], mask, ratios_other)
                refined_dtype.loc[mask] = assignments
                print(f"   Refined {mask.sum():,} 'Other' dwellings (Calibrated)")

        df_forecast['DTYPE'] = refined_dtype
        df_forecast['DTYPE_Detailed'] = refined_dtype
        return df_forecast
def validate_refinement_model(historic_input, forecast_refined_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "Validation_Report_DTYPE.txt"
    report_buffer = []

    def log(message=""):
        print(message)
        report_buffer.append(str(message))

    log(f"\n{'=' * 60}")
    log(f"🕵️‍♂️ VALIDATING DTYPE REFINEMENT")

    if isinstance(historic_input, list):
        dfs = [pd.read_csv(p, low_memory=False) for p in historic_input]
        df_hist = pd.concat(dfs, ignore_index=True)
    else:
        df_hist = pd.read_csv(historic_input, low_memory=False)

    df_future = pd.read_csv(forecast_refined_path, low_memory=False)

    # Check stats
    if 'TOTINC' in df_hist.columns and 'TOTINC' in df_future.columns:
        log(f"   Stats Check: Historic TOTINC Mean: {df_hist['TOTINC'].mean():.2f}")
        log(f"   Stats Check: Forecast TOTINC Mean: {df_future['TOTINC'].mean():.2f}")

    dtype_labels = {
        1: "Single-detached", 2: "Semi-detached", 3: "Row house",
        4: "Duplex", 5: "Apt 5+ Storeys", 6: "Apt <5 Storeys",
        7: "Other single-attached", 8: "Movable"
    }

    dist_hist = df_hist['DTYPE'].value_counts(normalize=True).sort_index() * 100
    dist_fut = df_future['DTYPE'].value_counts(normalize=True).sort_index() * 100

    df_comp = pd.DataFrame({
        'Historic': dist_hist,
        'Forecast': dist_fut
    }).fillna(0)
    df_comp.index = [dtype_labels.get(i, f"Code {i}") for i in df_comp.index]

    log("\n   --- Distribution Comparison (%) ---")
    log(df_comp.round(2).to_string())

    # Save Report
    with open(report_path, "w") as f:
        f.write("\n".join(report_buffer))
#HOUSEHOLD AGGREGATION -------------------------------------------------------------------------------------------------
class HouseholdAggregator:
    """
    Transforms individual episode lists into aggregated Household Profiles.
    Resolution: 5 Minutes (288 slots per 24 hours).

    Step A: Grid Construction (Individual)
    Step B: Binary Presence (Household) -> 'occPre'
    Step C: Social Density (Household)    -> 'occDensity'
    Step D: Activity Sets (Household)     -> 'occActivity'
    """

    def __init__(self, resolution_min=5):
        self.res = resolution_min
        self.slots = int(1440 / self.res)  # 288 slots for 24h

        # Social columns to sum for Step C (excluding 'Alone')
        self.social_cols = [
            'Spouse', 'Children', 'parents', 'friends',
            'otherHHs', 'others', 'otherInFAMs'
        ]

    def process_all(self, df_expanded):
        """
        Main driver function.
        Groups data by Household and Day Type, aggregates,
        and then merges aggregation back to individual grids.
        Includes ALL static columns from the input CSV (Demographics, etc.).
        """
        print(f"   Grouping data by Household and Day Type...")

        # Columns that change per episode and shouldn't be broadcasted statically
        time_varying_cols = [
            'start', 'end', 'EPINO', 'occACT', 'occPRE', 'social_sum',
            'Spouse', 'Children', 'parents', 'friends',
            'otherHHs', 'others', 'otherInFAMs'
        ]

        # Group by Household AND Day
        groups = df_expanded.groupby(['SIM_HH_ID', 'Day_Type'])

        full_data_results = []

        # Iterate through each household scenario
        for (hh_id, day_type), group_df in tqdm(groups, desc="Processing Households"):

            # 1. Map AgentID -> Grid DataFrame
            people_grids_map = {}
            # 2. Map AgentID -> Static Metadata (Series)
            people_meta_map = {}

            # FIX 1: Group by 'AgentID' (Unique Index) instead of 'occID'
            # This ensures distinct people with the same GSS ID are treated separately
            if 'AgentID' not in group_df.columns:
                raise ValueError(
                    "❌ Error: 'AgentID' column missing. Please re-run Step 2 (Expansion) with the updated script.")

            for agent_id, person_data in group_df.groupby('AgentID'):
                # Step A: Create 5-min grid for this person
                grid = self._create_individual_grid(person_data)
                people_grids_map[agent_id] = grid

                # Capture Static Metadata (Take 1st row, drop time-varying)
                meta = person_data.iloc[0].drop(labels=time_varying_cols, errors='ignore')
                people_meta_map[agent_id] = meta

            # 3. Steps B, C, D: Aggregate the household
            hh_profile = self._aggregate_household(list(people_grids_map.values()))

            # 4. INTEGRATION: Merge Household Data + Individual Grid + Static Metadata
            for agent_id, p_grid in people_grids_map.items():
                # a. Concatenate Household Profile + Individual Grid
                combined = pd.concat([hh_profile, p_grid], axis=1)

                # b. Add Static Metadata
                meta = people_meta_map[agent_id]
                for col_name, val in meta.items():
                    combined[col_name] = val

                # Ensure essential keys are correct
                combined['SIM_HH_ID'] = hh_id
                combined['Day_Type'] = day_type
                combined['AgentID'] = agent_id  # Persist Unique ID

                full_data_results.append(combined)

        # Combine all individuals into one big dataframe
        return pd.concat(full_data_results, ignore_index=True)

    def _create_individual_grid(self, episodes):
        """
        Step A: 5-Minute Grid Construction (Standardization)
        Converts variable start/end times into a fixed length array (288 slots).
        """
        # Initialize blank arrays
        loc_grid = np.zeros(self.slots, dtype=int)
        act_grid = np.zeros(self.slots, dtype=int)
        dens_grid = np.zeros(self.slots, dtype=int)

        # --- FIX 2: Density Logic (Ghost Density Fix) ---
        valid_social = [c for c in self.social_cols if c in episodes.columns]

        # Convert 1=Yes, 2=No, 9=Unknown to Binary (1=Yes, 0=Else)
        episodes_social = episodes[valid_social].replace({1: 1, 2: 0, 9: 0}).fillna(0)

        # MASK: Only count social density if occPRE == 1 (Home)
        # If occPRE is NOT 1 (e.g. Work/Travel), density becomes 0
        is_home = (episodes['occPRE'] == 1).astype(int)

        # Assign to copy to avoid warnings
        episodes = episodes.copy()
        episodes['social_sum'] = episodes_social.sum(axis=1) * is_home

        # Fill the grid based on episodes
        for _, row in episodes.iterrows():
            # Convert HHMM format to total minutes (e.g., 1030 -> 10*60 + 30 = 630)
            s_raw = int(row['start'])
            s_min = (s_raw // 100) * 60 + (s_raw % 100)
            e_raw = int(row['end'])
            e_min = (e_raw // 100) * 60 + (e_raw % 100)
            # Convert minutes to slot index
            s_idx = int(np.floor(s_min / self.res))
            e_idx = int(np.floor(e_min / self.res))

            s_idx = max(0, min(s_idx, self.slots - 1))
            e_idx = max(0, min(e_idx, self.slots))

            # Fill range
            if e_idx > s_idx:
                # Normal Case
                loc_grid[s_idx:e_idx] = row['occPRE']
                act_grid[s_idx:e_idx] = row['occACT']
                dens_grid[s_idx:e_idx] = row['social_sum']
            elif e_idx < s_idx:
                # WRAPPED EPISODE (Crosses Midnight)
                # 1. Fill from Start -> End of Day
                loc_grid[s_idx:] = row['occPRE']
                act_grid[s_idx:] = row['occACT']
                dens_grid[s_idx:] = row['social_sum']
                # 2. Fill from Start of Day -> End
                loc_grid[:e_idx] = row['occPRE']
                act_grid[:e_idx] = row['occACT']
                dens_grid[:e_idx] = row['social_sum']

        # Return dataframe for this individual
        return pd.DataFrame({
            'ind_occPRE': loc_grid,
            'ind_occACT': act_grid,
            'ind_density': dens_grid
        })

    def _aggregate_household(self, people_grids):
        """
        Executes Steps B, C, and D combining multiple individual grids.
        """
        # Create Time Index (00:00, 00:05, ... 23:55)
        time_slots = pd.date_range("00:00", "23:55", freq=f"{self.res}min").strftime('%H:%M')

        # Dataframe to store final household results
        hh_df = pd.DataFrame({'Time_Slot': time_slots})

        if not people_grids:
            hh_df['occPre'] = 0
            hh_df['occDensity'] = 0
            hh_df['occActivity'] = ""
            return hh_df

        # --- STEP B: Aggregated Presence (Binary) -> occPre ---
        # 1. Stack location arrays (using 'ind_occPRE')
        loc_stack = np.vstack([p['ind_occPRE'].values for p in people_grids])

        # 2. Convert to Binary Presence (1=Home, 0=Outside)
        presence_binary = (loc_stack == 1).astype(int)

        # 3. Sum vertically (How many people home?)
        occupancy_count = presence_binary.sum(axis=0)

        # 4. Household Binary Status (1 if anyone is home, else 0)
        hh_df['occPre'] = (occupancy_count >= 1).astype(int)

        # --- STEP C: Social Density -> occDensity ---
        # 1. Stack density arrays (using 'ind_density')
        dens_stack = np.vstack([p['ind_density'].values for p in people_grids])

        # 2. Sum vertically
        hh_df['occDensity'] = dens_stack.sum(axis=0)

        # --- STEP D: Aggregated Activity Sets -> occActivity ---
        # 1. Stack activity arrays (using 'ind_occACT')
        act_stack = np.vstack([p['ind_occACT'].values for p in people_grids])

        activity_sets = []

        # Iterate through each time slot (column)
        for t in range(self.slots):
            # Get activities and presence for this moment
            acts_at_t = act_stack[:, t]
            pres_at_t = presence_binary[:, t]  # Only consider people AT HOME

            # Filter: Keep activities only for people who are PRESENT (1)
            valid_acts = acts_at_t[pres_at_t == 1]

            # Get Unique, Sort, Convert to String
            if len(valid_acts) > 0:
                unique_acts = sorted(np.unique(valid_acts))
                # Remove 0 or NaNs if any slipped in
                unique_acts = [str(a) for a in unique_acts if a > 0]
                act_str = ",".join(unique_acts)
            else:
                act_str = "0"  # "0" indicates Unoccupied/No Activity

            activity_sets.append(act_str)

        hh_df['occActivity'] = activity_sets

        return hh_df
#VALIDATION OF AGGREGATION ---------------------------------------------------------------------------------------------
def validate_household_aggregation(df_full, report_path=None):
    """
    Performs logical checks on the aggregated data and saves report to txt.
    """
    # Buffer to hold log messages
    logs = []

    def log(message):
        print(message)
        logs.append(str(message))

    log(f"\n{'=' * 60}")
    log(f"🔎 VALIDATING HOUSEHOLD AGGREGATION")
    log(f"{'=' * 60}")

    # --- CHECK 1: COMPLETENESS ---
    log(f"\n1. CHECKING TIME GRID COMPLETENESS...")
    if 'AgentID' not in df_full.columns:
        log("   ❌ Error: 'AgentID' column missing. Cannot validate completeness.")
        return False

    counts = df_full.groupby(['AgentID', 'Day_Type']).size()

    if (counts == 288).all():
        log(f"   ✅ Success: All {len(counts)} person-days have exactly 288 time slots.")
    else:
        errors = counts[counts != 288]
        log(f"   ❌ Error: Found {len(errors)} incomplete profiles.")
        log(errors.head())

    # --- CHECK 2: LOGIC (Presence vs. Density) ---
    log(f"\n2. CHECKING LOGIC (Presence vs. Density)...")
    empty_house = df_full[df_full['occPre'] == 0]
    ghosts = empty_house[empty_house['occDensity'] > 0]

    if len(ghosts) == 0:
        log(f"   ✅ Success: No social density detected in empty houses.")
    else:
        log(f"   ❌ Error: Found {len(ghosts)} rows where House is Empty but Density > 0.")

    # --- CHECK 3: ACTIVITY CONSISTENCY ---
    log(f"\n3. CHECKING ACTIVITY STRINGS...")
    if 'occActivity' in empty_house.columns:
        ghost_activities = empty_house[empty_house['occActivity'].astype(str) != "0"]
        if len(ghost_activities) == 0:
            log(f"   ✅ Success: Activity is correctly marked '0' when empty.")
        else:
            log(f"   ❌ Error: Found {len(ghost_activities)} rows with activities in empty house.")

    # --- SAVE REPORT TO FILE ---
    if report_path:
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(logs))
                f.write("\n")  # Add newline at end
            # We don't print "Saved" here to avoid cluttering the console output if it's called often
        except Exception as e:
            print(f"   ❌ Error writing report file: {e}")

    return True
def visualize_multiple_households(df_full, n_samples=10, output_img_path=None, report_path=None):
    """
    Generates a Grid Plot for 'n_samples' random households.
    Optionally appends status to the report file.
    """
    if output_img_path is None:
        output_img_path = Path("Validation_Plot_Batch.png")

    msg_start = f"\n4. GENERATING VISUAL VERIFICATION PLOT ({n_samples} Households)..."
    print(msg_start)

    if report_path:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(msg_start + "\n")

    # 1. Filter for households with some activity (Density > 1)
    interesting_ids = df_full[df_full['occDensity'] > 1]['SIM_HH_ID'].unique()

    if len(interesting_ids) == 0:
        print("   ⚠️ No high-density households found. Sampling random ones.")
        interesting_ids = df_full['SIM_HH_ID'].unique()

    # 2. Random Sample
    actual_n = min(n_samples, len(interesting_ids))
    sample_ids = np.random.choice(interesting_ids, actual_n, replace=False)

    # 3. Setup Grid
    cols = 4
    rows = math.ceil(actual_n / cols)
    figsize_height = rows * 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, figsize_height), sharex=False)
    axes = axes.flatten()

    # 4. Plot Loop
    for i, ax in enumerate(axes):
        if i < actual_n:
            hh_id = sample_ids[i]

            # Get Data (Priority: Weekday -> Weekend)
            mask = (df_full['SIM_HH_ID'] == hh_id) & (df_full['Day_Type'] == 'Weekday')
            df_hh = df_full[mask].copy()

            if df_hh.empty:
                mask = (df_full['SIM_HH_ID'] == hh_id) & (df_full['Day_Type'] == 'Weekend')
                df_hh = df_full[mask].copy()

            df_plot = df_hh[['Time_Slot', 'occPre', 'occDensity']].drop_duplicates()
            x = range(len(df_plot))

            if df_plot.empty:
                ax.text(0.5, 0.5, "No Data", ha='center')
                continue

            # Plot
            ax.fill_between(x, df_plot['occPre'], step="pre", color='green', alpha=0.3, label='Occupied')
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])
            ax.set_ylabel("Presence", fontsize=8, color='green')

            ax2 = ax.twinx()
            ax2.plot(x, df_plot['occDensity'], color='blue', linewidth=1.5, label='Density')
            ax2.set_ylabel("Density", fontsize=8, color='blue')
            ax2.tick_params(axis='y', labelsize=8)

            ax.set_title(f"Household #{hh_id}", fontsize=10, fontweight='bold', pad=3)

            ticks = np.arange(0, 288, 48)
            labels = [df_plot['Time_Slot'].iloc[j] for j in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
            ax.grid(True, alpha=0.2)

            if i == 0:
                lines, lbls = ax.get_legend_handles_labels()
                lines2, lbls2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, lbls + lbls2, loc='upper left', fontsize=8)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_img_path)

    msg_end = f"   ✅ Batch Plot saved to: {output_img_path.name}"
    print(msg_end)

    if report_path:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(msg_end + "\n")
#OCC to BEM input ------------------------------------------------------------------------------------------------------
class BEMConverter:
    """
    Converts 5-minute ABM profiles into Hourly BEM Schedules.
    Output format: 60-minute resolution, fractional occupancy (0-1), metabolic rate (W).
    Includes Residential variables (DTYPE, BEDRM, PR, etc.) for building matching.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir

        # 2024 Compendium of Physical Activities Mapping (Activity Code -> Watts)
        # Assumes 1 MET ~= 70 Watts (Avg adult 70kg)
        self.metabolic_map = {
            '1': 125,  # Work & Related (~1.8 MET - Standing/Office)
            '2': 175,  # Household Work (~2.5 MET - Cleaning/Cooking)
            '3': 190,  # Caregiving (~2.7 MET - Active child/elder care)
            '4': 195,  # Shopping (~2.8 MET - Walking with cart)
            '5': 70,  # Sleep (~1.0 MET - Sleeping/Lying quietly)
            '6': 105,  # Eating (~1.5 MET - Sitting eating)
            '7': 170,  # Personal Care (~2.4 MET - Dressing/Showering)
            '8': 110,  # Education (~1.6 MET - Sitting in class/Studying)
            '9': 90,  # Socializing (~1.3 MET - Sitting talking)
            '10': 85,  # Passive Leisure (~1.2 MET - TV/Reading + fidgeting)
            '11': 245,  # Active Leisure (~3.5 MET - Walking/Exercise)
            '12': 105,  # Volunteer (~1.5 MET - Light effort)
            '13': 140,  # Travel (~2.0 MET - Driving/Walking mix)
            '14': 135,  # Miscellaneous (~1.9 MET - Standing/Misc tasks)
            '0': 0  # Empty
        }

        # DTYPE Mapping (Code -> Description)
        self.dtype_map = {
            '1': "SingleD", # Detached
            '2': "SemiD",
            '3': "Attached",
            '4': "DuplexD",
            '5': "HighRise",
            '6': "MidRise",
            '7': "OtherA", # Attached
            '8': "Movable",
            # Fallbacks
            'Apartment': "Apt (Unspec.)",
            'Other dwelling': "Other"
        }

        # PR (Region) Mapping (ID -> Description)
        self.pr_map = {
            1: "Eastern Canada",
            2: "Quebec",
            3: "Ontario",
            4: "Prairies",
            5: "British Columbia",
            6: "Northern Canada",
            99: "Others"
        }

    def process_households(self, df_full):
        print(f"\n🚀 Starting BEM Conversion (Hourly Resampling)...")

        # 1. Prepare Time Index
        # We need a dummy date to enable resampling
        df_full['datetime'] = pd.to_datetime(df_full['Time_Slot'], format='%H:%M')

        # 2. Map Activities to Watts (Vectorized)
        print("   Mapping metabolic rates...")
        df_full['watts_5min'] = df_full['occActivity'].apply(self._calculate_watts)

        # 3. Group by Household & DayType
        groups = df_full.groupby(['SIM_HH_ID', 'Day_Type'])

        bem_schedules = []

        # List of residential variables to carry over (Added PR)
        target_res_cols = ['DTYPE', 'BEDRM', 'CONDO', 'ROOM', 'REPAIR', 'PR']

        for (hh_id, day_type), group in tqdm(groups, desc="Generating Schedules"):
            # Get Static Attributes (First row of the group)
            hh_size = group['HHSIZE'].iloc[0]

            # Extract residential vars safely (handle if missing)
            res_data = {}
            for col in target_res_cols:
                val = group[col].iloc[0] if col in group.columns else "Unknown"

                # Apply Mappings
                if col == 'DTYPE':
                    val_str = str(int(val)) if pd.notnull(val) and val != "Unknown" else str(val)
                    res_data[col] = self.dtype_map.get(val_str, val)
                elif col == 'PR':
                    # Map ID -> String Name directly (already encoded 1-6 or 99)
                    try:
                        region_id = int(float(val))
                    except (ValueError, TypeError):
                        region_id = 99
                    res_data[col] = self.pr_map.get(region_id, "Others")
                else:
                    res_data[col] = val

            # --- HOURLY RESAMPLING ---
            # Set index to datetime for resampling
            g_indexed = group.set_index('datetime')

            # Resample 5min -> 60min (Mean)
            hourly = g_indexed.resample('60min').agg({
                'occPre': 'mean',  # Fraction of hour home (0.0 - 1.0)
                'occDensity': 'mean',  # Avg social density
                'watts_5min': 'mean'  # Avg metabolic rate
            }).reset_index()

            # --- BEM FORMULAS ---

            # 1. Reconstruct People Count: (1 person + Social Density) * Presence Fraction
            estimated_count = hourly['occPre'] * (hourly['occDensity'] + 1)

            # 2. Normalize to Schedule (0-1) by dividing by HH Capacity
            occupancy_sched = (estimated_count / hh_size).clip(upper=1.0)

            # 3. Create Result DataFrame
            # Construct the dictionary with all columns
            data_dict = {
                'SIM_HH_ID': hh_id,
                'Day_Type': day_type,
                'Hour': hourly['datetime'].dt.hour,
                'HHSIZE': hh_size,
                # Unpack the residential variables here
                **res_data,
                'Occupancy_Schedule': occupancy_sched.round(3),  # 0 to 1
                'Metabolic_Rate': hourly['watts_5min'].round(1)  # Watts
            }

            hourly_df = pd.DataFrame(data_dict)
            bem_schedules.append(hourly_df)

        # Combine
        return pd.concat(bem_schedules, ignore_index=True)

    def _calculate_watts(self, act_str):
        """
        Parses activity string '1,5' -> maps to Watts -> returns average.
        """
        if str(act_str) == "0": return 0

        codes = str(act_str).split(',')
        watts = [self.metabolic_map.get(c.strip(), 100) for c in codes]  # Default 100W if unknown
        return sum(watts) / len(watts)
def visualize_bem_distributions(df_bem, output_dir=None):
    """
    Generates two validation plot files:
    1. 'BEM_Schedules_2025_temporals.png': Population Distributions & Sample Household.
    2. 'BEM_Schedules_2025_non_temporals.png': Residential stats (DTYPE, BEDRM, ROOM, PR).
    """
    print(f"\n📊 GENERATING BEM DISTRIBUTION PLOTS...")

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    path_temporal = output_dir / "BEM_Schedules_2025_temporals.png"
    path_nontemporal = output_dir / "BEM_Schedules_2025_non_temporals.png"

    # Set style
    sns.set_theme(style="whitegrid")

    # =========================================================
    # 1. TEMPORAL PLOTS (3x2 Grid)
    # =========================================================
    fig1, axes1 = plt.subplots(3, 2, figsize=(16, 15))

    # --- ROW 1: HISTOGRAMS ---
    sns.histplot(
        data=df_bem, x='Occupancy_Schedule', bins=20, kde=False,
        color='green', alpha=0.6, ax=axes1[0, 0]
    )
    axes1[0, 0].set_title("Population Distribution: Occupancy Fractions")
    axes1[0, 0].set_xlabel("Occupancy (0=Empty, 1=Full)")

    active_watts = df_bem[df_bem['Metabolic_Rate'] > 0]
    sns.histplot(
        data=active_watts, x='Metabolic_Rate', bins=30, kde=True,
        color='orange', alpha=0.6, ax=axes1[0, 1]
    )
    axes1[0, 1].set_title("Population Distribution: Metabolic Rates (Occupied)")
    axes1[0, 1].set_xlabel("Watts per Person")

    # --- ROW 2: AVERAGE PROFILES ---
    sns.lineplot(
        data=df_bem, x='Hour', y='Occupancy_Schedule', hue='Day_Type',
        estimator='mean', errorbar=('sd', 1),
        palette={'Weekday': 'green', 'Weekend': 'teal'}, ax=axes1[1, 0]
    )
    axes1[1, 0].set_title("Population Trend: Average Presence Schedule")
    axes1[1, 0].set_ylim(0, 1.05)
    axes1[1, 0].set_xticks(range(0, 25, 4))

    sns.lineplot(
        data=active_watts, x='Hour', y='Metabolic_Rate', hue='Day_Type',
        estimator='mean', errorbar=None,
        palette={'Weekday': 'orange', 'Weekend': 'red'}, ax=axes1[1, 1]
    )
    axes1[1, 1].set_title("Population Trend: Average Metabolic Intensity (Heat Output)")
    axes1[1, 1].set_xticks(range(0, 25, 4))

    # --- ROW 3: SAMPLE HOUSEHOLD ---
    occupancy_check = df_bem.groupby('SIM_HH_ID')['Occupancy_Schedule'].max()
    valid_ids = occupancy_check[occupancy_check > 0].index

    if len(valid_ids) > 0:
        sample_id = np.random.choice(valid_ids)
        sample_data = df_bem[df_bem['SIM_HH_ID'] == sample_id]
        wd_data = sample_data[sample_data['Day_Type'] == 'Weekday'].sort_values('Hour')
        we_data = sample_data[sample_data['Day_Type'] == 'Weekend'].sort_values('Hour')

        def plot_dual_axis(ax, data, title):
            if data.empty: return
            x = data['Hour']
            ax.fill_between(x, data['Occupancy_Schedule'], color='green', alpha=0.3, label='Occupancy')
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Occupancy Fraction", color='green', fontsize=10)
            ax2 = ax.twinx()
            ax2.plot(x, data['Metabolic_Rate'], color='darkorange', linewidth=2.5, label='Heat Gain')
            ax2.set_ylabel("Metabolic Rate (W)", color='darkorange', fontsize=10)
            ax2.set_ylim(0, 250)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(range(0, 25, 4))
            ax.set_xlabel("Hour of Day")

        plot_dual_axis(axes1[2, 0], wd_data, f"Sample Household #{sample_id}: Weekday Schedule")
        plot_dual_axis(axes1[2, 1], we_data, f"Sample Household #{sample_id}: Weekend Schedule")
    else:
        axes1[2, 0].text(0.5, 0.5, "No Valid Samples Found", ha='center')
        axes1[2, 1].axis('off')

    plt.tight_layout()
    fig1.savefig(path_temporal)
    plt.close(fig1)
    print(f"   ✅ Temporal Plot saved: {path_temporal.name}")

    # =========================================================
    # 2. NON-TEMPORAL PLOTS (Updated to Include PR)
    # =========================================================
    cols_static = [c for c in ['SIM_HH_ID', 'DTYPE', 'BEDRM', 'ROOM', 'PR'] if c in df_bem.columns]
    df_static = df_bem[cols_static].drop_duplicates(subset=['SIM_HH_ID'])

    if len(df_static) > 0:
        # Increased grid size for PR (1 row, 4 cols)
        fig2, axes2 = plt.subplots(1, 4, figsize=(22, 6))

        # DTYPE
        if 'DTYPE' in df_static.columns:
            sns.countplot(data=df_static, x='DTYPE', hue='DTYPE', palette='viridis', ax=axes2[0], legend=False)
            axes2[0].set_title("Dwelling Types")
            axes2[0].tick_params(axis='x', rotation=15, labelsize=8)
            axes2[0].set_ylabel("Count")

        # BEDRM
        if 'BEDRM' in df_static.columns:
            sns.countplot(data=df_static, x='BEDRM', hue='BEDRM', palette='magma', ax=axes2[1], legend=False)
            axes2[1].set_title("Bedroom Counts")
            axes2[1].set_ylabel("Count") 

        # ROOM
        if 'ROOM' in df_static.columns:
            sns.histplot(data=df_static, x='ROOM', discrete=True, color='purple', alpha=0.7, ax=axes2[2])
            axes2[2].set_title("Room Counts")
            axes2[2].set_ylabel("Count")

        # PR (Region)
        if 'PR' in df_static.columns:
            # PR is now a string thanks to process_households, so no mapping needed
            sns.countplot(data=df_static, x='PR', hue='PR', palette='coolwarm', ax=axes2[3], legend=False)
            axes2[3].set_title("Region (PR)")
            axes2[3].tick_params(axis='x', rotation=15)
            axes2[3].set_ylabel("Count")

        plt.tight_layout()
        fig2.savefig(path_nontemporal)
        plt.close(fig2)
        print(f"   ✅ Non-Temporal Plot saved: {path_nontemporal.name}")
    else:
        print("   ⚠️ Skipped Non-Temporal plots (Columns missing).")
if __name__ == '__main__':
    #DIRECTORIES -------------------------------------------------------------------------------------------------------
    # BASE_DIR = pathlib.Path("C:/Users/o_iseri/Desktop/2ndJournal")
    BASE_DIR = pathlib.Path("/Users/orcunkoraliseri/Desktop/Postdoc/eSim/Occupancy")

    DATA_DIR = BASE_DIR / "DataSources_CENSUS"
    OUTPUT_DIR = BASE_DIR / "Outputs_CENSUS"
    OUTPUT_DIR_ALIGNED = BASE_DIR / "Outputs_Aligned"
    OUTPUT_DIR_ALIGNED.mkdir(parents=True, exist_ok=True)

    # --- NEW: Define a directory to save your trained models ---
    MODEL_DIR = BASE_DIR / "saved_models_cvae"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # This creates the folder

    # --- Raw Files ---
    cen06_filtered = OUTPUT_DIR / "cen06_filtered.csv"
    cen11_filtered = OUTPUT_DIR / "cen11_filtered.csv"

    # --- Edited Files ---
    cen06_filtered2 = OUTPUT_DIR / "cen06_filtered2.csv"
    cen11_filtered2 = OUTPUT_DIR / "cen11_filtered2.csv"
    cen16_filtered2 = OUTPUT_DIR / "cen16_filtered2.csv"
    cen21_filtered2 = OUTPUT_DIR / "cen21_filtered2.csv"

    # --- Forecasted Files ---
    cen25 = OUTPUT_DIR / "Generated/forecasted_population_2025.csv"
    cen30 = OUTPUT_DIR / "Generated/forecasted_population_2030.csv"

    # --- Aligned Files ---
    aligned_CENSUS = OUTPUT_DIR_ALIGNED / "Aligned_Census_2025.csv"
    aligned_GSS = OUTPUT_DIR_ALIGNED / "Aligned_GSS_2022.csv"

    # VALIDATION
    VALIDATION_FORECAST_DIR = OUTPUT_DIR / "Validation_Forecasting_VisualbyColumn"
    VALIDATION_FORECASTVIS_DIR = OUTPUT_DIR / "Validation_Forecasting_Visual"
    VALIDATION_PR_MATCH_DIR = OUTPUT_DIR / "Validation_ProfileMatcher"
    VALIDATION_HH_AGG_DIR = OUTPUT_DIR / "Validation_HHaggregation"

    #TRAINING ----------------------------------------------------------------------------------------------------------
    """
    file_paths = {2006: cen06_filtered2, 2011: cen11_filtered2, 2016: cen16_filtered2, 2021: cen21_filtered2}
    processed_data, demo_cols, bldg_cols, data_scalers = prepare_data_for_generative_model(file_paths,sample_frac=1)
    encoder, decoder, cvae_model, training_history = train_cvae(df_processed=processed_data, demo_cols=demo_cols, bldg_cols=bldg_cols,
                                                                continuous_cols= ['EMPIN', 'TOTINC', 'INCTAX', 'VALUE'],
                                                                latent_dim=128,  epochs=100, batch_size=4096)
  
    # --- 3. THIS IS THE NEW PART: Save your models ---
    print("--- Training complete. Saving models to disk... ---")

    # Save the components to your new MODEL_DIR
    encoder.save(MODEL_DIR / 'cvae_encoder.keras')
    decoder.save(MODEL_DIR / 'cvae_decoder.keras')

    print("--- Models successfully saved! ---")

    print("\n--- C-VAE Training Complete ---")
    plot_training_history(training_history)

    check_reconstruction_quality(encoder, decoder, processed_data, demo_cols, bldg_cols)

    # --- B: Load Pre-Trained Models (Replaces training) ---
    print(f"--- Loading pre-trained models from: {MODEL_DIR} ---")
    """
    #TESTING -----------------------------------------------------------------------------------------------------------
    """
    file_paths = {2006: cen06_filtered2, 2011: cen11_filtered2, 2016: cen16_filtered2, 2021: cen21_filtered2}
    processed_data, demo_cols, bldg_cols, data_scalers = prepare_data_for_generative_model(file_paths,sample_frac=1)
    encoder = keras.models.load_model(MODEL_DIR / 'cvae_encoder.keras', custom_objects={'Sampling': Sampling})
    decoder = keras.models.load_model(MODEL_DIR / 'cvae_decoder.keras')
    print("--- Models loaded successfully! ---")
    validate_vae_reconstruction(encoder, decoder, processed_data, demo_cols, bldg_cols, continuous_cols=['EMPIN', 'TOTINC', 'INCTAX', 'VALUE'], n_samples=10, output_dir=OUTPUT_DIR)
    """
    #FORECASTING -------------------------------------------------------------------------------------------------------
    """
    file_paths = {2006: cen06_filtered2, 2011: cen11_filtered2, 2016: cen16_filtered2, 2021: cen21_filtered2}
    processed_data, demo_cols, bldg_cols, data_scalers = prepare_data_for_generative_model(file_paths, sample_frac=1)

    # Load Models
    # Ensure 'Sampling' class is available for custom_objects
    encoder = keras.models.load_model(MODEL_DIR / 'cvae_encoder.keras', custom_objects={'Sampling': Sampling})
    decoder = keras.models.load_model(MODEL_DIR / 'cvae_decoder.keras')

    # 3. Train Temporal Drift (Cluster-Based)
    print("\n=== Step 3: Modeling Temporal Drift ===")
    # UPDATED: Now returns the model + the full latent population + the last year
    temporal_model, last_population_z, last_year = train_temporal_model(encoder, processed_data, demo_cols, bldg_cols)

    # 4. Generate Forecasts
    TARGET_YEARS = [2025, 2030]
    N_SAMPLES = 2000
    OUTPUT_DIR = Path(OUTPUT_DIR) / "Generated"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in TARGET_YEARS:
        print(f"\n=== Step 4: Forecasting for {year} ===")

        # UPDATED: Pass last_population_z for resampling, and last_year for momentum calc
        gen_raw, bldg_raw, _ = generate_future_population(
            decoder,
            temporal_model,
            last_population_z,
            last_year,
            processed_data,
            bldg_cols,
            target_year=year,
            n_samples=N_SAMPLES,
            variance_factor=1.15
        )

        # UPDATED: Pass ref_df (processed_data) to enable Quantile Mapping for income tails
        df_forecast = post_process_generated_data(
            gen_raw,
            demo_cols,
            bldg_raw,
            bldg_cols,
            data_scalers,
            ref_df=processed_data
        )

        # Add Year
        df_forecast['YEAR'] = year

        # Save
        save_path = OUTPUT_DIR / f"forecasted_population_{year}.csv"
        df_forecast.to_csv(save_path, index=False)
        print(f"✅ Saved {year} forecast to: {save_path}")
        print(df_forecast.head())
    """
    #VALIDATION OF FORECASTING_VISUAL ----------------------------------------------------------------------------------
    """
    file_paths = {2006: cen06_filtered2, 2011: cen11_filtered2, 2016: cen16_filtered2, 2021: cen21_filtered2}
    processed_data, demo_cols, bldg_cols, data_scalers = prepare_data_for_generative_model(file_paths, sample_frac=1)
    encoder = keras.models.load_model(MODEL_DIR / 'cvae_encoder.keras', custom_objects={'Sampling': Sampling})
    decoder = keras.models.load_model(MODEL_DIR / 'cvae_decoder.keras')
    
    # Run Validation
    validate_forecast_trajectory(encoder, processed_data, demo_cols, bldg_cols, VALIDATION_FORECASTVIS_DIR)
    #validate_forecast_distributions(encoder=encoder, decoder=decoder, df_processed=processed_data, demo_cols=demo_cols,
    #                                bldg_cols=bldg_cols, scalers=data_scalers, output_dir=VALIDATION_FORECAST_DIR)
    """
    #ASSEMBLE HOUSEHOLD ------------------------------------------------------------------------------------------------
    """ 
    # --- Run Assembly for 2025 ---
    df_linked_2025 = assemble_households(cen25, target_year=2025, output_dir=OUTPUT_DIR)
    """
    #PROFILE MATCHER ---------------------------------------------------------------------------------------------------
    """
    IO_DIR = Path(OUTPUT_DIR)
    print("1. Loading Data...")
    df_census = pd.read_csv(aligned_CENSUS)
    df_gss = pd.read_csv(aligned_GSS, low_memory=False)

    # 2. Run Matching
    matcher = MatchProfiler(df_census, df_gss, dday_col="DDAY", id_col="occID")
    df_matched = matcher.run_matching()

    # 3. Save Matched Keys (Lightweight)
    df_matched.to_csv(OUTPUT_DIR_ALIGNED / "Matched_Population_Keys.csv", index=False)
    print(f"   Saved Keys: Matched_Population_Keys.csv")

    # 4. Expand & Save Full Schedules (Heavyweight)
    expander = ScheduleExpander(df_gss, id_col="occID")
    verify_sample(df_matched, expander)

    # Define output path for the massive file
    expanded_path = IO_DIR / "Full_Expanded_Schedules.csv"
    generate_full_expansion(df_matched, expander, expanded_path)

    print("\n✅ Workflow Complete.")
    """
    #VALIDATION: PROFILE MATCHER ---------------------------------------------------------------------------------------
    """
    IO_DIR_ALIGNED = Path(OUTPUT_DIR_ALIGNED)
    IO_DIR_VALID = Path(VALIDATION_PR_MATCH_DIR)
    df_matched = pd.read_csv(IO_DIR_ALIGNED / "Matched_Population_Keys.csv")
    df_gss = pd.read_csv(IO_DIR_ALIGNED / "Aligned_GSS_2022.csv", low_memory=False)

    # Re-initialize Expander
    # We need to import or paste the ScheduleExpander class here first!
    expander = ScheduleExpander(df_gss, id_col="occID")

    # Run Validation
    validate_matching_quality(df_matched, expander, save_path=(IO_DIR_VALID / "Validation_ProfileMatcher_2025.txt"))
    """
    #PROFILE MATCHER: POST-PROCESSING & VALIDATION OF POST-PROCESSING --------------------------------------------------
    """
    import pandas as pd
    from pathlib import Path
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    IO_DIR = Path(OUTPUT_DIR)
    # 1. The "Teacher": Historic Data (Must have detailed DTYPE 1-8)
    # Replace this with the actual path to your 2006 or 2011 raw data
    HISTORIC_DATA_PATHS = [cen06_filtered, cen11_filtered]
    # 2. The "Student": Step 2 Output (Has coarse DTYPE 1-3)
    INPUT_FORECAST_PATH = IO_DIR / "Full_Expanded_Schedules.csv"
    # 2. The Source of Truth (Matched Keys with original Census attributes)
    INPUT_KEYS_PATH = IO_DIR / "Matched_Population_Keys.csv"
    # 3. The Result: Input for Step 3
    OUTPUT_REFINED_PATH = IO_DIR / "Full_Expanded_Schedules_Refined.csv"
    VALIDATION_DIR = IO_DIR / "Validation_ProfileMatcher_PostProcessing"
    # =============================================================================
    # EXECUTION
    # =============================================================================
    print(f"\n🚀 Starting Step 2b: DTYPE Refinement (Merged Strategy)...")
    # 1. Load Forecast
    if not INPUT_FORECAST_PATH.exists():
        print(f"❌ Error: Forecast file not found at {INPUT_FORECAST_PATH}")
        exit()
    df_forecast = pd.read_csv(INPUT_FORECAST_PATH, low_memory=False)

    # 2. Merge Keys (The Upgrade)
    if INPUT_KEYS_PATH.exists():
        df_keys = pd.read_csv(INPUT_KEYS_PATH, low_memory=False)
        df_forecast = merge_keys_into_forecast(df_forecast, df_keys)
    else:
        print("⚠️ Keys file not found. Falling back to deriving CFSIZE/TOTINC.")

    # 3. Load Historic Data for Training
    print("Loading Historic Data...")
    historic_dfs = []
    for path in HISTORIC_DATA_PATHS:
        if path.exists():
            historic_dfs.append(pd.read_csv(path, low_memory=False))

    if historic_dfs:
        df_hist = pd.concat(historic_dfs, ignore_index=True)

        # 4. Train & Apply
        refiner = DTypeRefiner(IO_DIR)
        refiner.train_models(df_hist)

        df_refined = refiner.apply_refinement(df_forecast)

        df_refined.to_csv(OUTPUT_REFINED_PATH, index=False)
        print(f"✅ Saved Refined Data to: {OUTPUT_REFINED_PATH}")

        # 5. Validate
        validate_refinement_model(HISTORIC_DATA_PATHS, OUTPUT_REFINED_PATH, VALIDATION_DIR)
    else:
        print("❌ No historic data found for training.")
    """
    #HOUSEHOLD AGGREGATION ---------------------------------------------------------------------------------------------
    """
    IO_DIR = Path(OUTPUT_DIR)
    expanded_file = IO_DIR / "Full_Expanded_Schedules_Refined.csv"
    output_full = IO_DIR / "Full_data.csv"

    # 2. Load Data
    print("1. Loading Expanded Schedules...")
    if not expanded_file.exists():
        print(f"❌ Error: {expanded_file} not found. Run Step 2 first.")
    else:
        df_expanded = pd.read_csv(expanded_file, low_memory=False)

        # 3. Initialize Aggregator
        aggregator = HouseholdAggregator(resolution_min=5)

        # 4. Run Process
        print("2. Starting Process (Padding + Aggregation)...")
        # Now returns the full dataset with all original columns integrated
        df_final = aggregator.process_all(df_expanded)

        # 5. Save
        print(f"3. Saving Full Integrated Data to: {output_full.name}...")
        df_final.to_csv(output_full, index=False)

        # 6. Verification
        print("\n--- Verification: Columns in Output ---")
        print(f"Total Columns: {len(df_final.columns)}")
        print(f"Sample Columns: {list(df_final.columns[:10])} ... {list(df_final.columns[-3:])}")

        print("\n✅ Step 3 Complete. Full Integrated Data generated.")
    """
    #VALIDATION: HOUSEHOLD AGGREGATION ---------------------------------------------------------------------------------
    """
    IO_DIR = Path(OUTPUT_DIR)
    IO_VALID_HHagg_DIR = Path(VALIDATION_HH_AGG_DIR)
    full_data_path = IO_DIR / "Full_data.csv"
    plot_path = IO_VALID_HHagg_DIR / "Validation_Plot_Batch.png"
    report_path = IO_VALID_HHagg_DIR / "Validation_Report_HH.txt"  # New Output File

    if not full_data_path.exists():
        print("❌ Error: Full_data.csv not found.")
    else:
        print("Loading data for validation...")
        df_full = pd.read_csv(full_data_path, low_memory=False)

        # Run Checks (Writes 1-3 to file)
        validate_household_aggregation(df_full, report_path=report_path)

        # Run Visuals (Appends 4 to file)
        visualize_multiple_households(df_full, n_samples=16, output_img_path=plot_path, report_path=report_path)

        print(f"\n✅ Full Validation Report saved to: {report_path.name}")
    """
    #OCC to BEM input --------------------------------------------------------------------------------------------------
    """
    IO_DIR = Path(OUTPUT_DIR)
    full_data_path = IO_DIR / "Full_data.csv"
    output_path = IO_DIR / "BEM_Schedules_2025.csv"
    output_path_vis = IO_DIR

    if not full_data_path.exists():
        print("❌ Error: Full_data.csv not found.")
    else:
        print("1. Loading Household Data...")
        df_full = pd.read_csv(full_data_path, low_memory=False)

        # Initialize Converter
        converter = BEMConverter(output_dir=IO_DIR)

        # Run
        df_bem = converter.process_households(df_full)

        # Save
        # float_format='%.3f' ensures 0.333 is written as "0.333" not ".333"
        print(f"2. Saving Hourly BEM Input to: {output_path.name}")
        df_bem.to_csv(output_path, index=False, float_format='%.3f')

        # Verify
        print("\n--- Verification: Sample Household ---")

        # Force pandas to show 3 decimal places with leading zero
        pd.options.display.float_format = '{:.3f}'.format

        # Show relevant columns including new residential ones
        cols_to_show = ['SIM_HH_ID', 'Hour', 'DTYPE', 'BEDRM', "ROOM", "PR", 'Occupancy_Schedule', 'Metabolic_Rate']
        # Filter cols that actually exist in output
        valid_cols = [c for c in cols_to_show if c in df_bem.columns]

        print(df_bem[valid_cols].head(12).to_string(index=False))

        print("\n✅ Step 4 Complete. Ready for EnergyPlus/Honeybee.")
        visualize_bem_distributions(df_bem, output_dir=output_path_vis)

    #IO_DIR = Path(OUTPUT_DIR)
    #full_data_path = IO_DIR / "Full_data.csv"
    #df = pd.read_csv(full_data_path)
    #print(df.columns)
    """