import sys
import os
import logging
import time
import warnings
import json
import folium
from IPython.display import display
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

from yellowbrick.cluster import SilhouetteVisualizer
from llvmlite.ir.values import ReturnValue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from pandas.io.formats.format import return_docstring
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.linear_model import LinearRegression

try:
    import skfuzzy as fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: scikit-fuzzy not available. Fuzzy C-Means will be skipped.")
    FUZZY_AVAILABLE = False

# UMAP for visualization (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: umap-learn not available. Will use t-SNE for visualization.")
    UMAP_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.append(str(Path(__file__).parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT_DIR / "clustering_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class ClusteringService:
    
    def __init__(
        self,
        algorithm: str,
        scaled_df: pd.DataFrame,
        preprocessed_df: pd.DataFrame,
        X: pd.DataFrame,
        n_clusters: int
    ):
        self.algorithm = algorithm
        self.scaled_df = scaled_df
        self.preprocessed_df = preprocessed_df
        self.X = X
        self.n_clusters = n_clusters

    def run_algorithm(self):
        algorithm = self.algorithm
        X = self.X
        n_clusters = self.n_clusters
        
        OUTPUT_PATH = SAVE_DIR / f"{algorithm}" /f"k{n_clusters}"
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        # 1. Run Clustering
        if algorithm == "kmeans":
            print("running kmeans algorithm")
            results = self._run_kmeans()
        elif algorithm == "fcm":
            print("running fcm algorithm")
            results = self._run_fcm()
        elif algorithm == "spectral":
            print("running spectral algorithm")
            results = self._run_spectral()
        
        model = results.get("model")
        labels = results.get("labels")
        silhouette_avg = results.get("silhouette_score")
        dbi_score = results.get("dbi_score")
        

        # 2. Create Silhouette Visualization
        self._generate_silhouette_visualization(
            model, labels, silhouette_avg, OUTPUT_PATH
        )
        
        # 3. Create Scatter Plot
        self._generate_scatter_plot(labels, OUTPUT_PATH)
            
        # 4. Create Box Plot For Price Distribution
        cluster_map = pd.DataFrame({
            "City": self.scaled_df["City"],
            "Cluster": labels
        })
        merged_df = self.preprocessed_df.merge(cluster_map, on="City", how="left")
        self._generate_box_plot(merged_df, OUTPUT_PATH)
        
        # 5. Create Line Chart
        self._generate_line_chart(merged_df, OUTPUT_PATH)
        
        # 6. Create Plot Volatility Trends
        df_prepared = self._load_and_prepare_data(cluster_map)
        df_volatility = self._calculate_monthly_volatility(df_prepared)
        self._plot_volatility_trends(df_volatility, OUTPUT_PATH)
        
        # 7. Create Radar Chart
        df_trends = self._calculate_linear_trends(df_prepared)
        df_cluster_metrics = self._calculate_cluster_metrics(df_prepared, df_volatility, df_trends)
        df_radar_data = self._normalize_metrics_for_radar(df_cluster_metrics)
        self._plot_radar_chart(df_radar_data, OUTPUT_PATH)
        
        # 8. Create Map Visualization
        df = self.scaled_df.copy()

        labeled_df = df.merge(cluster_map, on="City", how="left")

        columns_to_drop = [c for c in labeled_df.columns if c not in ["City", "Cluster"]]

        labeled_df.drop(columns=columns_to_drop, axis=1, inplace=True)
        
        coordinates_file = Path("data/city_coordinates.json") 

        df_for_map = self._prepare_data_for_map(
            # price_data_filepath='food_prices_consolidated.csv',
            cluster_data=labeled_df, # Use your actual clustered DataFrame here
            coordinates_filepath=coordinates_file
        )
        
        cluster_map_object = self._create_cluster_map(df_for_map, OUTPUT_PATH)
        
        return

    def _run_kmeans(self):
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=1e-4
        )
        
        labels = kmeans.fit_predict(self.X)
        silhouette_avg = silhouette_score(self.X, labels)
        dbi_score = davies_bouldin_score(self.X, labels)
        
        return {
            "model": kmeans,
            "labels": labels,
            "silhouette_score": silhouette_avg,
            "dbi_score": dbi_score
        }
        
        return
    
    def _run_fcm(self):
        
        # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        #     X.T,  # Transpose for skfuzzy format
        #     k,
        #     config.fcm_params['m'],
        #     error=config.fcm_params['error'],
        #     maxiter=config.fcm_params['maxiter'],
        #     init=None,
        #     seed=config.random_state
        # )
        
        return

    def _run_spectral(self):
        
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            random_state=42,
            affinity="rbf",
            gamma=1.0,
            n_init=10
        )
        
        labels = spectral.fit_predict(self.X)
        silhouette_avg = silhouette_score(self.X, labels)
        dbi_score = davies_bouldin_score(self.X, labels)
        
        return {
            "model": spectral,
            "labels": labels,
            "silhouette_score": silhouette_avg,
            "dbi_score": dbi_score
        }

    def _generate_silhouette_visualization(
        self,
        model,
        labels,
        silhouette_avg,
        output_path
    ):
        if self.algorithm in ["kmeans", "fcm"]:
            visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
            visualizer.fit(self.X)
            visualizer.show(outpath=output_path / "silhouette_visualization.png")
            plt.close()
        else:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(10, 7)

            # The silhouette plot is only valid for n_clusters > 1
            n_clusters = len(np.unique(labels))

            # Set the y-axis limits
            ax.set_ylim([0, len(self.X) + (n_clusters + 1) * 10])

            # Compute the silhouette score for each sample
            sample_silhouette_values = silhouette_samples(self.X, labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # Choose a color for the cluster
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            # --- Plotting the average silhouette score line ---
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")

            # --- Final Plot Adjustments ---
            ax.set_title("Silhouette Plot for Spectral Clustering")
            ax.set_xlabel("Silhouette coefficient values")
            ax.set_ylabel("Cluster label")
            ax.set_yticks([])
            ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            plt.savefig(output_path / "silhouette_visualization.png")
            plt.close()
    
    def _generate_scatter_plot(self, labels, output_path):
        pca = PCA(n_components=10, random_state=42)
        X_pca = pca.fit_transform(self.X)

        # Step 2: UMAP untuk reduksi ke 2D
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
        X_umap = reducer.fit_transform(X_pca)

        import matplotlib.colors as mcolors

        n_clusters = len(np.unique(labels))
        colors = list(mcolors.TABLEAU_COLORS.values())[:n_clusters]

        for i, color in enumerate(colors):
            cluster_points = X_umap[labels == i]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                s=70, alpha=0.9, edgecolor='k',
                label=f'Cluster {i}', color=color
            )

        plt.title("2D Cluster Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.savefig(output_path / "scatter_plot.png")
        plt.close()
   
    def _generate_box_plot(self, merged_df, output_path):
        if 'Year' not in merged_df.columns:
            merged_df['Year'] = pd.to_datetime(merged_df['Date']).dt.year

        commodities = sorted(merged_df["Commodity"].unique())
        clusters = sorted(merged_df["Cluster"].unique())
        n_commodities = len(commodities)
        n_clusters = len(clusters)

        print(f"📊 Plotting {n_commodities} commodities across {n_clusters} clusters")
        print(f"Years in dataset: {sorted(merged_df['Year'].unique())}")

        n_cols = 2
        n_rows = int(np.ceil(n_commodities / n_cols))

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
        axes = axes.flatten() if n_commodities > 1 else [axes]

        palette = sns.color_palette("Set2", n_colors=n_clusters)
        cluster_colors = dict(zip(clusters, palette))

        for i, commodity in enumerate(commodities):
            ax = axes[i]
            
            df_commodity = merged_df[merged_df['Commodity'] == commodity]
            
            # Check if there's data
            if df_commodity.empty:
                ax.text(0.5, 0.5, 'No Data Available', 
                    ha='center', va='center', fontsize=12)
                ax.set_title(commodity, fontsize=14, weight='bold')
                continue
            
            # Create boxplot with consistent cluster ordering
            sns.boxplot(
                x='Year', 
                y='Price', 
                hue='Cluster',
                hue_order=clusters,
                data=df_commodity,
                ax=ax,
                palette=cluster_colors,
                fliersize=3,
                linewidth=1.3,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white', 
                            markeredgecolor='black', markersize=5)
            )
            
            ax.set_title(f"{commodity}", fontsize=14, weight='bold', pad=12)
            ax.set_xlabel("Year", fontsize=11, weight='semibold')
            ax.set_ylabel("Price (Rp)", fontsize=11, weight='semibold')
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # Remove individual legends (we'll add one global legend)
            if ax.get_legend():
                ax.get_legend().remove()
            
            # Rotate x-axis labels if there are many years
            if len(df_commodity['Year'].unique()) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Create a single legend for the entire figure
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            ncol = min(n_clusters, 6)
            fig.legend(
                handles, labels,
                title='City Cluster',
                loc='upper center',
                bbox_to_anchor=(0.5, 0.99),
                ncol=ncol,
                fontsize=11,
                title_fontsize=12,
                frameon=True,
                fancybox=True,
                shadow=True,
                edgecolor='gray'
            )

        # Add title
        year_min = merged_df['Year'].min()
        year_max = merged_df['Year'].max()
        fig.suptitle(
            f"Distribusi Harga per Komoditas berdasarkan Tahun dan Klaster ({year_min}–{year_max})", 
            fontsize=18, 
            weight='bold', 
            y=1.01
        )

        plt.tight_layout()

        # Save the figure
        plt.savefig(output_path / "commodity_price_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Print color mapping
        print("\n🎨 Cluster Color Legend:")
        print("-" * 40)
        for cluster in clusters:
            color = cluster_colors[cluster]
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            )
            print(f"■ {cluster}: {hex_color}")

        print(f"\n✓ Plot saved as 'commodity_price_boxplots.png'")
        print(f"✓ Total data points: {len(merged_df)}")
    
    def _generate_line_chart(self, merged_df, output_path):
        # --- 1. Load and Prepare Data ---
        try:
            # Ensure Date column is datetime
            if 'Date' not in merged_df.columns or merged_df['Date'].dtype != 'datetime64[ns]':
                merged_df['Date'] = pd.to_datetime(merged_df['Date'])
            
            print("=== Data Information ===")
            print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
            print(f"Years: {sorted(merged_df['Date'].dt.year.unique())}")
            print(f"Clusters: {sorted(merged_df['Cluster'].unique())}")
            print(f"Commodities: {sorted(merged_df['Commodity'].unique())}\n")

            # --- 2. Process Data for Plotting ---
            trend = (
                merged_df
                .groupby(["Date", "Cluster", "Commodity"], as_index=False)["Price"]
                .mean()
                .sort_values("Date")
            )
            
            # Ensure Date is datetime (redundant but safe)
            trend["Date"] = pd.to_datetime(trend["Date"])

            # --- 3. Setup for Visualization ---
            commodities = sorted(trend["Commodity"].unique())
            clusters = sorted(trend["Cluster"].unique())
            n_commodities = len(commodities)
            n_clusters = len(clusters)
            
            # Create consistent color palette
            palette = dict(zip(clusters, sns.color_palette("tab10", n_colors=n_clusters)))
            
            # Calculate date limits from actual data
            min_date_limit = trend['Date'].min()
            max_date_limit = trend['Date'].max()
            
            print(f"Plotting {n_commodities} commodities with {n_clusters} clusters")
            print(f"Plot date range: {min_date_limit.date()} to {max_date_limit.date()}\n")

            # --- 4. Create Subplots ---
            n_cols = 2
            n_rows = int(np.ceil(n_commodities / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, 
                figsize=(17, n_rows * 4.5), 
                constrained_layout=True
            )
            axes = axes.flatten() if n_commodities > 1 else [axes]

            for i, commodity in enumerate(commodities):
                ax = axes[i]
                df_c = trend[trend["Commodity"] == commodity]
                
                # Check for empty data
                if df_c.empty:
                    ax.text(0.5, 0.5, 'No Data Available', 
                        ha='center', va='center', fontsize=12)
                    ax.set_title(commodity, fontsize=15, weight='bold')
                    continue

                # Plot with consistent cluster ordering
                sns.lineplot(
                    data=df_c,
                    x="Date",
                    y="Price",
                    hue="Cluster",
                    hue_order=clusters,  # Ensure consistent order
                    ax=ax,
                    palette=palette,
                    linewidth=2,
                    alpha=0.85,
                    legend=False  # We'll add custom legend
                )

                ax.set_title(commodity, fontsize=15, weight='bold', pad=10)
                ax.set_ylabel("Average Price (Rp)", fontsize=11)
                ax.set_xlabel("")
                ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
                
                # Set x-axis to show only years within data range
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                
                # CRITICAL: Set x-axis limits to prevent empty years
                ax.set_xlim(min_date_limit, max_date_limit)
                
                # Style x-axis
                ax.tick_params(axis='x', rotation=0, labelsize=10)
                
                # Add custom legend to each subplot
                handles = [plt.Line2D([0], [0], color=palette[cluster], 
                                    linewidth=2, label=cluster) 
                        for cluster in clusters]
                ax.legend(
                    handles=handles,
                    title="Cluster",
                    loc='best',
                    fontsize=9,
                    title_fontsize=10,
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='gray',
                    fancybox=True
                )

            # --- 5. Clean Up ---
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            # Add main title
            year_min = min_date_limit.year
            year_max = max_date_limit.year
            fig.suptitle(
                f"Tren Rata-rata Harga Pangan per Klaster ({year_min}–{year_max})",
                fontsize=19,
                weight='bold',
                y=1.04
            )
            
            # Save and display
            plt.savefig(output_path / "final_food_price_trends.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Plot saved as 'final_food_price_trends.png'")
            print(f"✓ Successfully plotted {n_commodities} commodities")
            
            # Print color mapping
            print("\n🎨 Cluster Color Legend:")
            print("-" * 40)
            for cluster in clusters:
                color = palette[cluster]
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]*255), int(color[1]*255), int(color[2]*255)
                )
                print(f"■ {cluster}: {hex_color}")

        except KeyError as e:
            print(f"❌ Error: Missing column - {e}")
            print(f"Available columns: {merged_df.columns.tolist()}")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_and_prepare_data(self, cluster_map: pd.DataFrame) -> pd.DataFrame:
        """
        Loads the food price data, converts date columns, and prepares it for analysis.

        Args:
            filepath: The path to the CSV file.

        Returns:
            A pandas DataFrame with a proper 'Date' column and a 'YearMonth' period column.
        """
        df = self.preprocessed_df.copy()
        
        # --- Core Preparation Steps ---
        # Convert 'Date' column to datetime objects for time-series analysis
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create a 'YearMonth' column to enable grouping by month.
        # The 'to_period' function is perfect for this.
        df['YearMonth'] = df['Date'].dt.to_period('M')
        
        df = df.merge(cluster_map, on="City", how="left")
        
        print("Data loaded and prepared successfully.")
        return df
    
    def _calculate_monthly_volatility(self, df: pd.DataFrame, min_sample_size: int = 3) -> pd.DataFrame:
        """
        Calculates the monthly Coefficient of Variation (CV) for price volatility.

        Args:
            df: The preprocessed DataFrame from the previous step.
            min_sample_size: The minimum number of price recordings in a month
                            to calculate a meaningful CV.

        Returns:
            A DataFrame containing the monthly CV for each commodity and cluster.
        """
        if df.empty:
            print("Input DataFrame is empty. Cannot calculate volatility.")
            return pd.DataFrame()

        # --- Group by month, commodity, and cluster to perform calculations ---
        # We calculate mean, std deviation, and the count of data points for each group.
        monthly_stats = df.groupby(['YearMonth', 'Commodity', 'Cluster'])['Price'].agg(
            price_mean='mean',
            price_std='std',
            price_count='count'
        ).reset_index()

        # --- Filter out groups with insufficient data ---
        # Calculating CV on 1 or 2 data points is not meaningful.
        significant_data = monthly_stats[monthly_stats['price_count'] >= min_sample_size].copy()

        # --- Calculate the Coefficient of Variation (CV) ---
        # CV = (Standard Deviation / Mean) * 100
        # We handle cases where the mean is zero to avoid division errors.
        significant_data['CV'] = np.where(
            significant_data['price_mean'] > 0,
            (significant_data['price_std'] / significant_data['price_mean']) * 100,
            0
        )
        
        # Convert 'YearMonth' period back to a timestamp for plotting
        significant_data['Date'] = significant_data['YearMonth'].dt.to_timestamp()
        
        print("Monthly volatility calculated successfully.")
        return significant_data

    def _plot_volatility_trends(self, df_cv: pd.DataFrame, output_path):
        """
        Generates and saves a multi-subplot visualization of price volatility (CV).

        Args:
            df_cv: The DataFrame containing the calculated monthly CV.
        """
        if df_cv.empty:
            print("Volatility DataFrame is empty. Cannot generate plot.")
            return

        commodities = sorted(df_cv["Commodity"].unique())
        clusters = sorted(df_cv["Cluster"].unique())
        n_commodities = len(commodities)
        
        palette = dict(zip(clusters, sns.color_palette("tab10", n_colors=len(clusters))))

        n_cols = 2
        n_rows = int(np.ceil(n_commodities / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4.5), constrained_layout=True)
        axes = axes.flatten()

        min_date_limit = df_cv['Date'].min()
        max_date_limit = df_cv['Date'].max()

        for i, commodity in enumerate(commodities):
            ax = axes[i]
            df_c = df_cv[df_cv["Commodity"] == commodity]

            sns.lineplot(
                data=df_c,
                x="Date",
                y="CV",
                hue="Cluster",
                ax=ax,
                palette=palette,
                linewidth=1.5,
                alpha=0.9,
                legend=True
            )

            ax.set_title(f"Volatilitas {commodity}", fontsize=15, weight='bold')
            ax.set_ylabel("Coefficient of Variation (%)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(title="Cluster", loc='upper left', fontsize='small')
            
            # Format X-axis to show years
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlim(min_date_limit, max_date_limit)
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=0)

        # Clean up empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        min_year = min_date_limit.year
        max_year = max_date_limit.year
        fig.suptitle(f"Volatilitas Harga Bulanan (CV) per Klaster ({min_year}–{max_year})", fontsize=24, weight='bold', y=1.05)
        
        # Save the figure
        plt.savefig(output_path / "volatility_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Volatility plot has been generated and saved as 'volatility_trends.png'.")

    def _calculate_linear_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the linear regression trend for each commodity-cluster combination.

        Args:
            df: The preprocessed DataFrame with a 'Date' column.

        Returns:
            A DataFrame containing the slope and intercept for each trend line.
        """
        if df.empty:
            print("Input DataFrame is empty. Cannot calculate trends.")
            return pd.DataFrame()

        trend_results = []
        
        # Find the first date in the entire dataset to use as a reference point (day 0)
        start_date = df['Date'].min()

        # Convert date to a numerical format (days since start_date)
        df['DayNum'] = (df['Date'] - start_date).dt.days
        
        # Iterate over every unique group of commodity and cluster
        for (commodity, cluster), group_data in df.groupby(['Commodity', 'Cluster']):
            
            # Reshape data for scikit-learn
            X = group_data[['DayNum']] # Independent variable (time)
            y = group_data['Price']    # Dependent variable (price)
            
            # We need at least 2 points to fit a line
            if len(group_data) < 2:
                continue
                
            # Fit the linear regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Store the results
            trend_results.append({
                'Commodity': commodity,
                'Cluster': cluster,
                'slope': model.coef_[0],      # The trend (price change per day)
                'intercept': model.intercept_
            })

        print("Linear trend calculations complete.")
        return pd.DataFrame(trend_results)
    
    def _calculate_cluster_metrics(self, df_prep, df_vol, df_trn):
        """
        Calculates and aggregates the core metrics for each cluster.
        """
        # Metric 1: Price Level (Normalized within each commodity before averaging)
        df_prep['PriceNormalized'] = df_prep.groupby('Commodity')['Price'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        price_level = df_prep.groupby('Cluster')['PriceNormalized'].mean().reset_index(name='Price Level')

        # Metric 2: Volatility (Average CV)
        volatility = df_vol.groupby('Cluster')['CV'].mean().reset_index(name='Volatility')

        # Metric 3: Trend (Average slope)
        trend = df_trn.groupby('Cluster')['slope'].mean().reset_index(name='Trend')
        
        # Merge all metrics into a single DataFrame
        df_metrics = pd.merge(price_level, volatility, on='Cluster')
        df_metrics = pd.merge(df_metrics, trend, on='Cluster')
        
        return df_metrics.set_index('Cluster')
    
    def _normalize_metrics_for_radar(self, df_metrics):
        """
        Normalizes metrics to a 1-10 scale for plotting.
        """
        scaler = MinMaxScaler(feature_range=(1, 10))
        df_normalized = pd.DataFrame(scaler.fit_transform(df_metrics),
                                    columns=df_metrics.columns,
                                    index=df_metrics.index)
        return df_normalized
    
    def _plot_radar_chart(self, df_radar, output_path):
        """
        Generates a radar chart from the normalized cluster data.
        """
        labels = df_radar.columns
        num_vars = len(labels)
        
        # Create angles for the plot
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # Close the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Helper function to plot a single cluster's data
        def add_to_radar(cluster_name, color):
            values = df_radar.loc[cluster_name].tolist()
            values += values[:1] # Close the circle
            ax.plot(angles, values, color=color, linewidth=2.5, linestyle='solid', label=cluster_name)
            ax.fill(angles, values, color=color, alpha=0.25)
        
        # To keep the chart readable, we'll plot a subset of clusters.
        # Here, we select the first 5 clusters as a representative sample.
        clusters_to_plot = df_radar.index[:5]
        colors = plt.cm.get_cmap('tab10', len(clusters_to_plot))

        for i, cluster in enumerate(clusters_to_plot):
            add_to_radar(cluster, colors(i))

        # Formatting the chart
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=12)
        
        plt.title('Comparative Profile of City Clusters', size=20, color='black', y=1.1)
        plt.legend(title='City Clusters', loc='upper right', bbox_to_anchor=(1.4, 1.1))
        
        plt.savefig(output_path / "cluster_profile_radar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _prepare_data_for_map(self, cluster_data: pd.DataFrame, coordinates_filepath: str) -> pd.DataFrame:
        """
        Consolidates city, cluster, and coordinate data into a single DataFrame.

        Args:
            cluster_data: DataFrame from clustering containing 'City' and 'Cluster' columns.
            coordinates_filepath: Path to the user-provided JSON file with city coordinates.

        Returns:
            A DataFrame ready for plotting, with columns for City, Cluster, Lat, and Lon.
        """
        try:
            # --- 1. Load the coordinates mapping from your JSON file ---
            with open(coordinates_filepath, 'r') as f:
                city_coordinates = json.load(f)
            
            # Convert it into a DataFrame
            df_coords = pd.DataFrame(city_coordinates.items(), columns=['City', 'Coordinates'])
            df_coords[['Lat', 'Lon']] = pd.DataFrame(df_coords['Coordinates'].tolist(), index=df_coords.index)
            df_coords.drop('Coordinates', axis=1, inplace=True)
            
            # --- 2. Get the city-to-cluster mapping ---
            df_city_clusters = cluster_data[['City', 'Cluster']].drop_duplicates().reset_index(drop=True)

            # --- 3. Merge the DataFrames ---
            df_map_data = pd.merge(df_city_clusters, df_coords, on='City', how='inner')
            
            print("Data successfully prepared for mapping.")
            print(f"Found coordinates for {len(df_map_data)} cities.")
            
            return df_map_data

        except FileNotFoundError:
            print(f"Error: The file '{coordinates_filepath}' was not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred during data preparation: {e}")
            return pd.DataFrame()
    
    def _create_cluster_map(self, df_map_data: pd.DataFrame, output_path):
        """
        Creates and displays an interactive map visualizing the city clusters.

        Args:
            df_map_data: DataFrame from Part 1 containing City, Cluster, Lat, and Lon.
            output_filename: The name of the HTML file to save the map to.
            
        Returns:
            A folium.Map object that will be displayed in the notebook.
        """
        if df_map_data.empty:
            print("Input data for map is empty. Cannot generate visualization.")
            return None

        # Validate required columns
        required_cols = ['City', 'Cluster', 'Lat', 'Lon']
        missing_cols = [col for col in required_cols if col not in df_map_data.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None

        # --- 1. Initialize the map centered on Indonesia ---
        map_center = [-2.548926, 118.0148634]
        cluster_map = folium.Map(
            location=map_center, 
            zoom_start=5, 
            tiles="OpenStreetMap"
        )

        # --- 2. Create a color palette for the clusters ---
        num_clusters = int(df_map_data['Cluster'].max() + 1)
        # Using a built-in colormap for clear, distinct colors
        from matplotlib.colors import to_hex
        colors = plt.cm.get_cmap('tab10', num_clusters)
        cluster_colors = {i: to_hex(colors(i)) for i in range(num_clusters)}
        
        print(f"🗺️  Creating map with {num_clusters} clusters")
        print(f"📍 Plotting {len(df_map_data)} cities")

        # --- 3. Add a marker for each city ---
        for _, row in df_map_data.iterrows():
            cluster_id = int(row['Cluster'])
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=8,  # Slightly larger for better visibility
                color=cluster_colors.get(cluster_id, '#808080'),
                fill=True,
                fill_color=cluster_colors.get(cluster_id, '#808080'),
                fill_opacity=0.7,
                weight=2,  # Border thickness
                popup=folium.Popup(
                    f"<b>{row['City']}</b><br>Cluster: {cluster_id}", 
                    max_width=200
                )
            ).add_to(cluster_map)

        # --- 4. Add a legend ---
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin-bottom: 5px; font-weight: bold;">City Clusters</p>
        '''
        for cluster_id, color in cluster_colors.items():
            legend_html += f'''
            <p style="margin: 3px 0;">
                <span style="background-color:{color}; 
                            width: 20px; height: 20px; 
                            display: inline-block; 
                            border: 1px solid black;
                            margin-right: 5px;"></span>
                Cluster {cluster_id}
            </p>
            '''
        legend_html += '</div>'
        cluster_map.get_root().html.add_child(folium.Element(legend_html))

        # --- 5. Save the map to an HTML file ---
        save_path = output_path / "cluster_map.html"
        try:
            cluster_map.save(str(save_path))
            print(f"✓ Map saved as cluster_map.html")
        except Exception as e:
            print(f"❌ Error saving map: {e}")
            
        # Print cluster color mapping
        print("\n🎨 Cluster Colors:")
        for cluster_id, color in cluster_colors.items():
            print(f"  Cluster {cluster_id}: {color}")
        
        return cluster_map
    
def main():
    """
    Example of running the complete preprocessing pipeline.
    
    """
    
    SCALED_DATA_PATH = Path("data/features/scaled/feature_matrix_robust_scaled_20251006_175724.csv")
    PREPROCESSED_DATA_PATH = Path("data/processed/food_prices_consolidated.csv")
    
    scaled_df = pd.read_csv(SCALED_DATA_PATH)
    preprocessed_df = pd.read_csv(PREPROCESSED_DATA_PATH)
    
    X = scaled_df.drop(columns=["City"])
    
    clustering_service = ClusteringService(
        algorithm="kmeans",
        scaled_df=scaled_df,
        preprocessed_df=preprocessed_df,
        X=X,
        n_clusters=3
    )
    
    clustering_service.run_algorithm()


if __name__ == "__main__":
    exit(main())