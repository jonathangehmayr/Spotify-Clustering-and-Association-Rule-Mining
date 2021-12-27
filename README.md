# Spotify-Clustering-and-Association-Rule-Mining

This project was developed in university with the purpose of applying unsupervised machine learning models in practice. The main tasks of this project are association rule mining and clustering. 

## Project structure

The repo is structured in the following way:

    
    ├── Spotify-Clustering-and-Association-Rule-Mining
    ├── Association Rule Mining                                       # Association rule mining scripts
    │   ├── spotify_association_rule_mining_walk_through.ipynb        # Explanatory jupyter notebook
    │   └── spotify_association_rules.py                              # Main script for mining association rule mining
    ├── Clustering                                                    # Clustering scripts
    │   ├──  spotify_clustering_walk_through.ipynb                    # Explanatory jupyter notebook
    │   ├──  spotify_clustering.py                                    # Main script for clustering
    │   └── tree_cluster_rules_retrieval.py                           # Cluster property extraction with decision tree
    ├── Spotify_dataset.csv                                           # Spotify dataset
    ├── README.md
    └── .gitignore

## Project Description

### Data
The used dataset is a subset from [the-spotify-hit-predictor-dataset](https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset/code) assembled by university. The [Spotify_dataset.csv](https://github.com/jonathangehmayr/Spotify-Clustering-and-Association-Rule-Mining/blob/main/Spotify_dataset.csv) file contains following variables:

* **track** - The Name of the track.
* **artist** - The Name of the Artist.
* **danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
* **energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. 
* **key** - The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C?/D?, 2 = D, and so on. If no key was detected, the value is -1.
* **loudness** - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. 
* **mode** - Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
* **speechiness** - Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. 
* **acousticness** - A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. The distribution of values for this feature look like this:
* **instrumentalness** - Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. The distribution of values for this feature look like this:
* **liveness** - Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
* **valence** - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
* **tempo** - The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. 
* **duration_ms** - The duration of the track in milliseconds.
* **time_signature** - An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
* **chorus_hit** - This the the author's best estimate of when the chorus would start for the track. Its the timestamp of the start of the third section of the track.
* **sections** - The number of sections the particular track has. This feature was extracted from the data received by the API call for Audio Analysis of that particular track.
* **decade** - The decade in which the track was released. It can take on the following values: '60s', '70s', '80s', '90s', '00s', '10s'.
* **hit** - Hit or flop? It can be either 0 or 1. 1 implies that this song has featured in the weekly list (Issued by Billboards) of Hot-100 tracks in that decade at least once and is therefore a hit. 0 Implies that the track is a flop.


### Association Rule Mining
The main goal of the association rule mining tasks was to find hidden patterns that occur for songs labeled as hit songs. In the preprocessing step continous variables were binned into a discrete number of groups and categorial variables were transformed into a different representation format. Furthermore, as the `track` variable could provide interesting insights it was analyzed using NLP. After preprocessing of the data the Apriori algorithm was applied to infer frequent itemsets. The result of the Apriori algorithm was used to deduce association rules. Since, the goal was to find patterns of hit songs only association rules were considered with hit song labels as consequents. The results indicate that hit songs often have in common high `energy`, `danceability`, `loudness`, `speed` and low `acousticness`, `instrumentalness`, `speechiness`, `liveness` values , matching logical assumption about hit songs. Interestingly, the language variables extracted from the `track` variable was not once among the items in association rules for hit songs, indicating that song titles are not characterising hit songs.

### Clustering
Clustering the Spotify data aimed to find groups with samples having common properties. For this task an experimental analysis was defined like following:

  1. **Determination of how many clusters describe the data best by calculation of silhouette coefficients**
  2. **Exploration of which variable is a possible target class candidate by iteratively using each variable of the data as target variable and storing of the F1-score.**
  3. **Further analyis of the run in 2. leading to the highest F1-score**
  4. **Plotting of the samples in 3d space; firstly colored according to their true class of the best run in 2. and secondly to their belonging cluster predicted by the clustering algorithm.**
  5. **Extraction of cluster properties by training a decision tree**

Calculating the silhouette coefficients for different amount of clusters suggested that 2 clusters lead to the best results, whereas the silhouette coefficient was still quite low. Using KMeans as clustering algorithm the target variable candidate matching best the predicted clusters was the variable `energy`. Thus, in the followng the data samples were plotted in 3d space on the one hand colored regarding to the `energy ` variable and on the other according to the predicted cluster labels. In the last analysis step the goal was to find common properties for intra cluster samples. Training a decision tree on the data labeled with the predicted cluster labels and retrieving the splitting criterions yielded the characteristics of the clusters. In the script `spotify_clustering.py` functions are defined to run this analyis with wheter KMeans or hierarchical clustering algorithms.

## Exemplary Notebooks

Applying the scripts in this repo is explained with two jupyter notebooks, one for the association rule mining and one for the clustering respectively.

* [Notebook](https://github.com/jonathangehmayr/Spotify-Clustering-and-Association-Rule-Mining/blob/main/Association%20Rule%20Mining/spotify_association_rule_mining_walk_through.ipynb) for association rule mining
* [Notebook](https://github.com/jonathangehmayr/Spotify-Clustering-and-Association-Rule-Mining/blob/main/Clustering/spotify_clustering_walk_through.ipynb) for clustering




