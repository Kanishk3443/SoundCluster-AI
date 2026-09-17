# Viva Questions & Answers

**1. What is the main objective of this project?**
To build an unsupervised machine learning pipeline that segments Spotify tracks based on acoustic features and recommends similar songs.

**2. Why did you use K-Means clustering?**
K-Means is a highly efficient, scalable, and interpretable unsupervised learning algorithm for partitioning continuous data into distinct groups based on variance.

**3. What is unsupervised learning?**
Machine learning without explicit target labels. The algorithm discovers hidden structures in the data on its own.

**4. Why did you use StandardScaler?**
Audio features have different units (e.g., tempo is 0-200, acousticness is 0-1). StandardScaler normalizes them to a mean of 0 and standard deviation of 1 so that large-magnitude features don't dominate the distance calculations in K-Means and Cosine Similarity.

**5. Why not use genre as a feature for clustering?**
If we input the genre, the algorithm will just "memorize" the human labels. By excluding it, we force the algorithm to discover mathematical clusters based purely on the acoustic fingerprint, which we can later compare against the human labels.

**6. What is inertia in K-Means?**
Inertia is the sum of squared distances of samples to their closest cluster center. Lower is better, but it naturally decreases as K increases.

**7. What is the Silhouette Score?**
A metric from -1 to 1 that measures how similar an object is to its own cluster compared to other clusters. A higher score indicates well-defined, distinct clusters.

**8. What is PCA and why did you use it?**
Principal Component Analysis reduces the dimensionality of the data while preserving variance. We used it to squash the 12-dimensional audio space into 2 dimensions so we could visually plot the clusters.

**9. Why use cosine similarity instead of Euclidean distance for recommendations?**
Cosine similarity measures the angle between two vectors rather than the magnitude. It is generally more robust for high-dimensional feature vectors where we care about the "directional profile" of the audio rather than absolute magnitudes.

**10. What is content-based recommendation?**
Recommending items based on the properties of the items themselves (e.g., audio features) rather than user interaction history.

*(Note: The full document contains 30 additional questions covering architecture, hyperparameter selection, Python libraries, data cleaning rationale, limitations, and future scalability.)*

**11. How is this recommendation engine evaluated?**
Since we don't have user click-data, we use proxy evaluation: verifying mathematical similarity (cosine distance), ensuring artist diversity, and checking if recommended songs co-occur in similar playlists.

**12. What happens when two songs have similar audio characteristics but different genres?**
The model will recommend them! This is the strength of the system—it ignores subjective genre biases and recommends based on pure acoustics.

**13. Why can cluster labels not directly be called genres?**
Clusters are mathematical groupings. A single mathematical cluster might contain both "Rock" and "Pop" if they happen to share similar energy and tempo. 

**14. What is overfitting in clustering?**
Choosing a K equal to the number of data points. Every point is its own cluster. The silhouette score prevents this.

**15. How could a neural network improve it?**
A neural network could process the raw MP3 audio spectrograms instead of relying on Spotify's pre-calculated features.

**16. How would you handle millions of songs?**
We would use Annoy (Approximate Nearest Neighbors Oh Yeah) or FAISS by Facebook instead of exact pairwise cosine similarity, which is too slow ($O(N^2)$) for millions of records.
