# 🎵 SoundCluster AI

### Intelligent Music Clustering & Recommendation System

**SoundCluster AI** is a machine learning-based music analysis and recommendation system that analyzes audio features, discovers patterns between tracks, groups similar music, and generates recommendations through an interactive Streamlit application.

The project combines **K-Means clustering, PCA-based visualization, and content-based recommendation** to explore relationships between different music tracks.

---

## 📌 About the Project

With large collections of music available across digital platforms, discovering songs with similar characteristics can be challenging.

SoundCluster AI explores how **Machine Learning** can be used to understand music through its numerical audio characteristics.

The system processes music data, analyzes relevant audio features, applies clustering techniques to discover groups of similar tracks, and uses content-based similarity to generate recommendations.

The project also provides an interactive **Streamlit dashboard** for exploring the results.

---

## 🎯 Objectives

* Analyze numerical audio characteristics of music.
* Identify patterns and relationships within music data.
* Group similar tracks using machine learning.
* Visualize music clusters in a lower-dimensional space.
* Recommend tracks based on content similarity.
* Provide an interactive interface for music exploration.

---

## ✨ Key Features

### 🎧 Audio Feature Analysis

Analyzes numerical characteristics of music tracks to understand their similarities and differences.

### 🤖 K-Means Music Clustering

Uses **K-Means clustering** to group tracks with similar audio characteristics.

### 📊 PCA Visualization

Uses **Principal Component Analysis (PCA)** to reduce feature dimensions and visualize music clusters.

### 🎵 Content-Based Recommendation

Generates recommendations by comparing the characteristics of tracks and identifying similar music.

### 🖥️ Interactive Streamlit Dashboard

Provides an interactive interface for exploring the music analysis and recommendation results.

### 📈 Data Visualization

Visualizes patterns and relationships within the music dataset.

---

## 🧠 Machine Learning Approach

The project follows a machine learning workflow:

```text
              Music Dataset
                    │
                    ▼
            Data Preprocessing
                    │
                    ▼
          Audio Feature Selection
                    │
                    ▼
             Feature Scaling
                    │
                    ▼
            K-Means Clustering
                    │
             ┌──────┴──────┐
             ▼             ▼
       Music Clusters   PCA Analysis
             │             │
             └──────┬──────┘
                    ▼
          Similarity Analysis
                    │
                    ▼
       Content-Based Recommendation
                    │
                    ▼
         Streamlit Web Application
```

---

## 🔬 Technologies Used

| Technology           | Purpose                              |
| -------------------- | ------------------------------------ |
| **Python**           | Core programming language            |
| **Pandas**           | Data processing and analysis         |
| **NumPy**            | Numerical computation                |
| **Scikit-learn**     | Machine learning, K-Means and PCA    |
| **Streamlit**        | Interactive web application          |
| **Jupyter Notebook** | Data exploration and experimentation |

---

## 📊 Audio Features

The recommendation and clustering workflow works with numerical audio characteristics such as:

* Danceability
* Energy
* Acousticness
* Valence
* Tempo
* Loudness
* Instrumentalness
* Speechiness
* Liveness

These features allow tracks to be represented numerically and compared using machine learning and similarity-based techniques.

---

## 🔄 Recommendation Workflow

```text
Select a Track
      │
      ▼
Extract Audio Features
      │
      ▼
Process & Scale Features
      │
      ▼
Compare Track Characteristics
      │
      ▼
Identify Similar Tracks
      │
      ▼
Generate Recommendations
```

---

## 📊 Clustering & Visualization

### K-Means Clustering

K-Means is used to divide the music dataset into groups based on similarities in their audio characteristics.

This helps identify collections of tracks that share similar musical properties.

### PCA Visualization

Principal Component Analysis is used to reduce the dimensionality of the feature space so that the resulting clusters can be visualized more effectively.

```text
High-Dimensional Audio Features
              │
              ▼
             PCA
              │
              ▼
      Reduced Dimensions
              │
              ▼
      Cluster Visualization
```

---

## 🖥️ Application

SoundCluster AI includes an interactive **Streamlit application** for exploring the implemented music analysis and recommendation workflow.

The application provides an accessible interface for interacting with the underlying machine learning system.

### Run the application

```bash
streamlit run app/app.py
```

---

## 📂 Project Structure

```text
SoundCluster-AI/
│
├── spotify/
│   ├── app/
│   │   └── app.py
│   │
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   │
│   ├── models/
│   │
│   ├── notebooks/
│   │
│   ├── requirements.txt
│   ├── run_project.py
│   └── README.md
│
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kanishk3443/SoundCluster-AI.git
```

### 2. Navigate to the project

```bash
cd SoundCluster-AI/spotify
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app/app.py
```

The Streamlit application will open in your browser.

> **Note:** If preprocessing or model-generation steps are required, run the corresponding scripts or notebooks provided in the project before launching the application.

---

## 📈 Expected Results

SoundCluster AI provides a machine learning workflow for:

* Identifying groups of similar music
* Exploring audio-feature relationships
* Visualizing music clusters
* Finding similar tracks
* Generating content-based recommendations

---

## 🎓 Learning Outcomes

This project provided practical experience in:

* Python programming
* Data preprocessing
* Exploratory data analysis
* Feature selection and scaling
* K-Means clustering
* Principal Component Analysis
* Content-based recommendation
* Similarity analysis
* Data visualization
* Streamlit application development
* Machine learning project organization

---

## 👨‍💻 Author

**Kanishk M**

Student | Computer Science Engineering

GitHub: [Kanishk3443](https://github.com/Kanishk3443)

---

## 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for the complete license terms.
