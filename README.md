# Matrix Decomposition

A project focused on building a movie recommendation system using Matrix Factorization. The idea for this pet project came after studying linear algebra and learning about the Netflix Prize (2006), where this method demonstrated impressive performance.

Key result: The model predicts user ratings ~12% more accurately than the average KinoPoisk rating.

Kaggle Notebook: [Matrix Decomposition](https://www.kaggle.com/code/ivan4956/matrix-decomposition)  
*(Recommended to view on Kaggle - the notebook is easier to read there)*

---

## Dataset

Collected via KinoPoisk web scraping:

- 309 users  
- 4,495 movies (after filtering)  
- 108,769 ratings (after cleaning)  
- Matrix sparsity: 7.8%

---

## Main Stages

#### 1. DATA COLLECTION

- Parsing KinoPoisk user profiles  
- Extracting ratings, movie info (year, duration, rating)  
- Final raw dataset: ~130,000 records  


#### 2. PREPROCESSING

**Data Cleaning**

- Removed rating value "1" (treated as social artifacts rather than true opinions)  
- Excluded movies with fewer than 5 ratings (to ensure statistical reliability)  
- Final dataset: 108,769 ratings

**Preparation for Modeling**
- Train/Test split: 90% / 10%  
- Created user-movie rating matrix (pivot table)  
- Generated a binary mask to track known ratings  


#### MATRIX FACTORIZATION

**Algorithm**
- Decomposition of sparse rating matrix:  `R ≈ P × Q`
- Optimization via Gradient Descent

**Model Parameters**
- Latent factors: `K = 80`  
- Iterations: `400`  
- Learning rate: `0.001`  
- Regularization (λ): `0.015`

**Metrics**
- Train Loss (MSE): `0.547`  
- Test RMSE: `1.178`

#### 4. RESULTS ANALYSIS

| Model | RMSE |
|-------|-----:|
| KinoPoisk average rating | 1.334 |
| Matrix Factorization     | 1.178 |
| Improvement              | 11.7% |

Matrix Factorization effectively reconstructs missing values in the sparse user-item matrix.  
The model outperforms the KinoPoisk average (population-level estimate).

---

## Conclusion

This project successfully recreated a simplified movie-recommendation workflow (similar to Netflix), using 100k+ KinoPoisk ratings. The effectiveness of matrix factorization for recommendation systems is confirmed.

This method is not limited to rating prediction - it can be used across various business domains where many users, items, and ratings interact:

- E-commerce (product recommendations)  
- Streaming services (movies, music)  
- Online education (courses, materials)  
- Content platforms (articles, news)  

Any task with a sparse **user-item matrix** can benefit from this approach.


## Tech Stack

`Python` • `NumPy` • `Pandas` • `Matplotlib` • `Seaborn`
