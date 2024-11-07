https://www.researchgate.net/publication/221570758_Adapt_the_mRMR_Criterion_for_Unsupervised_Feature_Selection

**Minimum Redundancy - Maximum Relevance (mRMR)** is traditionally a supervised feature selection method that selects features based on their relevance to the target variable while minimizing redundancy among selected features. However, when adapting it for **unsupervised feature selection**, the challenge becomes identifying features that provide the most information about the dataset's structure without using a target label.

To use **mRMR** in an unsupervised context while minimizing **Sammon error**, the process can be adapted as follows:

### Steps for Unsupervised mRMR Using Sammon Error Minimization

1. **Compute Pairwise Distances for All Features:**
   - Start by calculating the pairwise distances between all data points based on the feature space. Since we are performing unsupervised feature selection, this means measuring the distances based on each feature separately (or subsets of features).
   - Use a distance metric like **Euclidean distance** or any appropriate measure depending on the data (e.g., Manhattan, cosine similarity).

2. **Calculate Relevance (Maximum Relevance):**
   - Relevance in unsupervised feature selection can be interpreted as the ability of each feature to **preserve the structure** of the data when mapped to a lower-dimensional space.
   - To measure relevance:
     - Evaluate how much each individual feature contributes to preserving the original pairwise distances.
     - One way to measure this is by calculating the **Sammon error** using each feature or subset of features. The lower the Sammon error, the more relevant the feature is to maintaining the original data structure.

     Mathematically, the **Sammon error** for a single feature \( f_i \) is:

     \[
     E_{f_i} = \frac{1}{\sum_{i<j} d_{ij}} \sum_{i<j} \frac{(d_{ij} - \delta_{ij})^2}{d_{ij}}
     \]

     Where:
     - \( d_{ij} \) is the distance between data points \( i \) and \( j \) based on **all features**.
     - \( \delta_{ij} \) is the distance between data points \( i \) and \( j \) using only the feature \( f_i \).

     The **relevance** of feature \( f_i \) can be inversely related to the Sammon error \( E_{f_i} \).

3. **Minimize Redundancy (Minimum Redundancy):**
   - Redundancy measures how similar or redundant a feature is with respect to other selected features.
   - In the unsupervised context, redundancy can be computed by analyzing the correlation or mutual information between pairs of features. Two features are redundant if they capture similar structures or information.
   - Compute the **correlation coefficient** (or mutual information) between each pair of features and aim to select features that have low redundancy (i.e., low correlation or mutual information).

4. **Select Features Using mRMR:**
   - Now that you have calculated both **relevance** (minimizing Sammon error) and **redundancy**, apply the mRMR principle.
   - For each feature \( f_i \), you want to maximize relevance (by minimizing Sammon error) while minimizing redundancy with the already selected features:

     \[
     \text{mRMR}(f_i) = \max \left(\text{Relevance}(f_i) - \lambda \times \text{Redundancy}(f_i, \text{Selected Features})\right)
     \]

     Where \( \lambda \) is a hyperparameter that controls the tradeoff between relevance and redundancy.

5. **Iterative Feature Selection:**
   - Start by selecting the feature with the highest relevance (i.e., the one that minimizes Sammon error the most).
   - Iteratively select additional features that maximize the mRMR score, adding features that preserve data structure while reducing redundancy with previously selected features.

### Algorithm Summary

1. **Compute pairwise distances** for all data points in the original feature space.
2. **Calculate the Sammon error** for each feature, which will serve as the relevance score.
3. **Calculate feature redundancy** using correlation or mutual information between features.
4. **Iteratively select features** by applying mRMR: choose features that minimize Sammon error (high relevance) while reducing redundancy.
5. **Stop when a desired number of features is selected** or when adding more features doesn’t reduce Sammon error significantly.

### Practical Considerations

- **Distance metric choice**: The choice of distance metric (e.g., Euclidean, Manhattan) will impact both the Sammon error and the feature selection process.
- **Hyperparameter tuning**: \( \lambda \) controls the balance between redundancy minimization and relevance maximization, and it may need to be tuned based on the data.
- **Scaling and normalization**: It is crucial to scale the features before applying this method, as unscaled data can distort distance measurements and Sammon error.

This adaptation allows you to perform **unsupervised feature selection** by focusing on preserving the data structure (via Sammon error) while ensuring the selected features are complementary (low redundancy).