The Mann-Whitney test, also known as the Mann-Whitney U test or Wilcoxon rank-sum test, is a non-parametric test used to compare two independent samples to determine whether their distributions differ. Here’s a step-by-step guide on how to perform the test:

---

### Steps to Perform the Mann-Whitney Test

#### 1. Formulate Hypotheses

- Null Hypothesis (\(H_0\)): The distributions of the two groups are identical.
- Alternative Hypothesis (\(H_a\)): The distributions of the two groups are not identical (two-tailed), or one distribution tends to be larger than the other (one-tailed).

#### 2. Collect Data

- ~~Gather the Sammon errors (or other metric values) from two groups (e.g., SFS and REINFORCE results) for each number of metrics.~~
- ~~Ensure the samples are independent (values in one group do not influence the other).~~

#### 3. Choose a Significance Level

- Select a significance level, often \( \alpha = 0.05 \).

#### 4. Rank the Data

- Combine the data from both groups.
- Assign ranks to the combined data, with the smallest value getting rank 1.
- If there are ties (identical values), assign the average of the ranks they would occupy.

#### 5. Calculate Test Statistic

- Compute the sum of the ranks for each group (\(R_1\) and \(R_2\)).
- Use the smaller rank sum to compute the U statistic:
     \[
     U = n_1 n_2 + \frac{n_1 (n_1 + 1)}{2} - R_1
     \]
     where:
  - \(n_1\) and \(n_2\) are the sample sizes of the two groups,
  - \(R_1\) is the sum of ranks for group 1.

#### 6. Obtain the \(p\)-Value

- Compare the U statistic to the Mann-Whitney U distribution or use statistical software to compute the \(p\)-value.

#### 7. Interpret Results

- If \(p\)-value \(< \alpha\): Reject \(H_0\) and conclude that the distributions differ.
- If \(p\)-value \(\geq \alpha\): Fail to reject \(H_0\) and conclude that there is no significant difference between the distributions.

---

### Example Using Python

Here’s how you can perform the Mann-Whitney test using Python's scipy.stats library:

from scipy.stats import mannwhitneyu

# Sample data: Sammon errors for SFS and REINFORCE

sfs_errors = [0.1, 0.15, 0.12, 0.14, 0.11]
reinforce_errors = [0.11, 0.13, 0.12, 0.13, 0.14]

# Perform Mann-Whitney U test

stat, p_value = mannwhitneyu(sfs_errors, reinforce_errors, alternative='two-sided')

# Print the results

print(f"Mann-Whitney U statistic: {stat}")
print(f"P-value: {p_value}")

# Interpret the results

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The distributions are different.")
else:
    print("Fail to reject the null hypothesis: The distributions are similar.")

---

### Output Explanation

- Mann-Whitney U statistic: A measure of rank differences between the two groups.
- \(p\)-value: The probability of observing the data under the null hypothesis.
- If the \(p\)-value is small (below \( \alpha \)), the test indicates a significant difference between the groups.

This process can be repeated for each subset size (e.g., 2 metrics, 3 metrics) to compare the results of SFS and REINFORCE across different conditions. Let me know if you need help implementing this for your dataset!

## Hypothesis Testing

### Identical Results for SFS and REINFORCE

To determine whether Sequential Forward Selection (SFS) and the REINFORCE algorithm yield identical results, we conducted the Mann-Whitney test, a non-parametric alternative to the independent \(t\)-test. This test compares the distributions of Sammon errors for the two methods across different numbers of selected metrics.

The null hypothesis (\(H_0\)) states that the cumulative distribution functions (CDFs) of the Sammon errors for SFS and REINFORCE are identical, indicating no significant difference between the methods. The alternative hypothesis (\(H_a\)) posits that the CDFs of the Sammon errors for the two methods are not identical.

**Table 1: Results from Mann-Whitney test for Sammon error**

| Number of Metrics | Class    | Method   |
|--------------------|----------|----------|
| 2                 | X.XXXX   | Y.YYYY   |
| 3                 | X.XXXX   | Y.YYYY   |
| 4                 | X.XXXX   | Y.YYYY   |
| 5                 | X.XXXX   | Y.YYYY   |
| 6                 | X.XXXX   | Y.YYYY   |
| 7                 | X.XXXX   | Y.YYYY   |
| 8                 | X.XXXX   | Y.YYYY   |

### Interpretation of Results

For each test, a \(p\)-value below the significance level (\( \alpha = 0.05 \)) would indicate that the null hypothesis should be rejected, suggesting that the results of the two methods are not identical. Conversely, a \(p\)-value above 0.05 implies that the null hypothesis cannot be rejected, supporting the claim that the methods yield identical results.

In cases where the \(p\)-value exceeds 0.05, we conclude that there is no statistically significant difference between the Sammon errors of SFS and REINFORCE, and the two methods can be considered to produce similar results.

### Meta-Analysis

To further confirm the similarity between the results of SFS and REINFORCE, we conducted a meta-analysis by aggregating the Sammon errors from both methods. The aggregated analysis supports the Mann-Whitney test results, strengthening the evidence for the similarity or difference between the methods.

### External Validity

The external validity of these findings is supported by the diverse selection of datasets used in this study. By including repositories of varying sizes and characteristics, the generalizability of the results across different types of software projects is enhanced.
