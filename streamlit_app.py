import streamlit as st
import numpy as np
import pandas as pd

st.title("Bayesian Test for Difference in CPA")

st.write(
    """
Compare two cells (A and B) based on **Cost per Acquisition (CPA)** using a
Bayesian Gamma-Poisson model. 

### How it works:
1. We assume each cell's _conversion rate_ (conversions per dollar spent) has a Gamma(\u03B1, \u03B2) prior.
2. Observed data: Cell i has `Spent_i` (total $) and `Conversions_i`.
3. Posterior for the rate \u03BB_i is Gamma(\u03B1_post, \u03B2_post).
4. We **invert** samples from the rate distribution to get a distribution for CPA_i = 1 / \u03BB_i.
5. We compare CPA_A vs. CPA_B to find:
   - Probability( CPA_A < CPA_B )
   - 95% credible intervals for each CPA
   - Which cell is likely to have a lower CPA.
"""
)

# Create a form to group inputs together
with st.form("cpa_form"):
    st.subheader("Cell A Inputs")
    spent_A = st.number_input(
        "Spent (Cell A)", 
        min_value=0.0, 
        value=100.0, 
        step=1.0, 
        key="spent_A"
    )
    conv_A = st.number_input(
        "Conversions (Cell A)", 
        min_value=0, 
        value=10, 
        step=1, 
        key="conv_A"
    )

    st.subheader("Cell B Inputs")
    spent_B = st.number_input(
        "Spent (Cell B)", 
        min_value=0.0, 
        value=100.0, 
        step=1.0, 
        key="spent_B"
    )
    conv_B = st.number_input(
        "Conversions (Cell B)", 
        min_value=0, 
        value=12, 
        step=1, 
        key="conv_B"
    )

    # Let user pick a threshold for "winning probability"
    threshold = st.selectbox(
        "Winning Probability Threshold", 
        [0.95, 0.90, 0.80], 
        index=0, 
        key="prob_threshold"
    )

    # The form submit button
    submitted = st.form_submit_button("Calculate")

if submitted:
    # Basic validation
    if spent_A <= 0 or spent_B <= 0:
        st.error("Please enter positive spend for both cells.")
        st.stop()
    if conv_A < 0 or conv_B < 0:
        st.error("Please enter non-negative conversions.")
        st.stop()

    # --- Bayesian setup ---
    # Prior: Gamma(alpha_prior, beta_prior), fairly uninformative:
    alpha_prior = 1.0
    beta_prior = 1.0

    # Posterior parameters for each cell:
    #    alpha_post = alpha_prior + conversions
    #    beta_post  = beta_prior  + spend
    alpha_post_A = alpha_prior + conv_A
    beta_post_A = beta_prior + spent_A

    alpha_post_B = alpha_prior + conv_B
    beta_post_B = beta_prior + spent_B

    # Sample from posterior of each cell's rate (lambda = conversions/$)
    n_samples = 100_000
    rate_samples_A = np.random.gamma(shape=alpha_post_A, scale=1.0/beta_post_A, size=n_samples)
    rate_samples_B = np.random.gamma(shape=alpha_post_B, scale=1.0/beta_post_B, size=n_samples)

    # Convert rate samples to CPA samples: CPA = 1 / rate
    cpa_samples_A = 1.0 / rate_samples_A
    cpa_samples_B = 1.0 / rate_samples_B

    # Compute posterior means, medians, and 95% credible intervals
    cpa_mean_A = np.mean(cpa_samples_A)
    cpa_mean_B = np.mean(cpa_samples_B)

    cpa_median_A = np.median(cpa_samples_A)
    cpa_median_B = np.median(cpa_samples_B)

    cpa_ci95_A = np.percentile(cpa_samples_A, [2.5, 97.5])
    cpa_ci95_B = np.percentile(cpa_samples_B, [2.5, 97.5])

    # Probability that A's CPA is lower than B's
    prob_A_lower = np.mean(cpa_samples_A < cpa_samples_B)
    prob_B_lower = 1 - prob_A_lower

    # Observed CPA
    observed_cpa_A = (spent_A / conv_A) if conv_A > 0 else np.nan
    observed_cpa_B = (spent_B / conv_B) if conv_B > 0 else np.nan

    # Summarize results in a DataFrame
    df = pd.DataFrame(
        {
            "Cell": ["A", "B"],
            "Spent": [spent_A, spent_B],
            "Conversions": [conv_A, conv_B],
            "Observed CPA (Spent/Conversions)": [
                observed_cpa_A,
                observed_cpa_B,
            ],
            "Posterior Mean CPA": [cpa_mean_A, cpa_mean_B],
            "Median CPA": [cpa_median_A, cpa_median_B],
            "95% Credible Interval (Lower)": [cpa_ci95_A[0], cpa_ci95_B[0]],
            "95% Credible Interval (Upper)": [cpa_ci95_A[1], cpa_ci95_B[1]],
        }
    )

    st.subheader("Posterior Estimates of CPA")
    st.dataframe(df.style.format(precision=4))

    st.markdown(
        f"""
        **Probability that Cell A has a lower CPA than Cell B** = {prob_A_lower:.3f}  
        **Probability that Cell B has a lower CPA than Cell A** = {prob_B_lower:.3f}  
        """
    )

    # Declare a winner if above threshold
    if prob_A_lower >= threshold:
        st.success(
            f"**Cell A** is the winner with probability "
            f"{prob_A_lower:.1%} of having a lower CPA."
        )
    elif prob_B_lower >= threshold:
        st.success(
            f"**Cell B** is the winner with probability "
            f"{prob_B_lower:.1%} of having a lower CPA."
        )
    else:
        st.info(
            f"Neither cell has reached the {threshold*100:.0f}% winning probability threshold."
        )
