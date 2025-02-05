import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import gamma  # for reference, though we use np.random.gamma below

st.title("Experiment Winner Calculator")
st.write(
    """
This calculator uses a Bayesian approach (similar in spirit to the methodology used by Meta Ads)
to determine which experiment cell is the winner.
For each cell, enter the amount spent and the number of conversions.
The app computes a posterior conversion rate (conversions per unit spend),
estimates 95% credible intervals, and calculates the winning probability.
A cell is declared the winner if its winning probability is 95% or above.
"""
)

# Ask for the number of experiment groups (cells)
n_cells = st.number_input("Number of Experiment Groups (Cells)", min_value=2, value=2, step=1)

st.write("### Enter Data for Each Cell")
# Use a form so the user can enter all data then click “Calculate”
with st.form("input_form"):
    cells = []
    # For each cell, get its name, amount spent, and conversions.
    for i in range(int(n_cells)):
        st.subheader(f"Cell {i+1}")
        cell_name = st.text_input(f"Cell {i+1} Name", value=f"Cell {i+1}", key=f"cell_name_{i}")
        spent = st.number_input(
            f"Amount Spent for {cell_name}",
            min_value=0.0,
            value=100.0,
            step=1.0,
            key=f"spent_{i}",
            help="Enter the amount of money spent (for example, in dollars)."
        )
        conversions = st.number_input(
            f"Number of Conversions for {cell_name}",
            min_value=0,
            value=10,
            step=1,
            key=f"conv_{i}",
            help="Enter the total number of conversions observed."
        )
        cells.append({"name": cell_name, "spent": spent, "conversions": conversions})
    submitted = st.form_submit_button("Calculate")

if submitted:
    st.write("## Results")

    # --- Bayesian Analysis Setup ---
    # We assume that conversions arise as a Poisson process with rate λ (conversions per dollar spent).
    # With a Gamma(α, β) prior on λ, and assuming:
    #    conversions ~ Poisson(λ * spent)
    # the posterior is also Gamma distributed with parameters:
    #    α_post = α_prior + conversions
    #    β_post = β_prior + spent
    #
    # Here we choose a relatively noninformative prior: Gamma(1, 1)
    alpha_prior = 1
    beta_prior = 1

    n_samples = 100000  # number of simulation draws
    cell_samples = {}   # to store simulation draws for each cell
    results = []        # to store summary statistics for each cell

    # For each cell, compute posterior parameters and simulate draws
    for cell in cells:
        # Compute the posterior parameters
        alpha_post = alpha_prior + cell["conversions"]
        beta_post = beta_prior + cell["spent"]
        # Draw samples from Gamma(alpha_post, scale=1/beta_post)
        samples = np.random.gamma(alpha_post, 1.0 / beta_post, n_samples)
        cell_samples[cell["name"]] = samples

        # Compute summary statistics:
        posterior_mean = alpha_post / beta_post
        ci_lower = np.percentile(samples, 2.5)
        ci_upper = np.percentile(samples, 97.5)
        observed_rate = cell["conversions"] / cell["spent"] if cell["spent"] > 0 else np.nan

        results.append({
            "Cell": cell["name"],
            "Amount Spent": cell["spent"],
            "Conversions": cell["conversions"],
            "Observed Rate (Conversions/Spend)": observed_rate,
            "Posterior Mean": posterior_mean,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper,
            # We will add "Winning Probability" below.
        })

    # --- Determine the Winning Probability ---
    # Stack the simulation draws for all cells into a 2D array.
    # Each row corresponds to a cell.
    cell_names = list(cell_samples.keys())
    samples_array = np.vstack([cell_samples[name] for name in cell_names])
    # For each simulation draw (each column), determine which cell had the highest lambda
    winners = np.argmax(samples_array, axis=0)
    # Count how often each cell “wins”
    win_counts = np.bincount(winners, minlength=len(cell_names))
    win_probabilities = win_counts / n_samples

    # Add the winning probability to each cell’s results.
    for i, name in enumerate(cell_names):
        for result in results:
            if result["Cell"] == name:
                result["Winning Probability"] = win_probabilities[i]

    # --- Declare a Winner (if any) ---
    # For this example, we declare a winner if the cell’s winning probability is at least 95%.
    winner_row = max(results, key=lambda x: x["Winning Probability"])
    if winner_row["Winning Probability"] >= 0.95:
        st.success(
            f"The winner is **{winner_row['Cell']}** with a winning probability of "
            f"{winner_row['Winning Probability']*100:.1f}%."
        )
    else:
        st.info("No cell has reached the 95% confidence threshold to be declared a winner.")

    # Display detailed results in a table.
    df_results = pd.DataFrame(results)
    st.write("### Detailed Results per Cell")
    st.dataframe(df_results)
