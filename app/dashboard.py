"""Streamlit dashboard for the airline revenue-management DP project."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from airline_dp.model import solve_dp, solve_mean_fare_dp
from airline_dp.policies import accept_all_policy, accept_if_fare_at_least, fixed_threshold_policy
from airline_dp.simulation import compare_policies

st.set_page_config(page_title="Airline Revenue Optimization · Summer Project", page_icon="✈️", layout="wide")
st.title("✈️ Airline Revenue Optimization")
st.caption("Summer 2026 learning project · Finite-horizon stochastic dynamic programming")
st.markdown("Independent educational model developed after academic discussions with Prof. Wang Zizhuo. This is not an operational airline pricing tool.")

with st.sidebar:
    st.header("Model parameters")
    capacity = st.slider("Seat capacity, C", 1, 40, 10)
    horizon = st.slider("Booking periods, T", 1, 100, 30)
    arrival_probability = st.slider("Arrival probability, λ", 0.0, 1.0, 0.5, 0.05)
    st.subheader("Fare distribution")
    fares = [float(st.number_input("Low fare (₹)", 0, 1000000, 1000, 100)),
             float(st.number_input("Medium fare (₹)", 0, 1000000, 2500, 100)),
             float(st.number_input("High fare (₹)", 0, 1000000, 5000, 100))]
    weights = np.array([st.slider("Low-fare weight", 0.0, 1.0, 0.50, 0.05),
                        st.slider("Medium-fare weight", 0.0, 1.0, 0.30, 0.05),
                        st.slider("High-fare weight", 0.0, 1.0, 0.20, 0.05)])
    replications = st.select_slider("Monte Carlo replications", options=[1000, 2500, 5000, 10000, 25000], value=10000)
    seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=12345)

if weights.sum() <= 0:
    st.error("At least one fare weight must be positive.")
    st.stop()
probabilities = weights / weights.sum()
solution = solve_dp(capacity, horizon, arrival_probability, fares, probabilities)
ce_solution = solve_mean_fare_dp(capacity, horizon, arrival_probability, fares, probabilities)
mean_fare = float(np.dot(fares, probabilities))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimal expected revenue", f"₹{solution.expected_revenue:,.0f}")
c2.metric("Mean offered fare", f"₹{mean_fare:,.0f}")
c3.metric("Initial bid price", f"₹{solution.thresholds[horizon, capacity]:,.0f}")
c4.metric("State space", f"{(capacity + 1) * (horizon + 1):,} states")

tab_model, tab_policy, tab_eval, tab_notes = st.tabs(["Value function", "Bid-price policy", "Policy evaluation", "Model & assumptions"])
with tab_model:
    st.subheader("Optimal value function Vₜ(x)")
    frame = pd.DataFrame(solution.values[1:, 1:], index=np.arange(1, horizon + 1), columns=np.arange(1, capacity + 1))
    frame.index.name, frame.columns.name = "Periods remaining", "Seats remaining"
    st.plotly_chart(px.imshow(frame, origin="lower", aspect="auto", color_continuous_scale="Blues",
                              labels={"x":"Seats remaining","y":"Periods remaining","color":"Expected revenue (₹)"}),
                    use_container_width=True)
    x_pick = st.slider("Fix seats x", 1, capacity, capacity, key="value_x")
    st.plotly_chart(px.line(x=np.arange(horizon + 1), y=solution.values[:, x_pick],
                            labels={"x":"Periods remaining","y":"Expected revenue (₹)"},
                            title=f"Value versus time (x={x_pick})"), use_container_width=True)

with tab_policy:
    st.subheader("Marginal seat value / bid price")
    threshold_frame = pd.DataFrame(solution.thresholds[1:, 1:], index=np.arange(1, horizon + 1), columns=np.arange(1, capacity + 1))
    threshold_frame.index.name, threshold_frame.columns.name = "Periods remaining", "Seats remaining"
    st.plotly_chart(px.imshow(threshold_frame, origin="lower", aspect="auto", color_continuous_scale="Reds",
                              labels={"x":"Seats remaining","y":"Periods remaining","color":"Bid price (₹)"}),
                    use_container_width=True)
    col1, col2, col3 = st.columns(3)
    x_state = col1.number_input("Seats remaining", min_value=1, max_value=capacity, value=capacity)
    t_state = col2.number_input("Periods remaining", min_value=1, max_value=horizon, value=horizon)
    offered_fare = col3.number_input("Offered fare (₹)", min_value=0, max_value=1000000, value=2500, step=100)
    bid = float(solution.thresholds[int(t_state), int(x_state)])
    st.metric("State bid price", f"₹{bid:,.0f}")
    if offered_fare >= bid:
        st.success(f"ACCEPT — ₹{offered_fare:,.0f} is at least the bid price.")
    else:
        st.warning(f"REJECT — preserve capacity; fare is below ₹{bid:,.0f}.")
    st.caption("Δₜ(x) = Vₜ₋₁(x) − Vₜ₋₁(x−1); ties are accepted.")

with tab_eval:
    st.write("Policies are evaluated on shared simulated arrival and fare paths to make comparisons less noisy.")
    policies = [("Optimal DP", accept_if_fare_at_least(solution)),
                ("Mean-fare CE policy", accept_if_fare_at_least(ce_solution)),
                ("Fixed threshold = mean fare", fixed_threshold_policy(mean_fare)),
                ("Accept all", accept_all_policy)]
    results = compare_policies(policies, solution, int(replications), int(seed))
    rows = [{"Policy": name, "Mean revenue (₹)": r.mean_revenue, "Standard error (₹)": r.standard_error,
             "Approx. 95% interval low (₹)": r.mean_revenue - 1.96*r.standard_error,
             "Approx. 95% interval high (₹)": r.mean_revenue + 1.96*r.standard_error,
             "Mean seats sold": r.mean_seats_sold, "Fill rate": r.fill_rate} for name, r in results.items()]
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(comparison, x="Policy", y="Mean revenue (₹)", error_y="Standard error (₹)"), use_container_width=True)
    st.caption(f"Exact DP value: ₹{solution.expected_revenue:,.2f}; Monte Carlo is an estimate with sampling uncertainty.")

with tab_notes:
    st.latex(r"V_t(x)=(1-\lambda)V_{t-1}(x)+\lambda\sum_i p_i\max\{f_i+V_{t-1}(x-1),V_{t-1}(x)\}")
    st.latex(r"\Delta_t(x)=V_{t-1}(x)-V_{t-1}(x-1),\quad \text{accept iff } f\geq\Delta_t(x)")
    st.markdown("""**State:** seats and periods remaining.  
**Action:** accept or reject the offered fare.  
**Reward:** fare if accepted, zero otherwise.  
**Assumptions:** known discrete fare distribution, at most one arrival per period, no cancellations, refunds, overbooking, or variable costs.  
This model is a learning tool, not empirical validation or a production pricing system.""")
    st.download_button("Download value function (CSV)", data=pd.DataFrame(solution.values).to_csv(index_label="t").encode("utf-8"),
                       file_name="value_function.csv", mime="text/csv")
