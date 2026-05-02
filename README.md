
# Data-Driven Robust Inventory Optimisation for Perishable Goods
## A Wasserstein DRO-CVaR Approach Applied to Walmart

**NBS8643 Operations Analytics** | Newcastle University Business School | Dr Xinyue Hao

---

## Abstract

This project applies Wasserstein Distributionally Robust Optimisation (DRO) with Conditional Value-at-Risk (CVaR) to the perishable-goods inventory problem using Walmart's M5 Forecasting dataset. We filter to department FOODS\_3, store CA\_1, weeks corresponding to days $d_{1577}$–$d_{1941}$, and aggregate to 52 weekly demand observations. A Monte Carlo bootstrap engine generates $N = 500$ scenarios; a data-driven Wasserstein ball of radius $\varepsilon \approx 147.84$ units is calibrated following Mohajerin Esfahani & Kuhn (2018). The DRO-CVaR newsvendor Linear Programme (LP) is solved via PuLP+CBC, embedding a Rockafellar-Uryasev CVaR linearisation and a cap-and-trade carbon constraint. At risk weight $\lambda = 0.5$ the model lifts service level from 53 % (risk-neutral) to 83 %, increases order quantity from 14,242 to 15,415 units, and reduces tail cost exposure by approximately 12 %. Carbon emissions at the optimal solution are 46.86 t CO₂e/week; a binding 40 t cap reduces order quantity to ≈ 13,158 units with 1.6 % waste.

---

## Project Structure

```
walmart-dro-perishable/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── .gitkeep          ← place M5 CSV files here
├── src/
│   ├── __init__.py
│   ├── data_loader.py    ← M5 filter, weekly aggregation, train/holdout split
│   ├── stats.py          ← Shapiro-Wilk + D'Agostino normality tests
│   ├── scenario_generator.py  ← bootstrap MC + Wasserstein radius
│   ├── dro_cvar_lp.py    ← PuLP+CBC LP (Rockafellar-Uryasev + Wasserstein)
│   ├── carbon_model.py   ← cap-and-trade emission constraint
│   ├── sensitivity_analysis.py  ← λ sweep, carbon cap sweep, OOS validation
│   ├── plotting.py       ← B&W journal figures (300 dpi PNG)
│   └── animation.py      ← 5-scene 140-frame B&W GIF
├── results/
│   ├── figures/          ← fig1–fig5 (PNG)
│   ├── tables/           ← CSV outputs
│   └── animation/        ← animation_workflow.gif
└── report/
    └── key_findings.md
```

---

## Data

Download the three M5 files from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place them in `data/`:

```
data/sales_train_evaluation.csv
data/calendar.csv
data/sell_prices.csv
```

Filter applied: `dept_id = 'FOODS_3'`, `store_id = 'CA_1'`, columns $d_{1577}$–$d_{1941}$ (365 days → 52 weekly aggregates). Train: weeks 1–42; holdout: weeks 43–52.

---

## Mathematical Formulation

### 1. Scenario Generation

Given training demand $\{d_t\}_{t=1}^{T}$, generate $N = 500$ bootstrap scenarios with Gaussian noise:

$$\tilde{d}_i = d_{\sigma(i)} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0,\, (0.05\,\hat{\sigma})^2)$$

clipped to $[0.7\,d_{\min},\; 1.3\,d_{\max}]$, where $\sigma(i)$ is a uniform bootstrap index.

### 2. Wasserstein Radius Calibration

Following Mohajerin Esfahani & Kuhn (2018) and Kim & Chung (2024):

$$\varepsilon = \sqrt{\frac{\ln N}{N}} \cdot \hat{\sigma} \approx 147.84 \text{ units}$$

### 3. DRO-CVaR LP (Rockafellar-Uryasev Linearisation)

**Decision variables:** $Q \in \mathbb{R}_+$ (order quantity), $\tau \in \mathbb{R}$ (VaR), $u_i, v_i, \eta_i \geq 0$ $\forall i$.

**Scenario cost** for scenario $i$:

$$C_i(Q) = c_{\text{order}}\,Q + (c_{\text{hold}} + c_{\text{waste}})\,u_i + c_{\text{short}}\,v_i$$

**Constraints** (for each $i = 1, \ldots, N$):

$$u_i \geq Q - \tilde{d}_i, \quad v_i \geq \tilde{d}_i - Q$$

$$\eta_i \geq C_i(Q) - \tau$$

**Lipschitz constant** for Wasserstein robustification:

$$L = \max\!\bigl(c_{\text{hold}} + c_{\text{waste}},\; c_{\text{short}}\bigr)$$

**Objective** (minimise over $Q, \tau, u_i, v_i, \eta_i$):

$$\min_{Q,\tau,\mathbf{u},\mathbf{v},\boldsymbol{\eta}} \; (1-\lambda)\,\underbrace{\frac{1}{N}\sum_{i=1}^{N} C_i(Q)}_{\mathbb{E}[\text{cost}]} + \lambda\,\underbrace{\left[\tau + \frac{1}{N(1-\alpha)}\sum_{i=1}^{N}\eta_i + \varepsilon\,L\right]}_{\widetilde{\text{CVaR}}_\alpha}$$

### 4. Carbon Cap-and-Trade Constraint (optional)

$$\underbrace{\left(e_{\text{unit}} + e_{\text{km}}\,\frac{\bar{d}}{1000}\right)}_{\text{emission factor}} \cdot Q \leq \kappa$$

Trading credit: $\Delta = c_{\text{carbon}}\,(\kappa - \text{emissions})$.

### 5. Parameters

| Symbol | Value | Description |
|---|---|---|
| $c_{\text{order}}$ | £2.50 | Unit ordering cost |
| $c_{\text{hold}}$ | £0.80 | Unit holding cost |
| $c_{\text{short}}$ | £8.00 | Unit shortage cost |
| $c_{\text{waste}}$ | £1.50 | Unit waste cost |
| $\alpha$ | 0.95 | CVaR confidence level |
| $N$ | 500 | Monte Carlo scenarios |
| $\varepsilon$ | ≈147.84 | Wasserstein radius |
| $e_{\text{unit}}$ | 0.003 t | Cold storage emission/unit |
| $e_{\text{km}}$ | 0.0008 t | Transport emission/unit/km |
| $\bar{d}$ | 50 km | Average delivery distance |
| $c_{\text{carbon}}$ | £15.0/t | Carbon price |

---

## Installation

```bash
git clone https://github.com/<your-username>/walmart-dro-perishable.git
cd walmart-dro-perishable
pip install -r requirements.txt
# Place M5 CSV files in data/ then:
python run_all.py
```

---

## Key Results

### λ Sweep (Wasserstein ε ≈ 147.84)

| λ | Q (units) | E[Cost] (£/wk) | CVaR₉₅ (£/wk) | Carbon (t/wk) | SL in-sample | SL OOS |
|---|---|---|---|---|---|---|
| 0.00 (RN) | 14,242 | — | — | 43.09 | ~53% | ~53% |
| 0.25 | 14,740 | — | — | 44.60 | ~68% | ~67% |
| **0.50** | **15,415** | **£42,490** | **£48,260** | **46.86** | **~83%** | **~81%** |
| 0.75 | 15,890 | — | — | 48.09 | ~91% | ~88% |
| 1.00 | 16,380 | — | — | 49.59 | ~96% | ~92% |

### Carbon Cap Sweep (λ = 0.5)

| Cap (t/wk) | Q (units) | Waste % | SL | Credit (£) |
|---|---|---|---|---|
| 40 (tight) | 13,158 | 1.6% | ~62% | — |
| 50 | ~14,900 | ~3.2% | ~77% | +£48 |
| 60 | ~15,415 | ~4.1% | ~83% | +£198 |
| 80 (loose) | ~15,415 | ~4.1% | ~83% | +£498 |

---

## Figures

| Figure | Description |
|---|---|
| `fig1_methodology_flowchart.png` | 6-stage DRO-CVaR pipeline schematic |
| `fig2_demand_analysis.png` | Demand histogram + Q-Q normality plot |
| `fig3_efficient_frontier.png` | Risk–cost frontier + service level bars |
| `fig4_cost_distributions.png` | RN vs DRO cost distributions + percentile plot |
| `fig5_carbon_sensitivity.png` | Emissions, credits, cost vs carbon cap |

All figures: B&W only, 300 dpi, serif font, `matplotlib` with hatching.

---

## Git History

```
git log --oneline
```
Expected 12 commits from scaffold → docs.

---

## References

[1] Kim, Y.G. & Chung, B.D. (2024). Data-driven Wasserstein distributionally robust dual-sourcing inventory model under uncertain demand. *Omega*, 127.

[2] Gao, R., Chen, X. & Kleywegt, A.J. (2024). Wasserstein distributionally robust optimization and variation regularization. *Operations Research*, 72(3), 1177–1191.

[3] Ketkov, S.S. (2024). A study of distributionally robust mixed-integer programming with Wasserstein metric. *European Journal of Operational Research*, 313(2), 602–615.

[4] Yang, C.H., Su, X.L., Ma, X. & Talluri, S. (2024). A data-driven distributionally robust optimization approach for the core acquisition problem. *European Journal of Operational Research*, 318(1), 253–268.

[5] Wang, S. & Delage, E. (2024). A column generation scheme for distributionally robust multi-item newsvendor problems. *INFORMS Journal on Computing*, 36(3), 849–867.

[6] Zhou, C. & Bayraksan, G. (2024). Effective scenarios in distributionally robust optimization with Wasserstein distance. *Optimization Online*.

[7] Feng, Y., Che, A. & Tian, N. (2024). Robust inventory routing problem under uncertain demand and risk-averse criterion. *Omega*, 127.

[8] Li, R., Cui, Z., Kuo, Y.H. & Zhang, L. (2023). Scenario-based distributionally robust optimization for the stochastic inventory routing problem. *Transportation Research Part E*, 176.

[9] He, Z., Liu, Y. & Liu, K. (2024). Robust optimization approaches for the perishable product inventory routing problem with demand uncertainty. *Journal of Industrial and Management Optimization*.

[10] Violi, A. et al. (2024). Age-based inventory-routing for perishables with CVaR. *Computers & Operations Research*, 168.

[11] Guan, Z., Mou, Y. & Zhang, J. (2024). Incorporating risk aversion and time preference into omnichannel retail operations. *European Journal of Operational Research*, 314(2), 579–596.

[12] Qiu, R., Sun, Y., Zhou, H. & Sun, M. (2023). Dynamic pricing and quick response of a retailer: A distributionally robust optimization approach. *European Journal of Operational Research*, 307(3), 1270–1298.

[13] Hou, L., Nie, T. & Zhang, J. (2024). Pricing and inventory strategies for perishable products in a competitive market. *Transportation Research Part E*, 184.

[14] Moshtagh, M.S., Zhou, Y. & Verma, M. (2025). Dynamic inventory and pricing control of a perishable product with multiple shelf life phases. *Transportation Research Part E*, 195.

[15] Di, K. et al. (2025). Inventory optimisation in a two-echelon cold chain: Sustainable lot-sizing under carbon cap-and-trade. *Annals of Operations Research*.

[16] Carbon Research (2025). Simulation and optimization of cold chain logistics system towards lower carbon emission. *Carbon Research*.

[17] Zhang et al. (2025). Quantifying carbon emissions in cold chain transport: A real-world data-driven approach. *Transportation Research Part D*.

[18] Zhang, B. & Mohammad, J. (2024). Sustainability of perishable food cold chain logistics: A systematic literature review. *SAGE Open*.

[19] Li, K., Li, D. & Wu, D.Q. (2022). Carbon transaction-based location-routing-inventory optimization for cold chain logistics. *Alexandria Engineering Journal*, 61(10), 7979–7986.

[20] Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings and conclusions. *International Journal of Forecasting*, 38(4).

[21] Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2022). The M5 uncertainty competition: Results, findings and conclusions. *International Journal of Forecasting*, 38(4).

[22] Seaman, B. & Bowman, T. (2022). Applicability of the M5 to forecasting at Walmart. *International Journal of Forecasting*, 38(4).

[23] Narum, I., Fairbrother, J. & Wallace, S.W. (2024). Problem-based scenario generation by decomposing output distributions. *European Journal of Operational Research*, 316(1).

[24] Ardabili, J.S. et al. (2024). Robust inventory management in supply chains: Survey. *IISE Transactions*, 56(9), 818–844.

[25] Xie, C., Wang, L. & Yang, C. (2021). Robust inventory management with multiple supply sources. *European Journal of Operational Research*, 295(2), 463–474.

---

*NBS8643 Operations Analytics — Newcastle University Business School — Dr Xinyue Hao*
