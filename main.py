"""
Portfolio Construction & Risk Analysis Framework

A comprehensive quantitative portfolio analysis system implementing:
- Multiple portfolio construction strategies (GMV, Mean-Variance, Equal-Weight, Active)
- Rolling window backtesting with realistic holding periods
- Historical simulation VaR estimation and validation
- Performance analytics and visualization

Author: Portfolio Analytics Team
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
np.random.seed(42)  # Reproducibility
plt.style.use('seaborn-v0_8-darkgrid')  # Modern plot style

# ============================================================================
# DATA PIPELINE: Loading and Preprocessing
# ============================================================================

def load_and_preprocess_data():
    """
    Load and preprocess market data.
    
    Loads historical stock prices, market factors, and risk-free rates.
    Performs data quality checks, date alignment, and missing data handling.
    
    Returns:
        tuple: (stocks_df, factors_df, stock_columns)
            - stocks_df: DataFrame with aligned stock prices
            - factors_df: DataFrame with market factor and risk-free rate
            - stock_columns: List of valid stock tickers
    """
    print("=" * 80)
    print("DATA PIPELINE: Loading and Preprocessing")
    print("=" * 80)
    
    # Load stock data
    stocks_df = pd.read_csv('Stocks_data.csv')
    stocks_df['Dates'] = pd.to_datetime(stocks_df['Dates'], format='%d-%m-%Y')
    stocks_df = stocks_df.sort_values('Dates').reset_index(drop=True)
    
    # Load market factor and risk-free rate
    factors_df = pd.read_csv('market_Factor_risk_Free.csv')
    factors_df['Date'] = pd.to_datetime(factors_df['Date'], format='%d-%m-%Y')
    factors_df = factors_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Initial stocks data shape: {stocks_df.shape}")
    print(f"Initial factors data shape: {factors_df.shape}")
    
    # Find common dates
    common_dates = pd.Series(list(set(stocks_df['Dates']) & set(factors_df['Date'])))
    common_dates = common_dates.sort_values().reset_index(drop=True)
    
    # Filter to common dates
    stocks_df = stocks_df[stocks_df['Dates'].isin(common_dates)].reset_index(drop=True)
    factors_df = factors_df[factors_df['Date'].isin(common_dates)].reset_index(drop=True)
    
    print(f"After matching dates - Stocks shape: {stocks_df.shape}, Factors shape: {factors_df.shape}")
    print(f"Date range: {stocks_df['Dates'].min()} to {stocks_df['Dates'].max()}")
    
    # Remove stocks with excessive missing data (>5% missing)
    stock_columns = [col for col in stocks_df.columns if col not in ['Dates', 'NIFTY Index']]
    missing_pct = stocks_df[stock_columns].isna().sum() / len(stocks_df) * 100
    
    stocks_to_keep = missing_pct[missing_pct <= 5].index.tolist()
    print(f"\nRemoving stocks with >5% missing data")
    print(f"Stocks retained: {len(stocks_to_keep)} out of {len(stock_columns)}")
    
    # Keep only valid stocks plus date and index
    cols_to_keep = ['Dates'] + stocks_to_keep + ['NIFTY Index']
    stocks_df = stocks_df[cols_to_keep]
    
    # Forward fill remaining missing values (holidays, etc.)
    stocks_df[stocks_to_keep] = stocks_df[stocks_to_keep].ffill()
    stocks_df[stocks_to_keep] = stocks_df[stocks_to_keep].bfill()
    
    print(f"Final stocks data shape: {stocks_df.shape}")
    print(f"Number of tradeable stocks: {len(stocks_to_keep)}")
    
    return stocks_df, factors_df, stocks_to_keep


# ============================================================================
# RETURNS ENGINE: Calculate Daily Returns
# ============================================================================

def calculate_returns(stocks_df, factors_df, stock_columns):
    """
    Calculate daily simple returns for all securities.
    
    Uses arithmetic returns: R_t = (P_t - P_{t-1}) / P_{t-1}
    Aligns factor data and normalizes market factor from percentage to decimal.
    
    Args:
        stocks_df: DataFrame with price data
        factors_df: DataFrame with market factors
        stock_columns: List of stock tickers to process
        
    Returns:
        tuple: (returns_df, factors_aligned)
    """
    print("\n" + "=" * 80)
    print("RETURNS ENGINE: Calculating Daily Returns")
    print("=" * 80)
    
    # Calculate simple returns for stocks
    returns_df = pd.DataFrame()
    returns_df['Dates'] = stocks_df['Dates'][1:].reset_index(drop=True)
    
    for col in stock_columns:
        returns_df[col] = stocks_df[col].pct_change().iloc[1:].reset_index(drop=True)
    
    # Calculate Nifty returns
    returns_df['NIFTY Index'] = stocks_df['NIFTY Index'].pct_change().iloc[1:].reset_index(drop=True)
    
    # Align factors data (already in returns format)
    # MF is in percentage, convert to decimal
    factors_aligned = factors_df[factors_df['Date'].isin(returns_df['Dates'])].reset_index(drop=True)
    factors_aligned['MF'] = factors_aligned['MF'] / 100  # Convert to decimal
    
    print(f"Returns data shape: {returns_df.shape}")
    print(f"Factors data shape: {factors_aligned.shape}")
    print(f"\nSample statistics (annualized):")
    print(f"Mean return (stocks): {returns_df[stock_columns].mean().mean() * 252:.4f}")
    print(f"Mean volatility (stocks): {returns_df[stock_columns].std().mean() * np.sqrt(252):.4f}")
    
    return returns_df, factors_aligned


# ============================================================================
# PORTFOLIO STRATEGIES: Construction Algorithms
# ============================================================================

def construct_gmv_portfolio(mu, Sigma):
    """
    Construct Global Minimum Variance (GMV) Portfolio.
    
    Implements Markowitz mean-variance optimization with variance minimization:
    - Objective: min w'Σw
    - Constraint: w'1 = 1 (fully invested)
    
    This strategy seeks the lowest-risk portfolio on the efficient frontier,
    ignoring expected returns. Includes covariance regularization for stability.
    
    Args:
        mu: Expected returns vector (n,)
        Sigma: Covariance matrix (n, n)
        
    Returns:
        np.array: Optimal portfolio weights
    """
    n = len(mu)
    
    # Objective: minimize 0.5 * w' * Sigma * w
    def objective(w):
        return 0.5 * w.T @ Sigma @ w
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds: allow short selling (can be changed if needed)
    bounds = tuple((-1, 1) for _ in range(n))
    
    # Initial guess: equal weights
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        print(f"GMV optimization failed: {result.message}")
        return np.ones(n) / n  # Return equal weights as fallback


def construct_tangency_portfolio(mu, Sigma, rf):
    """
    Construct Mean-Variance Efficient (Tangency) Portfolio.
    
    Maximizes the Sharpe ratio to find the optimal risk-adjusted portfolio:
    - Objective: max (w'μ - rf) / √(w'Σw)
    - Constraint: w'1 = 1
    
    Implements leverage constraints and covariance regularization to
    prevent extreme positions and improve numerical stability.
    
    Args:
        mu: Expected returns vector (n,)
        Sigma: Covariance matrix (n, n)
        rf: Risk-free rate (scalar)
        
    Returns:
        np.array: Optimal portfolio weights
    """
    n = len(mu)
    
    # Objective: minimize negative Sharpe ratio
    def neg_sharpe(w):
        port_return = w.T @ mu
        port_vol = np.sqrt(w.T @ Sigma @ w)
        if port_vol == 0:
            return 1e10
        return -(port_return - rf) / port_vol
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds: constrain leverage to prevent extreme positions
    # Using [-0.5, 1.5] to allow some short selling but limit extreme leverage
    bounds = tuple((-0.5, 1.5) for _ in range(n))
    
    # Initial guess: equal weights
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        print(f"Tangency optimization failed: {result.message}")
        return np.ones(n) / n


def construct_ew_portfolio(n):
    """
    Equal-Weighted Portfolio
    Simply allocates equal weight to all stocks
    """
    return np.ones(n) / n


def construct_active_portfolio(returns_df, factors_df, stock_columns, confidence_level=0.95):
    """
    Active Portfolio based on CAPM alpha significance
    
    Uses Single Index Model (SIM):
        R_i(t) = α_i + β_i R_M(t) + e_i(t)
    
    Process:
    1. Run regression: (R_i - RF) = α_i + β_i * MF + ε_i
    2. Test if α_i is significant at 95% confidence (t-test)
    3. For stocks with significant α, compute Information Ratio: IR_i = α_i / σ(e_i)
    4. Form active portfolio with weights proportional to: w_i ∝ α_i / σ²(e_i)
    
    From reference formulas:
    - Information Ratio: IR_A = α_A / σ(e_A)
    - Optimal Active Weight: w_A^0 = [α_A / σ²(e_A)] / [E(R_M) / σ_M²]
    
    If no stocks have significant alpha, return None (use market portfolio)
    """
    significant_alphas = {}
    significant_betas = {}
    residual_vars = {}
    
    # Prepare excess returns for stocks
    for stock in stock_columns:
        y = returns_df[stock].values - factors_df['RF'].values
        X = factors_df['MF'].values
        
        # Add constant for regression
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        try:
            coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            alpha = coeffs[0]
            beta = coeffs[1]
            
            # Calculate residuals and standard error
            y_pred = X_with_const @ coeffs
            residuals = y - y_pred
            n_obs = len(y)
            
            # Standard error of alpha
            residual_var = np.sum(residuals**2) / (n_obs - 2)
            X_var = np.sum((X - X.mean())**2)
            se_alpha = np.sqrt(residual_var * (1/n_obs + X.mean()**2 / X_var))
            
            # T-statistic
            t_stat = alpha / se_alpha if se_alpha > 0 else 0
            
            # Critical value for 95% confidence (two-tailed)
            critical_value = stats.t.ppf(1 - (1 - confidence_level) / 2, n_obs - 2)
            
            # Check significance
            if abs(t_stat) > critical_value:
                significant_alphas[stock] = alpha
                significant_betas[stock] = beta
                residual_vars[stock] = residual_var
        except:
            continue
    
    # If no significant alphas, return market portfolio
    if len(significant_alphas) == 0:
        print("  No significant alphas found, using market portfolio")
        return None  # Signal to use market portfolio
    
    # Construct active portfolio using Treynor-Black approach
    # From reference: Optimal Active Weight proportional to α_i / σ²(e_i)
    # This maximizes the portfolio Information Ratio
    active_stocks = list(significant_alphas.keys())
    stock_to_idx = {stock: i for i, stock in enumerate(stock_columns)}
    
    # Calculate optimal active weights: w_i ∝ α_i / σ²(e_i)
    # This comes from maximizing Sharpe²_P = Sharpe²_M + [α_A / σ(e_A)]²
    weights = np.zeros(len(stock_columns))
    
    for stock in active_stocks:
        alpha = significant_alphas[stock]
        res_var = residual_vars[stock]  # This is σ²(e_i)
        idx = stock_to_idx[stock]
        # Weight proportional to Information Ratio contribution: α_i / σ²(e_i)
        weights[idx] = alpha / res_var if res_var > 0 else 0
    
    # Normalize weights to sum to 1 (fully invested portfolio)
    if weights.sum() != 0:
        weights = weights / weights.sum()
    else:
        # Fallback to equal weight among significant stocks
        for stock in active_stocks:
            idx = stock_to_idx[stock]
            weights[idx] = 1.0 / len(active_stocks)
    
    print(f"  Active portfolio: {len(active_stocks)} stocks with significant alpha")
    
    return weights


# ============================================================================
# SECTION 4: ROLLING WINDOW BACKTESTING
# ============================================================================

def rolling_window_backtest(returns_df, factors_df, stock_columns):
    """
    Perform rolling window backtest:
    - Formation period: 6 months
    - Holding period: 3 months
    - Roll forward by 3 months each time
    
    Returns: DataFrame with 3-month holding period returns for each portfolio
    """
    print("\n" + "=" * 80)
    print("SECTION 4: Rolling Window Backtesting")
    print("=" * 80)
    
    # Approximate trading days
    FORMATION_DAYS = 126  # 6 months ≈ 126 trading days
    HOLDING_DAYS = 63     # 3 months ≈ 63 trading days
    ROLL_DAYS = 63        # Roll forward by 3 months
    
    results = []
    var_results = []
    
    start_idx = 0
    window_num = 0
    
    while start_idx + FORMATION_DAYS + HOLDING_DAYS <= len(returns_df):
        window_num += 1
        
        # Define formation and holding periods
        formation_end = start_idx + FORMATION_DAYS
        holding_end = formation_end + HOLDING_DAYS
        
        formation_start_date = returns_df['Dates'].iloc[start_idx]
        formation_end_date = returns_df['Dates'].iloc[formation_end - 1]
        holding_start_date = returns_df['Dates'].iloc[formation_end]
        holding_end_date = returns_df['Dates'].iloc[holding_end - 1]
        
        print(f"\nWindow {window_num}:")
        print(f"  Formation: {formation_start_date.strftime('%Y-%m-%d')} to {formation_end_date.strftime('%Y-%m-%d')}")
        print(f"  Holding:   {holding_start_date.strftime('%Y-%m-%d')} to {holding_end_date.strftime('%Y-%m-%d')}")
        
        # Extract formation period data
        formation_returns = returns_df.iloc[start_idx:formation_end][stock_columns].values
        formation_factors = factors_df.iloc[start_idx:formation_end]
        formation_returns_df = returns_df.iloc[start_idx:formation_end]
        
        # Estimate mean and covariance
        mu = formation_returns.mean(axis=0)
        Sigma = np.cov(formation_returns.T)
        
        # Add small regularization to covariance matrix for numerical stability
        # This prevents singular matrix issues and reduces extreme weights
        reg_factor = 1e-5 * np.trace(Sigma) / len(mu)
        Sigma_reg = Sigma + reg_factor * np.eye(len(mu))
        
        rf_mean = formation_factors['RF'].mean()
        
        # Construct portfolios
        w_gmv = construct_gmv_portfolio(mu, Sigma_reg)
        w_tangency = construct_tangency_portfolio(mu, Sigma_reg, rf_mean)
        w_ew = construct_ew_portfolio(len(stock_columns))
        w_active = construct_active_portfolio(formation_returns_df, formation_factors, stock_columns)
        
        # If no significant alpha, use market portfolio (proportional to market caps or equal weighted)
        if w_active is None:
            w_active = w_ew.copy()
        
        # Extract holding period returns
        holding_returns = returns_df.iloc[formation_end:holding_end][stock_columns].values
        nifty_holding_returns = returns_df.iloc[formation_end:holding_end]['NIFTY Index'].values
        
        # Calculate portfolio returns for each day in holding period
        gmv_daily = holding_returns @ w_gmv
        tangency_daily = holding_returns @ w_tangency
        ew_daily = holding_returns @ w_ew
        active_daily = holding_returns @ w_active
        
        # Compound returns over holding period
        gmv_return = np.prod(1 + gmv_daily) - 1
        tangency_return = np.prod(1 + tangency_daily) - 1
        ew_return = np.prod(1 + ew_daily) - 1
        active_return = np.prod(1 + active_daily) - 1
        nifty_return = np.prod(1 + nifty_holding_returns) - 1
        
        # Store results
        results.append({
            'Window': window_num,
            'Formation_Start': formation_start_date,
            'Formation_End': formation_end_date,
            'Holding_Start': holding_start_date,
            'Holding_End': holding_end_date,
            'GMV': gmv_return,
            'MV': tangency_return,
            'EW': ew_return,
            'Active': active_return,
            'NIFTY50': nifty_return
        })
        
        # VaR estimation using historical simulation
        # Use formation period returns to estimate VaR
        var_gmv = estimate_historical_var(formation_returns, w_gmv, HOLDING_DAYS, 0.99)
        var_tangency = estimate_historical_var(formation_returns, w_tangency, HOLDING_DAYS, 0.99)
        var_ew = estimate_historical_var(formation_returns, w_ew, HOLDING_DAYS, 0.99)
        var_active = estimate_historical_var(formation_returns, w_active, HOLDING_DAYS, 0.99)
        
        # Check VaR violations
        var_results.append({
            'Window': window_num,
            'Holding_End': holding_end_date,
            'GMV_VaR': var_gmv,
            'GMV_Return': gmv_return,
            'GMV_Violation': gmv_return < -var_gmv,
            'MV_VaR': var_tangency,
            'MV_Return': tangency_return,
            'MV_Violation': tangency_return < -var_tangency,
            'EW_VaR': var_ew,
            'EW_Return': ew_return,
            'EW_Violation': ew_return < -var_ew,
            'Active_VaR': var_active,
            'Active_Return': active_return,
            'Active_Violation': active_return < -var_active
        })
        
        print(f"  Returns: GMV={gmv_return:.4f}, MV={tangency_return:.4f}, EW={ew_return:.4f}, Active={active_return:.4f}, NIFTY={nifty_return:.4f}")
        
        # Roll forward
        start_idx += ROLL_DAYS
    
    results_df = pd.DataFrame(results)
    var_results_df = pd.DataFrame(var_results)
    
    print(f"\nTotal windows processed: {window_num}")
    
    return results_df, var_results_df


def estimate_historical_var(formation_returns, weights, holding_days, confidence_level):
    """
    Estimate VaR using historical simulation method
    
    Args:
        formation_returns: Historical returns from formation period (n_days x n_assets)
        weights: Portfolio weights
        holding_days: Number of days in holding period (N)
        confidence_level: Confidence level α (e.g., 0.99 for 99%)
    
    Returns:
        VaR estimate (positive number representing potential loss)
    
    From Risk Measures formulas:
        Portfolio Loss: L_t+1 := -(V_t+1 - V_t) = -R_t+1
        VaR Definition: VaR_α := q_α(L) (the α-quantile of loss distribution)
        
        For returns R, Loss L = -R, so:
        P(Loss > VaR_α) = 1 - α
        P(-R > VaR_α) = 1 - α
        P(R < -VaR_α) = 1 - α
        
        Therefore: VaR_α = -percentile(R, 1-α)
        
        For 99% confidence (α=0.99): VaR = -percentile(R, 1%)
        Violation occurs if realized return R < -VaR (i.e., loss exceeds VaR)
        
    N-day VaR Scaling (if needed): N-day VaR = 1-day VaR × √N
    """
    # Calculate daily portfolio returns from formation period
    portfolio_daily_returns = formation_returns @ weights
    
    # Check for extreme weights or numerical issues
    if np.any(np.abs(weights) > 10):
        # Extreme leverage detected, use more conservative approach
        # Cap weights to reduce instability
        weights_capped = np.clip(weights, -2, 2)
        weights_capped = weights_capped / np.sum(np.abs(weights_capped))  # Renormalize
        portfolio_daily_returns = formation_returns @ weights_capped
    
    # Simulate holding period returns by randomly sampling and compounding
    n_simulations = 10000
    simulated_returns = []
    
    for _ in range(n_simulations):
        # Sample with replacement
        sampled_daily = np.random.choice(portfolio_daily_returns, size=holding_days, replace=True)
        # Compound: (1+r1)*(1+r2)*...*(1+rn) - 1
        period_return = np.prod(1 + sampled_daily) - 1
        simulated_returns.append(period_return)
    
    simulated_returns = np.array(simulated_returns)
    
    # VaR at confidence level α: 
    # We want the loss threshold such that P(Loss > VaR) = 1-α
    # Since Loss = -Return, we want P(Return < -VaR) = 1-α
    # So -VaR is the (1-α) percentile of returns
    # Therefore VaR = -percentile(returns, (1-α)*100)
    
    alpha_percentile = (1 - confidence_level) * 100  # For 99%, this is 1%
    worst_case_return = np.percentile(simulated_returns, alpha_percentile)
    
    # VaR is the positive loss value
    var = -worst_case_return
    
    # Ensure VaR is positive (loss is a positive number)
    # If worst_case_return is positive (gain), VaR should be 0 or very small
    var = max(0, var)
    
    return var


# ============================================================================
# SECTION 5: PERFORMANCE EVALUATION
# ============================================================================

def calculate_performance_metrics(results_df):
    """
    Calculate performance metrics for each portfolio using standard formulas:
    
    From reference formulas:
    - Portfolio Expected Return: μ = E(R_p) = Σ w_i μ_i
    - Portfolio Variance: σ² = w'Σw
    - Sharpe Ratio: S_P = (r̄_P - r̄_f) / σ_P
    - Information Ratio (IR): IR_A = α_A / σ(e_A)
      where α_A is excess return over benchmark, σ(e_A) is tracking error
    
    Conventions:
    - 252 trading days per year
    - Each window return is a 3-month (63 trading days) return
    - Annualization: multiply mean by 4, multiply std by sqrt(4) = 2
    - Risk-free rate assumed ≈ 0 for Sharpe ratio (can be adjusted)
    """
    print("\n" + "=" * 80)
    print("SECTION 5: Performance Evaluation")
    print("=" * 80)
    
    portfolios = ['GMV', 'MV', 'EW', 'Active', 'NIFTY50']
    performance = {}
    
    # Assume risk-free rate ≈ 0 for simplicity (or compute from RF data)
    rf_annual = 0.0  # Can be updated to actual RF if needed
    
    for portfolio in portfolios:
        returns = results_df[portfolio].values
        
        # Mean return (3-month), annualized
        # Formula: μ = E(R_p)
        mean_return = returns.mean() * 4  # 4 quarters per year
        
        # Standard deviation (3-month), annualized
        # Formula: σ from σ² = w'Σw (computed from time series)
        std_return = returns.std() * 2  # sqrt(4) = 2
        
        # Sharpe Ratio: S_P = (r̄_P - r̄_f) / σ_P
        sharpe_ratio = (mean_return - rf_annual) / std_return if std_return > 0 else 0
        
        # Information Ratio: IR_A = α_A / σ(e_A)
        # where α_A = excess return vs benchmark
        #       σ(e_A) = tracking error (std dev of excess returns)
        if portfolio != 'NIFTY50':
            # Excess returns relative to Nifty 50 (α_A)
            excess_returns = returns - results_df['NIFTY50'].values
            
            # Mean excess return (annualized) - this is α_A
            alpha_a = excess_returns.mean() * 4
            
            # Tracking error σ(e_A) - std dev of excess returns (annualized)
            tracking_error = excess_returns.std() * 2  # Annualized
            
            # Information Ratio: IR_A = α_A / σ(e_A)
            info_ratio = alpha_a / tracking_error if tracking_error > 0 else 0
        else:
            info_ratio = 0  # Not applicable for benchmark
        
        performance[portfolio] = {
            'Mean Return (Ann.)': mean_return,
            'Std Dev (Ann.)': std_return,
            'Sharpe Ratio': sharpe_ratio,
            'Information Ratio': info_ratio
        }
    
    performance_df = pd.DataFrame(performance).T
    
    print("\nPerformance Metrics:")
    print(performance_df.to_string())
    
    return performance_df


# ============================================================================
# VISUALIZATION: Performance Charts
# ============================================================================

def plot_cumulative_returns(results_df):
    """
    Generate cumulative returns visualization.
    
    Creates a time-series plot comparing cumulative performance
    of all portfolio strategies vs. benchmark.
    
    Args:
        results_df: DataFrame with strategy returns
    """
    print("\n" + "=" * 80)
    print("VISUALIZATION: Generating Performance Charts")
    print("=" * 80)
    
    portfolios = ['GMV', 'MV', 'EW', 'Active', 'NIFTY50']
    
    # Calculate cumulative returns
    cumulative_returns = pd.DataFrame()
    cumulative_returns['Window'] = results_df['Window']
    
    for portfolio in portfolios:
        cumulative_returns[portfolio] = (1 + results_df[portfolio]).cumprod()
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    for portfolio in portfolios:
        plt.plot(cumulative_returns['Window'], cumulative_returns[portfolio], 
                marker='o', linewidth=2, markersize=4, label=portfolio)
    
    plt.xlabel('Window Number', fontsize=12)
    plt.ylabel('Cumulative Return (Base = 1)', fontsize=12)
    plt.title('Cumulative Returns: Portfolio Strategies vs Nifty 50', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cumulative_returns.png', dpi=300, bbox_inches='tight')
    print("Saved: cumulative_returns.png")
    plt.close()


def plot_var_backtest(var_results_df):
    """
    Plot VaR estimates vs realized returns for each portfolio
    """
    portfolios = ['GMV', 'MV', 'EW', 'Active']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, portfolio in enumerate(portfolios):
        ax = axes[idx]
        
        var_col = f'{portfolio}_VaR'
        return_col = f'{portfolio}_Return'
        violation_col = f'{portfolio}_Violation'
        
        windows = var_results_df['Window']
        var_values = -var_results_df[var_col].values  # Plot as negative
        realized_returns = var_results_df[return_col].values
        violations = var_results_df[violation_col].values
        
        # Plot VaR threshold
        ax.plot(windows, var_values, 'r--', linewidth=2, label='VaR (99%)', alpha=0.7)
        
        # Plot realized returns
        ax.plot(windows, realized_returns, 'b-', linewidth=1.5, label='Realized Return', alpha=0.8)
        
        # Mark violations
        violation_windows = windows[violations]
        violation_returns = realized_returns[violations]
        ax.scatter(violation_windows, violation_returns, color='red', s=100, 
                  marker='x', linewidths=3, label='VaR Violations', zorder=5)
        
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Window Number', fontsize=10)
        ax.set_ylabel('Return / Loss', fontsize=10)
        ax.set_title(f'{portfolio} Portfolio: VaR vs Realized Returns\n'
                    f'Violations: {violations.sum()} / {len(violations)}', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('var_backtest.png', dpi=300, bbox_inches='tight')
    print("Saved: var_backtest.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline.
    
    Orchestrates the complete portfolio analysis workflow:
    1. Data loading and preprocessing
    2. Return calculation
    3. Portfolio construction
    4. Rolling window backtesting
    5. Performance analytics
    6. Visualization
    7. Results export
    """
    print("\n" + "=" * 80)
    print("PORTFOLIO CONSTRUCTION & RISK ANALYSIS FRAMEWORK")
    print("Quantitative Portfolio Analytics | VaR Backtesting")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    stocks_df, factors_df, stock_columns = load_and_preprocess_data()
    
    # Step 2: Calculate returns
    returns_df, factors_aligned = calculate_returns(stocks_df, factors_df, stock_columns)
    
    # Step 3: Portfolio construction is done within rolling window function
    print("\n" + "=" * 80)
    print("SECTION 3: Portfolio Construction Functions")
    print("=" * 80)
    print("Functions implemented:")
    print("  1. Global Minimum Variance (GMV)")
    print("  2. Mean-Variance Efficient (Tangency)")
    print("  3. Equal-Weighted (EW)")
    print("  4. Active Portfolio (CAPM alpha significance)")
    
    # Step 4: Rolling window backtest
    results_df, var_results_df = rolling_window_backtest(returns_df, factors_aligned, stock_columns)
    
    # Step 5: Performance evaluation
    performance_df = calculate_performance_metrics(results_df)
    
    # Step 6: Visualizations
    plot_cumulative_returns(results_df)
    plot_var_backtest(var_results_df)
    
    # Step 7: Export results
    print("\n" + "=" * 80)
    print("EXPORT: Saving Results to CSV")
    print("=" * 80)
    
    # Export rolling returns
    output_returns = results_df[['Window', 'GMV', 'MV', 'EW', 'Active', 'NIFTY50']]
    output_returns.to_csv('rolling_3month_returns.csv', index=False)
    print("Saved: rolling_3month_returns.csv")
    
    # Export performance metrics
    performance_df.to_csv('performance_metrics.csv')
    print("Saved: performance_metrics.csv")
    
    # Export VaR results
    var_results_df.to_csv('var_backtest_results.csv', index=False)
    print("Saved: var_backtest_results.csv")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal rolling windows analyzed: {len(results_df)}")
    print(f"Analysis period: {results_df['Formation_Start'].iloc[0].strftime('%Y-%m-%d')} to {results_df['Holding_End'].iloc[-1].strftime('%Y-%m-%d')}")
    
    print("\n--- VaR Violation Summary (99% confidence) ---")
    for portfolio in ['GMV', 'MV', 'EW', 'Active']:
        violations = var_results_df[f'{portfolio}_Violation'].sum()
        total = len(var_results_df)
        violation_rate = violations / total * 100
        print(f"{portfolio:8s}: {violations:2d} / {total:2d} violations ({violation_rate:.2f}%)")
    
    print("\n--- Best Performing Portfolio (Sharpe Ratio) ---")
    best_portfolio = performance_df['Sharpe Ratio'].idxmax()
    best_sharpe = performance_df.loc[best_portfolio, 'Sharpe Ratio']
    print(f"{best_portfolio}: {best_sharpe:.4f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\n📈 Generated Outputs:")
    print("  ✓ cumulative_returns.png        - Performance visualization")
    print("  ✓ var_backtest.png              - VaR analysis charts")
    print("  ✓ rolling_3month_returns.csv    - Quarterly returns data")
    print("  ✓ performance_metrics.csv       - Performance statistics")
    print("  ✓ var_backtest_results.csv      - VaR backtest details")
    print("\n📊 All results exported to current directory.")
    print("🔍 Review outputs for insights and visualizations.")


if __name__ == "__main__":
    main()
