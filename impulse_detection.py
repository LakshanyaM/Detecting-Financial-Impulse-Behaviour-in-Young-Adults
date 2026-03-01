import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import os

def load_and_preprocess(filepath: str, nrows: int = 300_000) -> pd.DataFrame:
    print(f"[1/6] Loading data (up to {nrows:,} rows)…")
    df = pd.read_csv(filepath, nrows=nrows, index_col=0, low_memory=False)

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df = df.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)

    df['amt'] = pd.to_numeric(df['amt'], errors='coerce').fillna(0)
    df['category'] = df['category'].astype(str)
    df['gender'] = df['gender'].astype(str)

    print(f"   → {len(df):,} rows, {df['cc_num'].nunique():,} unique cards")
    return df


IMPULSE_CATEGORIES = {
    'shopping_net', 'shopping_pos', 'entertainment',
    'misc_net', 'misc_pos', 'food_dining'
}
ESSENTIAL_CATEGORIES = {
    'grocery_pos', 'grocery_net', 'gas_transport', 'health_fitness', 'home'
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Engineering behavioural features…")

    df['hour']       = df['trans_date_trans_time'].dt.hour
    df['dayofweek']  = df['trans_date_trans_time'].dt.dayofweek   
    df['day']        = df['trans_date_trans_time'].dt.day
    df['month']      = df['trans_date_trans_time'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_night']   = df['hour'].isin(range(21, 24)).astype(int)  
    df['is_late_night'] = df['hour'].isin([0, 1, 2]).astype(int)   
    df['is_end_of_month'] = (df['day'] >= 25).astype(int)          

    df['is_impulse_cat']   = df['category'].isin(IMPULSE_CATEGORIES).astype(int)
    df['is_essential_cat'] = df['category'].isin(ESSENTIAL_CATEGORIES).astype(int)

    df = df.sort_values(['cc_num', 'trans_date_trans_time'])

    for window in [7, 30]:
        df[f'rolling_mean_{window}'] = (
            df.groupby('cc_num')['amt']
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f'rolling_std_{window}'] = (
            df.groupby('cc_num')['amt']
              .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )

    df['amt_zscore'] = (df['amt'] - df['rolling_mean_30']) / (df['rolling_std_30'] + 1e-5)

    df['ts_unix'] = df['trans_date_trans_time'].astype(np.int64) // 1e9

    def txn_velocity_6h(group):
        times = group['ts_unix'].values
        velocity = np.zeros(len(times))
        for i, t in enumerate(times):
            velocity[i] = np.sum((times[:i+1] > t - 6*3600) & (times[:i+1] <= t))
        return pd.Series(velocity, index=group.index)

    df['txn_velocity_6h'] = df.groupby('cc_num', group_keys=False).apply(txn_velocity_6h)

    df['impulse_cat_ratio_last10'] = (
        df.groupby('cc_num')['is_impulse_cat']
          .transform(lambda x: x.rolling(10, min_periods=1).mean())
    )

    df['time_since_last_txn'] = (
        df.groupby('cc_num')['ts_unix']
          .transform(lambda x: x.diff().fillna(0))
    )

    df['rapid_repeat'] = (df['time_since_last_txn'] < 300).astype(int)

    eom_avg = df[df['is_end_of_month'] == 0].groupby('cc_num')['amt'].mean().rename('eom_baseline')
    df = df.merge(eom_avg, on='cc_num', how='left')
    df['eom_spike'] = np.where(
        (df['is_end_of_month'] == 1) & (df['amt'] > df['eom_baseline'] * 1.5), 1, 0
    )

    card_stats = df.groupby('cc_num')['amt'].agg(
        card_mean_spend='mean', card_std_spend='std'
    ).reset_index()
    df = df.merge(card_stats, on='cc_num', how='left')
    df['card_std_spend'] = df['card_std_spend'].fillna(0)

    df['gender_enc'] = (df['gender'] == 'M').astype(int)


    le = LabelEncoder()
    df['category_enc'] = le.fit_transform(df['category'])

    print("   → Feature engineering complete")
    return df

def create_impulse_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    A transaction is labelled IMPULSE if it meets ≥3 of these criteria:
      1. Is in an impulse category
      2. Amount z-score > 1.5  (significantly above personal baseline)
      3. Happens at night/late-night OR weekend
      4. High velocity (≥3 txns in 6 hours)
      5. Rapid repeat (<5 min from last txn)
      6. End-of-month and above 1.5× baseline
    """
    print("[3/6] Creating impulse behaviour labels…")
    score = (
        df['is_impulse_cat'].astype(int) +
        (df['amt_zscore'] > 1.5).astype(int) +
        ((df['is_night'] | df['is_weekend']).astype(int)) +
        (df['txn_velocity_6h'] >= 3).astype(int) +
        df['rapid_repeat'].astype(int) +
        df['eom_spike'].astype(int)
    )
    df['impulse_label'] = (score >= 3).astype(int)
    df['impulse_raw_score'] = score

    pct = df['impulse_label'].mean() * 100
    print(f"   → {pct:.1f}% transactions flagged as impulsive ({df['impulse_label'].sum():,})")
    return df


FEATURE_COLS = [
    'amt', 'hour', 'dayofweek', 'day', 'is_weekend', 'is_night',
    'is_late_night', 'is_end_of_month', 'is_impulse_cat', 'is_essential_cat',
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30',
    'amt_zscore', 'txn_velocity_6h', 'impulse_cat_ratio_last10',
    'time_since_last_txn', 'rapid_repeat', 'eom_spike',
    'card_mean_spend', 'card_std_spend', 'gender_enc', 'category_enc'
]

def train_model(df: pd.DataFrame):
    print("[4/6] Training model…")
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].fillna(0)
    y = df['impulse_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n   ── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Impulse']))
    print(f"   ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    return clf, X_test, y_test, y_proba, feats

def score_transactions(df: pd.DataFrame, clf, feats: list) -> pd.DataFrame:
    print("[5/6] Generating Impulse Risk Scores (0–100)…")
    X_full = df[feats].fillna(0)
    df['impulse_risk_score'] = (clf.predict_proba(X_full)[:, 1] * 100).round(1)

    profile = df.groupby('cc_num').agg(
        mean_risk_score=('impulse_risk_score', 'mean'),
        max_risk_score=('impulse_risk_score', 'max'),
        impulse_txn_count=('impulse_label', 'sum'),
        total_txn_count=('impulse_label', 'count'),
        total_spend=('amt', 'sum'),
        impulse_spend=('amt', lambda x: x[df.loc[x.index, 'impulse_label'] == 1].sum()),
        top_category=('category', lambda x: x.mode()[0] if len(x) > 0 else 'unknown'),
        night_txn_pct=('is_night', 'mean'),
        weekend_txn_pct=('is_weekend', 'mean'),
        eom_spike_count=('eom_spike', 'sum'),
        rapid_repeat_count=('rapid_repeat', 'sum'),
    ).reset_index()

    profile['impulse_ratio'] = (profile['impulse_txn_count'] / profile['total_txn_count']).round(3)
    profile['impulse_spend_ratio'] = (profile['impulse_spend'] / (profile['total_spend'] + 1e-5)).round(3)

 
    def tier(score):
        if score >= 60: return 'HIGH'
        elif score >= 35: return 'MEDIUM'
        else: return 'LOW'

    profile['risk_tier'] = profile['mean_risk_score'].apply(tier)
    print(f"   → Risk tiers: {profile['risk_tier'].value_counts().to_dict()}")
    return df, profile


NUDGES = {
    'HIGH': [
        "You've exceeded your usual spending pattern. Consider pausing non-essential purchases for 24 hours.",
        "Set a daily spending cap in your banking app to stay on track.",
        "Review this month's shopping & entertainment spend — it's significantly above your average.",
    ],
    'MEDIUM': [
        "Your impulse spending is slightly elevated this week. Try the 24-hour rule before buying.",
        "Consider moving 20% of discretionary funds to a savings sub-account.",
        "Group weekend purchases — batching shopping trips reduces impulse buys.",
    ],
    'LOW': [
        "Your spending patterns look healthy. Keep it up!",
        "Consider automating savings with the surplus from disciplined spending.",
        "You're on track — your essential-to-impulse ratio is well balanced.",
    ],
}

def get_nudges(risk_tier: str) -> list:
    return NUDGES.get(risk_tier, NUDGES['LOW'])

def save_visualizations(df: pd.DataFrame, profile: pd.DataFrame,
                         clf, X_test, y_test, y_proba, feats: list,
                         out_dir: str = "outputs"):
    print("[6/6] Saving visualizations…")
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['impulse_risk_score'], bins=50, color='steelblue', edgecolor='white')
    axes[0].axvline(35, color='orange', linestyle='--', label='Medium threshold')
    axes[0].axvline(60, color='red',    linestyle='--', label='High threshold')
    axes[0].set_title('Impulse Risk Score Distribution')
    axes[0].set_xlabel('Risk Score (0–100)')
    axes[0].legend()

    tier_counts = profile['risk_tier'].value_counts()
    colors = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'}
    bars = axes[1].bar(tier_counts.index, tier_counts.values,
                       color=[colors[t] for t in tier_counts.index])
    axes[1].set_title('Cardholders by Risk Tier')
    axes[1].set_ylabel('Number of Cardholders')
    for bar, val in zip(bars, tier_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(val), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig1_risk_distribution.png")
    plt.close()


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    hourly = df.groupby(['hour', 'impulse_label'])['amt'].mean().unstack().fillna(0)
    hourly.columns = ['Normal', 'Impulse']
    hourly.plot(ax=axes[0], marker='o', color=['steelblue', 'tomato'])
    axes[0].set_title('Avg Transaction Amount by Hour')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Avg Amount ($)')

    cat_impulse = (df.groupby('category')['impulse_label'].mean() * 100).sort_values(ascending=False)
    cat_impulse.plot(kind='bar', ax=axes[1], color='coral', edgecolor='white')
    axes[1].set_title('Impulse Rate by Category (%)')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig2_spending_patterns.png")
    plt.close()

  
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_df = pd.DataFrame({
        'feature': feats,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)
    ax.barh(importance_df['feature'], importance_df['importance'], color='teal')
    ax.set_title('Top Feature Importances (Random Forest)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig3_feature_importance.png")
    plt.close()


    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0,1],[0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve – Impulse Detection')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4_roc_curve.png")
    plt.close()

  
    fig, ax = plt.subplots(figsize=(10, 5))
    eom_daily = df.groupby('day').agg(
        avg_amt=('amt', 'mean'),
        impulse_pct=('impulse_label', 'mean')
    )
    ax2 = ax.twinx()
    ax.bar(eom_daily.index, eom_daily['avg_amt'], color='steelblue', alpha=0.6, label='Avg Amount')
    ax2.plot(eom_daily.index, eom_daily['impulse_pct'] * 100, color='red', marker='o', label='Impulse %')
    ax.set_xlabel('Day of Month')
    ax.set_ylabel('Avg Transaction Amount ($)', color='steelblue')
    ax2.set_ylabel('Impulse %', color='red')
    ax.set_title('End-of-Month Spending Surge')
    ax.axvspan(24.5, 31.5, alpha=0.1, color='red', label='EOM Zone')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig5_eom_surge.png")
    plt.close()

   
    sample = profile[profile['risk_tier'] == 'HIGH'].head(1)
    if sample.empty:
        sample = profile.head(1)

    tier = sample['risk_tier'].values[0]
    nudges = get_nudges(tier)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    color_map = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'}
    bg_color  = color_map.get(tier, '#27ae60')

    ax.add_patch(plt.Rectangle((0,0), 1, 1, color=bg_color, alpha=0.15,
                                transform=ax.transAxes, zorder=0))
    ax.text(0.5, 0.92, f"🎯 Behavioural Nudge — {tier} Risk Cardholder",
            ha='center', va='top', fontsize=13, fontweight='bold',
            color=bg_color, transform=ax.transAxes)
    for i, nudge in enumerate(nudges):
        ax.text(0.05, 0.72 - i*0.22, nudge,
                ha='left', va='top', fontsize=11,
                wrap=True, transform=ax.transAxes)
    ax.text(0.5, 0.05,
            f"Impulse Score: {sample['mean_risk_score'].values[0]:.1f} | "
            f"Impulse Ratio: {sample['impulse_ratio'].values[0]*100:.1f}%",
            ha='center', va='bottom', fontsize=10, style='italic',
            transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig6_nudge_card.png")
    plt.close()

    print(f"   → All figures saved to '{out_dir}/'")


def main(filepath: str = "credit_card_transactions.csv", nrows: int = 300_000):
    df      = load_and_preprocess(filepath, nrows)
    df      = engineer_features(df)
    df      = create_impulse_label(df)
    clf, X_test, y_test, y_proba, feats = train_model(df)
    df, profile = score_transactions(df, clf, feats)
    save_visualizations(df, profile, clf, X_test, y_test, y_proba, feats)

    # Sample output
    print("\n── Sample Cardholder Risk Profiles ──")
    cols = ['cc_num','mean_risk_score','risk_tier','impulse_ratio',
            'total_spend','impulse_spend','top_category']
    print(profile[cols].sort_values('mean_risk_score', ascending=False).head(10).to_string(index=False))

    # Save outputs
    profile.to_csv("outputs/cardholder_risk_profiles.csv", index=False)
    df[['cc_num','trans_date_trans_time','amt','category',
        'impulse_risk_score','impulse_label']].to_csv(
        "outputs/transaction_scores.csv", index=False)
    print("\nDone! Outputs saved to outputs/")
    return df, profile, clf

if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "credit_card_transactions.csv"
    main(fp)
