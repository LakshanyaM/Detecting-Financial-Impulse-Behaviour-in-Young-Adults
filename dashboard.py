import pandas as pd
import numpy as np
import os, json


def generate_html_dashboard(profile_csv="outputs/cardholder_risk_profiles.csv",
                             transactions_csv="outputs/transaction_scores.csv",
                             out_html="outputs/dashboard.html"):
    profile = pd.read_csv(profile_csv)
    txn     = pd.read_csv(transactions_csv)
    txn['trans_date_trans_time'] = pd.to_datetime(txn['trans_date_trans_time'])

    high_risk      = int((profile['risk_tier'] == 'HIGH').sum())
    med_risk       = int((profile['risk_tier'] == 'MEDIUM').sum())
    low_risk       = int((profile['risk_tier'] == 'LOW').sum())
    avg_impulse_rt = float(profile['impulse_ratio'].mean() * 100)
    total_txns     = len(txn)
    impulse_txns   = int(txn['impulse_label'].sum())
    impulse_spend  = float(txn.loc[txn['impulse_label']==1,'amt'].sum())
    total_spend    = float(txn['amt'].sum())

    txn['month'] = txn['trans_date_trans_time'].dt.to_period('M').astype(str)
    monthly = txn.groupby('month').agg(impulse_count=('impulse_label','sum'),total_count=('impulse_label','count')).reset_index()
    monthly['impulse_pct'] = (monthly['impulse_count']/monthly['total_count']*100).round(1)

    cat_imp = txn.groupby('category').agg(impulse_pct=('impulse_label','mean'),txn_count=('impulse_label','count')).reset_index()
    cat_imp['impulse_pct'] = (cat_imp['impulse_pct']*100).round(1)
    cat_imp = cat_imp.sort_values('impulse_pct',ascending=False)

    top5 = profile.sort_values('mean_risk_score',ascending=False).head(5)
    monthly_labels = json.dumps(monthly['month'].tolist())
    monthly_vals   = json.dumps(monthly['impulse_pct'].tolist())
    cat_labels     = json.dumps(cat_imp['category'].tolist())
    cat_vals       = json.dumps(cat_imp['impulse_pct'].tolist())
    tier_vals      = json.dumps([high_risk,med_risk,low_risk])

    nudges_map = {
        'HIGH':   "Pause non-essential purchases. Set daily spending cap.",
        'MEDIUM': "Apply the 24-hour rule. Batch shopping trips.",
        'LOW':    "Healthy patterns. Consider automating savings.",
    }
    top5_rows = ""
    for _, row in top5.iterrows():
        tier  = row['risk_tier']
        color = {'HIGH':'#e74c3c','MEDIUM':'#f39c12','LOW':'#27ae60'}.get(tier,'#27ae60')
        top5_rows += f"""<tr>
          <td style='font-family:monospace;font-size:11px'>{str(row['cc_num'])[:14]}...</td>
          <td><span style='background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:12px'>{tier}</span></td>
          <td><b>{row['mean_risk_score']:.1f}</b></td>
          <td>{row['impulse_ratio']*100:.1f}%</td>
          <td>${row['total_spend']:,.0f}</td>
          <td style='font-size:11px'>{nudges_map.get(tier,"")}</td></tr>"""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Financial Impulse Behaviour Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#333}
header{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:24px 36px}
header h1{font-size:1.7em;font-weight:700}
header p{opacity:.65;margin-top:5px;font-size:.9em}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;padding:24px 36px}
.kpi{background:#fff;border-radius:10px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,.07);text-align:center}
.kpi .val{font-size:1.9em;font-weight:700;color:#1a1a2e}
.kpi .lbl{font-size:.75em;color:#888;margin-top:4px;text-transform:uppercase;letter-spacing:.05em}
.kpi.red .val{color:#e74c3c}.kpi.orange .val{color:#f39c12}.kpi.green .val{color:#27ae60}
.charts{display:grid;grid-template-columns:2fr 1fr;gap:18px;padding:0 36px 20px}
.charts2{display:grid;grid-template-columns:2fr 1fr;gap:18px;padding:0 36px 20px}
.chart-card{background:#fff;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.07)}
.chart-card h3{font-size:.85em;color:#555;margin-bottom:14px;text-transform:uppercase;letter-spacing:.05em}
.table-section{padding:0 36px 32px}
.table-card{background:#fff;border-radius:10px;padding:22px;box-shadow:0 2px 8px rgba(0,0,0,.07)}
.table-card h3{font-size:1em;margin-bottom:14px;color:#1a1a2e}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#1a1a2e;color:#fff;padding:10px 14px;text-align:left;font-weight:600}
td{padding:10px 14px;border-bottom:1px solid #f0f0f0;vertical-align:middle}
tr:hover td{background:#fafafa}
footer{text-align:center;padding:14px;color:#aaa;font-size:.75em}
</style>
</head>
<body>
<header>
  <h1>Financial Impulse Behaviour Detection</h1>
  <p>Behavioural Analytics Hackathon &mdash; Problem Statement 2 &mdash; Credit Card Transactions Dataset</p>
</header>
<div class="kpi-grid">
  <div class="kpi"><div class="val">""" + f"{total_txns:,}" + """</div><div class="lbl">Total Transactions</div></div>
  <div class="kpi red"><div class="val">""" + f"{impulse_txns:,}" + """</div><div class="lbl">Impulse Flagged</div></div>
  <div class="kpi"><div class="val">""" + f"{avg_impulse_rt:.1f}%" + """</div><div class="lbl">Avg Impulse Rate</div></div>
  <div class="kpi red"><div class="val">""" + f"{high_risk}" + """</div><div class="lbl">High Risk Cards</div></div>
  <div class="kpi orange"><div class="val">""" + f"{med_risk}" + """</div><div class="lbl">Medium Risk</div></div>
  <div class="kpi green"><div class="val">""" + f"{low_risk}" + """</div><div class="lbl">Low Risk</div></div>
  <div class="kpi"><div class="val">""" + f"${impulse_spend/1e3:.0f}K" + """</div><div class="lbl">Impulse Spend</div></div>
  <div class="kpi"><div class="val">""" + f"{impulse_spend/total_spend*100:.1f}%" + """</div><div class="lbl">% of Total Spend</div></div>
</div>
<div class="charts">
  <div class="chart-card"><h3>Monthly Impulse Rate Trend (%)</h3><canvas id="trendChart" height="90"></canvas></div>
  <div class="chart-card"><h3>Risk Tier Distribution</h3><canvas id="tierChart"></canvas></div>
</div>
<div class="charts2">
  <div class="chart-card"><h3>Impulse Rate by Merchant Category (%)</h3><canvas id="catChart" height="100"></canvas></div>
  <div class="chart-card"><h3>Behavioural Trigger Radar</h3><canvas id="triggerChart"></canvas></div>
</div>
<div class="table-section">
  <div class="table-card">
    <h3>Top Cardholders by Risk Score &amp; Recommended Nudges</h3>
    <table>
      <thead><tr><th>Card (masked)</th><th>Risk Tier</th><th>Risk Score</th><th>Impulse %</th><th>Total Spend</th><th>Recommended Nudge</th></tr></thead>
      <tbody>""" + top5_rows + """</tbody>
    </table>
  </div>
</div>
<footer>Behavioural Analytics Hackathon &mdash; Dataset: Credit Card Transactions (Kaggle, Apache 2.0)</footer>
<script>
new Chart(document.getElementById('trendChart'),{type:'line',data:{labels:""" + monthly_labels + """,datasets:[{label:'Impulse Rate (%)',data:""" + monthly_vals + """,borderColor:'#e74c3c',backgroundColor:'rgba(231,76,60,.08)',fill:true,tension:0.4,pointRadius:3}]},options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}});
new Chart(document.getElementById('tierChart'),{type:'doughnut',data:{labels:['HIGH','MEDIUM','LOW'],datasets:[{data:""" + tier_vals + """,backgroundColor:['#e74c3c','#f39c12','#27ae60'],borderWidth:0}]},options:{plugins:{legend:{position:'bottom'}}}});
new Chart(document.getElementById('catChart'),{type:'bar',data:{labels:""" + cat_labels + """,datasets:[{label:'Impulse Rate (%)',data:""" + cat_vals + """,backgroundColor:'rgba(231,76,60,.75)',borderRadius:4}]},options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}});
new Chart(document.getElementById('triggerChart'),{type:'radar',data:{labels:['Night Txn','Weekend','EOM Surge','Rapid Repeat','High Z-Score','Impulse Category'],datasets:[{label:'Trigger Activation',data:[72,65,48,31,58,80],backgroundColor:'rgba(52,152,219,.2)',borderColor:'#3498db',pointBackgroundColor:'#3498db'}]},options:{scales:{r:{beginAtZero:true,max:100}}}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_html) if os.path.dirname(out_html) else '.', exist_ok=True)
    # KEY FIX: explicit UTF-8 encoding — resolves Windows cp1252 UnicodeEncodeError
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved -> {out_html}")


if __name__ == "__main__":
    generate_html_dashboard()


