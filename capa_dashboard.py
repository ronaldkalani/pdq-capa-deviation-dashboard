import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------------
# 📥 Load FAERS datasets
# ----------------------------
demo = pd.read_csv('DEMO20Q4.txt', sep='$', encoding='latin1')
drug = pd.read_csv('DRUG20Q4.txt', sep='$', encoding='latin1')
reac = pd.read_csv('REAC20Q4.txt', sep='$', encoding='latin1')
ther = pd.read_csv('THER20Q4.txt', sep='$', encoding='latin1')
indi = pd.read_csv('INDI20Q4.txt', sep='$', encoding='latin1')
outc = pd.read_csv('OUTC20Q4.txt', sep='$', encoding='latin1')

# ----------------------------
# 📉 Deviation Tracking
# ----------------------------
ther['start_dt'] = pd.to_datetime(ther['start_dt'], errors='coerce')
ther['end_dt'] = pd.to_datetime(ther['end_dt'], errors='coerce')
ther['deviation_flag'] = ther['end_dt'] < ther['start_dt']
deviation_count = ther['deviation_flag'].sum()
deviation_records = ther[ther['deviation_flag'] == True][['primaryid', 'caseid', 'start_dt', 'end_dt']]

# ----------------------------
# 🚨 CAPA Flag Simulation
# ----------------------------
adr_pairs = pd.merge(drug[['primaryid', 'drugname']], reac[['primaryid', 'pt']], on='primaryid', how='inner')
capa_flags = adr_pairs.groupby(['drugname', 'pt']).size().reset_index(name='count')
capa_flags = capa_flags[capa_flags['count'] >= 5].sort_values(by='count', ascending=False)

# ----------------------------
# ⚠️ Mismatch Detection
# ----------------------------
indi_reac = pd.merge(indi[['primaryid', 'indi_pt']], reac[['primaryid', 'pt']], on='primaryid', how='inner')
indi_reac['mismatch'] = indi_reac['indi_pt'].str.lower() != indi_reac['pt'].str.lower()
mismatch_count = indi_reac['mismatch'].sum()
mismatch_examples = indi_reac[indi_reac['mismatch'] == True][['primaryid', 'indi_pt', 'pt']].head(10)

# ----------------------------
# 🧠 Predictive Modeling – ADR Seriousness
# ----------------------------
model_status = "Model not executed yet"
report_df = pd.DataFrame()
audit_summary = {}

try:
    model_df = pd.merge(demo[['primaryid', 'age', 'sex', 'wt']], outc[['primaryid', 'outc_cod']], on='primaryid')
    model_df['sex'] = model_df['sex'].map({'M': 0, 'F': 1})
    model_df['outc_cod'] = model_df['outc_cod'].apply(lambda x: 1 if x == 'DE' else 0)
    model_df = model_df.dropna()

    X = model_df[['age', 'sex', 'wt']]
    y = model_df['outc_cod']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index()

    model_status = "✅ Model trained successfully"
    audit_summary = {
        "Total Records": len(model_df),
        "Records Used in Modeling": len(X),
        "Features Used": list(X.columns),
        "Target": "ADR Seriousness: Death (1) vs Non-Death (0)",
        "Model Used": "RandomForestClassifier",
        "Test Accuracy": round(clf.score(X_test, y_test), 3)
    }

except Exception as e:
    model_status = f"❌ Model failed: {str(e)}"

# ----------------------------
# 🎛️ Streamlit Sidebar Selector
# ----------------------------
st.set_page_config(page_title="PDQ Dashboard", layout="wide")
st.sidebar.title("📌 Dashboard Sections")
section = st.sidebar.selectbox(
    "Select a report section:",
    [
        "Deviation Tracking",
        "CAPA Candidates",
        "Mismatch Analysis",
        "Predictive Modeling",
        "Audit Summary Report"
    ]
)

st.title("🔬 PDQ Real-Time CAPA, Deviation & Risk Dashboard")

# ----------------------------
# Display Sections by Selection
# ----------------------------
if section == "Deviation Tracking":
    st.header("📉 Deviation Tracking (THER20Q4)")
    st.metric("Therapy Deviations Found", deviation_count)
    st.dataframe(deviation_records, use_container_width=True)

elif section == "CAPA Candidates":
    st.header("🚨 CAPA Flagged Drug-Reaction Pairs (DRUG + REAC)")
    st.write("Drug-reaction combinations with ≥ 5 reports may indicate repeat quality issues.")
    st.dataframe(capa_flags, use_container_width=True)

elif section == "Mismatch Analysis":
    st.header("⚠️ Indication vs Reaction Mismatch (INDI vs REAC)")
    st.metric("Mismatch Cases", mismatch_count)
    st.dataframe(mismatch_examples, use_container_width=True)

elif section == "Predictive Modeling":
    st.header("🧠 Predicting ADR Seriousness from Demographics")
    st.write(model_status)
    if not report_df.empty:
        st.dataframe(report_df, use_container_width=True)

elif section == "Audit Summary Report":
    st.header("📋 Model Audit Summary")
    st.json(audit_summary)

