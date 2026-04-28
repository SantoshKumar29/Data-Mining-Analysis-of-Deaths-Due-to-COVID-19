import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="COVID-19 Mortality Analytics Dashboard",
    layout="wide"
)

# =========================
# BACKGROUND IMAGE FUNCTION
# =========================
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main content container */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.55);
            padding: 2rem;
            border-radius: 15px;
        }}

        h1, h2, h3, h4, h5, h6, p, span {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.png")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("covid_deaths_cleaned.csv")

df = load_data()

@st.cache_data
def load_arm_rules():
    return pd.read_csv("arm_rules.csv")

@st.cache_data
def load_cluster_profiles():
    return pd.read_csv("cluster_profiles.csv")

@st.cache_resource
def load_log_reg_model():
    with open("log_reg_model.pkl", "rb") as f:
        return pickle.load(f)

# ======================
# Global Sidebar Filters
# ======================

st.sidebar.markdown("### Filters")

state_filter = st.sidebar.multiselect(
    "State",
    sorted(df["state"].dropna().unique())
)

vax_filter = st.sidebar.multiselect(
    "Vaccination Status",
    sorted(df["vax_status"].dropna().unique())
)

comorb_filter = st.sidebar.multiselect(
    "Comorbidity",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

bid_filter = st.sidebar.multiselect(
    "BID",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider(
    "Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

# ======================
# Apply Filters
# ======================

df_f = df.copy()

if state_filter:
    df_f = df_f[df_f["state"].isin(state_filter)]

if vax_filter:
    df_f = df_f[df_f["vax_status"].isin(vax_filter)]

if comorb_filter:
    df_f = df_f[df_f["comorb"].isin(comorb_filter)]

if bid_filter:
    df_f = df_f[df_f["bid"].isin(bid_filter)]

df_f = df_f[
    (df_f["age"] >= age_range[0]) &
    (df_f["age"] <= age_range[1])
]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Select Page")
page = st.sidebar.selectbox(
    "",
    ["Overview", "Demographics", "Vaccination & Health", "Records", "Data Mining Insights"]
)

# =========================
# OVERVIEW PAGE
# =========================
if page == "Overview":
    st.title("COVID-19 Mortality Analytics Dashboard")
    st.caption("Overview page")

    total_cases = len(df_f)
    total_deaths = df_f["bid"].sum()
    death_rate = (total_deaths / total_cases) * 100
    unvax_rate = (df_f["vax_status"] == "Unvaccinated").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", f"{total_cases:,}")
    c2.metric("Total Deaths", f"{total_deaths:,}")
    c3.metric("Death Rate (%)", f"{death_rate:.2f}%")
    c4.metric("Unvaccinated (%)", f"{unvax_rate:.2f}%")

    st.subheader("Dataset Preview")
    st.dataframe(df_f.head(10), use_container_width=True)

# =========================
# DEMOGRAPHICS PAGE
# =========================
if page == "Demographics":
    st.title("Demographic Analysis")
    st.caption("Distribution of COVID-19 cases by age, gender, and state")

    # -----------------------
    # AGE GROUP (PLOTLY)
    # -----------------------
    age_counts = df_f["age_group"].value_counts().sort_index()
    fig_age = px.bar(
        x=age_counts.index,
        y=age_counts.values,
        labels={"x": "Age Group", "y": "Number of Cases"},
        title="Cases by Age Group",
        template="plotly_dark"
    )
    fig_age.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True)
    st.plotly_chart(fig_age, use_container_width=True)

    # -----------------------
    # GENDER
    # -----------------------
    gender_counts = df_f["male"].map({1: "Male", 0: "Female"}).value_counts()
    fig_gender = px.bar(
        x=gender_counts.index,
        y=gender_counts.values,
        labels={"x": "Gender", "y": "Number of Cases"},
        title="Cases by Gender",
        template="plotly_dark"
    )
    fig_gender.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True)
    st.plotly_chart(fig_gender, use_container_width=True)

    # -----------------------
    # STATE
    # -----------------------
    state_counts = df_f["state"].value_counts()
    fig_state = px.bar(
        x=state_counts.index,
        y=state_counts.values,
        labels={"x": "State", "y": "Number of Cases"},
        title="Cases by State",
        template="plotly_dark"
    )
    fig_state.update_layout(
        xaxis_tickangle=-45,
        xaxis_fixedrange=True,
        yaxis_fixedrange=True
    )
    st.plotly_chart(fig_state, use_container_width=True)

# =========================
# VACCINATION & HEALTH PAGE
# =========================

if page == "Vaccination & Health":
    st.title("Vaccination & Health Factors")
    st.caption("Analysis by vaccination status, vaccine brand, comorbidity and BID")

    vax_counts = df_f["vax_status"].value_counts()
    fig_vax = px.bar(
        x=vax_counts.index, y=vax_counts.values,
        labels={"x": "Vaccination Status", "y": "Number of Death Records"},
        title="Deaths by Vaccination Status",
        template="plotly_dark"
    )
    fig_vax.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True)
    st.plotly_chart(fig_vax, use_container_width=True)

    com_counts = df_f["comorb"].map({1:"With Comorbidity", 0:"No Comorbidity"}).value_counts()
    fig_com = px.bar(
        x=com_counts.index, y=com_counts.values,
        labels={"x":"Comorbidity", "y":"Number of Death Records"},
        title="Comorbidity vs No Comorbidity",
        template="plotly_dark"
    )
    st.plotly_chart(fig_com, use_container_width=True)

    bid_counts = df_f["bid"].map({1:"Brought-in-Dead (BID)", 0:"Non-BID"}).value_counts()
    fig_bid = px.bar(
        x=bid_counts.index, y=bid_counts.values,
        labels={"x":"BID Status", "y":"Number of Death Records"},
        title="BID (Brought-in-Dead) Analysis",
        template="plotly_dark"
    )
    st.plotly_chart(fig_bid, use_container_width=True)

    vaccinated = df_f[df_f["brand1"].notna()]
    brand_counts = vaccinated["brand1"].value_counts().head(10)
    fig_brand = px.bar(
        x=brand_counts.index, y=brand_counts.values,
        labels={"x":"Vaccine Brand", "y":"Number of Death Records"},
        title="Deaths by Vaccine Brand (Dose 1) - Top 10",
        template="plotly_dark"
    )
    st.plotly_chart(fig_brand, use_container_width=True)

# =========================
# RECORDS PAGE
# =========================

if page == "Records":
    st.title("Filtered Death Records")
    st.caption("View and export filtered transactional records")

    st.dataframe(df_f, use_container_width=True)

    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered records as CSV",
        data=csv,
        file_name="filtered_covid_deaths.csv",
        mime="text/csv"
    )

    st.divider()

    # =================================================
    # EXPORTED DATA MINING OUTPUTS
    # =================================================
    st.subheader("Exported Data Mining Outputs")

    st.markdown("""
    This section provides access to the **final outputs generated from the data mining notebooks**.
    All models and summaries were trained, validated, and exported offline before being integrated
    into this dashboard.
    """)

    # -------------------------------------------------
    # Logistic Regression Model
    # -------------------------------------------------
    st.markdown("### Classification Model (Logistic Regression)")

    st.markdown("""
    This file contains the trained Logistic Regression model used for
    high-risk (BID) classification.
    """)

    with open("log_reg_model.pkl", "rb") as f:
        st.download_button(
            label="Download Logistic Regression Model (.pkl)",
            data=f,
            file_name="log_reg_model.pkl",
            mime="application/octet-stream"
        )

    st.divider()

    # -------------------------------------------------
    # Association Rule Mining Results
    # -------------------------------------------------
    st.markdown("### Association Rule Mining (ARM) Results")

    st.markdown("""
    This file contains association rules discovered using the Apriori algorithm,
    including support, confidence, and lift values.
    """)

    arm_rules = load_arm_rules()
    st.dataframe(arm_rules.head(10), use_container_width=True)

    arm_csv = arm_rules.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ARM Rules (.csv)",
        data=arm_csv,
        file_name="arm_rules.csv",
        mime="text/csv"
    )

    st.divider()

    # -------------------------------------------------
    # Cluster Profiles
    # -------------------------------------------------
    st.markdown("### Cluster Profiles (K-Means)")

    st.markdown("""
    This file contains the mean feature values for each cluster identified
    using K-Means clustering.
    """)

    cluster_profiles = load_cluster_profiles()
    st.dataframe(cluster_profiles, use_container_width=True)

    cluster_csv = cluster_profiles.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Cluster Profiles (.csv)",
        data=cluster_csv,
        file_name="cluster_profiles.csv",
        mime="text/csv"
    )

    st.markdown("""
    **Note:**  
    These outputs are provided for transparency, reproducibility, and further analysis.
    The dashboard does not retrain models; it only presents finalized results.
    """)


# =========================
# DATA MINING INSIGHTS PAGE
# =========================
if page == "Data Mining Insights":
    st.title("Data Mining Insights")
    st.caption("Insights derived from classification, association rule mining, and clustering")

    # -------------------------------------------------
    # LOAD DATA MINING OUTPUTS
    # -------------------------------------------------
    arm_rules = load_arm_rules()
    cluster_profiles = load_cluster_profiles()
    log_reg_model = load_log_reg_model()

    # =================================================
    # KEY FINDINGS SUMMARY
    # =================================================
    st.markdown("""
    ### Key Findings Summary

    This section consolidates evidence from three data mining techniques (classification, ARM, and clustering).
    Across all methods, risk patterns consistently concentrate around **age**, **comorbidity**, **vaccination status**,
    and **regional variation**. The agreement between supervised and unsupervised results strengthens confidence that
    the identified patterns are not artifacts of a single technique.
    """)

    st.divider()

    # =================================================
    # CLASSIFICATION — LOGISTIC REGRESSION
    # =================================================
    st.subheader("Risk Prediction Factors (Classification)")

    st.markdown("""
    Logistic Regression was used to classify cases into **high-risk (BID)** and **lower-risk** categories.
    The coefficient magnitude indicates how strongly each feature influences the model’s decision boundary.
    This does not mean causation; it indicates which attributes are most informative for distinguishing high-risk cases.
    """)

    # Rebuild encoded feature space (must match training)
    X_encoded = pd.get_dummies(
        df[[
            "age", "male", "malaysian", "comorb",
            "gap_positive_to_death", "days_last_dose_to_death",
            "vax_status", "age_group", "state"
        ]],
        drop_first=True
    )

    coef_df = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Coefficient": log_reg_model.coef_[0]
    })
    coef_df["Absolute Importance"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Absolute Importance", ascending=False)

    st.write("Top features ranked by absolute coefficient magnitude:")
    st.dataframe(coef_df.head(10), use_container_width=True)

    # Visual: feature importance
    fig_coef = px.bar(
        coef_df.head(10),
        x="Absolute Importance",
        y="Feature",
        orientation="h",
        title="Top Risk-Contributing Features (Logistic Regression)",
        template="plotly_dark"
    )
    fig_coef.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_coef, use_container_width=True)

    # Longer insight (classification)
    st.markdown("""
    **What this result means**

    The dominance of **age-group indicators** in the top-ranked features suggests that the model is heavily relying
    on age as a risk stratifier. This aligns with clinical expectations: mortality risk is generally higher in older groups.
    The appearance of **state-related indicators** among the strongest predictors implies that mortality outcomes in the dataset
    vary across regions. This can reflect differences such as population demographics, health system load, outbreak intensity,
    or reporting/health-seeking behaviors. Importantly, the model is not “blaming” a state; it is highlighting that
    **geography is informative** for prediction in this dataset.

    Vaccination-related features appearing among strong predictors indicates that vaccination status carries useful information
    for separating high-risk and lower-risk cases. In practical terms, this supports the interpretation that vaccination status
    is associated with different mortality patterns in the recorded cases. The coefficient sign should be read carefully:
    it depends on encoding and baseline categories, so the most reliable interpretation is the **ranking (importance)** rather than
    over-interpreting a single positive/negative direction.

    **Conclusion from classification**

    Logistic Regression indicates that **age**, **regional variation**, and **vaccination status** are key drivers of risk separation.
    This provides a quantitative basis to prioritize analysis and downstream profiling (ARM/clustering) around these variables.
    """)

    st.divider()

    # =================================================
    # ASSOCIATION RULE MINING (ARM)
    # =================================================
    st.subheader("Association Rule Mining (ARM)")

    st.markdown("""
    Association Rule Mining identifies **frequent co-occurring attribute combinations** without requiring a target label.
    In this study, ARM helps answer: *Which demographic/clinical/vaccination characteristics tend to appear together among records?*
    This complements classification by revealing “risk profiles” in the form of interpretable rules.
    """)

    # Clean rule text for readability
    def clean_rule(x):
        return ", ".join(list(eval(x))) if isinstance(x, str) else x

    arm_display = arm_rules.copy()
    arm_display["antecedents"] = arm_display["antecedents"].apply(clean_rule)
    arm_display["consequents"] = arm_display["consequents"].apply(clean_rule)

    arm_display = arm_display[["antecedents", "consequents", "support", "confidence", "lift"]]
    arm_display = arm_display.sort_values(["lift", "confidence", "support"], ascending=False)

    st.write("Top rules (ranked by lift, then confidence):")
    st.dataframe(arm_display.head(10), use_container_width=True)

    # Visual: lift vs confidence
    fig_arm = px.scatter(
        arm_display.head(25),
        x="confidence",
        y="lift",
        size="support",
        hover_data=["antecedents", "consequents", "support", "confidence", "lift"],
        title="ARM Rule Strength (Lift vs Confidence, bubble size = support)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_arm, use_container_width=True)

    # Longer insight (ARM)
    st.markdown("""
    **What this result means**

    ARM rules with **lift > 1** indicate that the antecedent and consequent occur together more often than expected by chance.
    In your results, several high-lift rules connect demographic groups (e.g., certain age groups and nationality) with vaccination status.
    This suggests that vaccination uptake patterns in the dataset are not uniformly distributed — they vary by population segments.

    High-confidence rules imply that when the antecedent occurs, the consequent occurs frequently as well.
    However, confidence alone can be misleading if the consequent is already very common in the dataset, which is why lift is critical.
    The combination of **high lift + high confidence** provides stronger evidence of a meaningful association.

    **How to use ARM insights**

    ARM does not “predict death”; instead, it identifies **which characteristics cluster together** among records.
    This is useful for creating **interpretable risk narratives** such as:
    - “Certain demographic segments are more likely to be unvaccinated in the dataset.”
    - “Vaccination and comorbidity patterns co-occur with specific age groups.”

    **Conclusion from ARM**

    ARM highlights that risk-related attributes do not act in isolation; they often appear as **multi-factor combinations**.
    These rule-based profiles can guide targeted messaging, outreach, or further analysis by focusing on the most frequent
    and most strongly associated attribute sets.
    """)

    st.divider()

    # =================================================
    # CLUSTERING — K-MEANS
    # =================================================
    st.subheader("Patient Risk Profiles (Clustering)")

    st.markdown("""
    K-Means clustering groups patients into **distinct profiles** based on selected features, without using the BID label.
    This helps identify latent structure: *What “types” of cases exist in the data, even before labeling them as high or low risk?*
    """)

    st.write("Cluster profile table (mean feature values per cluster):")
    st.dataframe(cluster_profiles, use_container_width=True)

    # Visual: age vs comorb with vaccination bubble
    if all(col in cluster_profiles.columns for col in ["age", "comorb", "vax_status_Unvaccinated"]):
        fig_cluster = px.scatter(
            cluster_profiles,
            x="age",
            y="comorb",
            size="vax_status_Unvaccinated",
            color=cluster_profiles.index.astype(str),
            title="Cluster Profiles by Age, Comorbidity, and Unvaccinated Proportion",
            template="plotly_dark",
            labels={"color": "Cluster"}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    # Longer insight (clustering)
    st.markdown("""
    **What this result means**

    Clustering reveals that the dataset is not homogeneous; it contains multiple sub-populations.
    One cluster typically shows a combination of **higher average age**, **higher comorbidity prevalence**, and **higher unvaccinated proportion**.
    That pattern is consistent with a clinically plausible high-risk profile.

    Another cluster may represent elderly but vaccinated individuals, suggesting that vaccination status helps separate subgroups even within older ages.
    This is important: it indicates that age alone does not fully explain the structure of the data — vaccination status and comorbidity
    meaningfully differentiate patient profiles.

    **Why clustering adds value beyond classification**

    - Classification tells you which features help distinguish risk labels.
    - Clustering tells you what “types of cases” exist **even without labels**.
    When a high-risk-like cluster aligns with the classification insights (age/comorbidity/vaccination), it strengthens the credibility of both methods.

    **Conclusion from clustering**

    Clustering supports the existence of distinct mortality-related profiles, particularly separating:
    - older + comorbid + unvaccinated (highest risk profile),
    - elderly but vaccinated (risk mitigated subgroup),
    - and other mixed profiles.
    This provides a structured basis for profiling and segmentation.
    """)

    st.divider()

    # =================================================
    # CROSS-TECHNIQUE CONCLUSION
    # =================================================
    st.markdown("""
    ### Overall Conclusion (Cross-Technique)

    The three techniques provide complementary evidence:

    - **Logistic Regression (Classification)** quantifies the strongest predictors of high-risk outcomes, with age and region emerging as dominant signals,
      and vaccination/comorbidity contributing substantial information.
    - **ARM** reveals that these variables often appear as combinations (profiles), not individually, indicating multi-factor patterns across records.
    - **Clustering** confirms that the dataset naturally separates into distinct profiles, including a clear elderly/comorbid/unvaccinated subgroup.

    Taken together, the results suggest that COVID-19 mortality risk patterns in this dataset can be meaningfully described through
    **demographic stratification (age), clinical vulnerability (comorbidity), vaccination coverage, and regional differences**.
    The convergence across multiple methods strengthens confidence that these are robust, actionable patterns rather than one-model artifacts.
    """)