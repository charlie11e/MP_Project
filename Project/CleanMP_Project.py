import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from scipy import stats



## Function for permutation test of 2 vs 3 scanners
def permutation_test(group1, group2):
    def weighted_mean(jobs,hours):
        return jobs.sum() / hours.sum()
            
    obs_diff = (
            weighted_mean(group2["Jobs"], group2["Hours"]) - 
            weighted_mean(group1["Jobs"], group1["Hours"])
        )

    # Combine job-hour pairs into a single array of tuples
    combined = np.array(list(zip(
            np.concatenate([group1["Jobs"].values, group2["Jobs"].values]),
            np.concatenate([group1["Hours"].values, group2["Hours"].values])
    )))

    # Create labels (0 = 2 scanners; 1 = 3 scanners)
    labels = np.array([0] * len(group1) + [1] * len(group2))

    # Permutations
    n_permutations = 10000
    diffs = []

    # Permutation Test
    for _ in range(n_permutations):
        shuffled_labels = np.random.permutation(labels)
    
        g2 = combined[shuffled_labels == 0]
        g3 = combined[shuffled_labels == 1]
    
        g2_jobs, g2_hours = g2[:, 0], g2[:, 1]
        g3_jobs, g3_hours = g3[:, 0], g3[:, 1]
    
        diff = weighted_mean(g3_jobs, g3_hours) - weighted_mean(g2_jobs, g2_hours)
        diffs.append(diff)

    # Calculate two-tailed p-value
    p_value = (np.sum(np.abs(diffs) >= np.abs(obs_diff)) + 1) / (n_permutations + 1)

    return obs_diff, p_value


## Function to filter dataset from user-selected parameters
def filter_data(lc_levels, dph, hours):
    filtered_data = Whole_Day[
        (Whole_Day["LC Level"].isin(lc_levels)) &
        (Whole_Day["Diverts Per Hour"] >= dph[0]) &
        (Whole_Day["Diverts Per Hour"] <= dph[1]) &
        (Whole_Day["Hours"] >= hours[0]) &
        (Whole_Day["Hours"] <= hours[1])
    ]
    return filtered_data


## Function to sum jobs for weighted JPH (by Scanner Count)
def sum_jobs(filtered_data):
    sum_jobs = (
        filtered_data
        .groupby("Scanner Count")["Jobs"]
        .sum()
        .reset_index()
    )
    return sum_jobs

## Function to sum hours for weighted JPH (by Scanner Count)
def sum_hours(filtered_data):
    sum_hours = (
        filtered_data
        .groupby("Scanner Count")["Hours"]
        .sum()
        .reset_index()
    )
    return sum_hours


## Function to merge the jobs and hours into a single dataframe to calculate manual JPH
def avg_jph(sum_jobs,sum_hours):
    avg_jph = pd.merge(sum_jobs, sum_hours, on = "Scanner Count")
    avg_jph["Manual JPH"] = avg_jph["Jobs"] / avg_jph["Hours"]
    avg_jph = avg_jph[["Scanner Count", "Manual JPH"]]

    # Reset index for plotting
    avg_jph.reset_index(drop=True, inplace=True)
    return avg_jph







# IMPORT DATASET
Whole_Day = pd.read_excel("Project/Whole_Day.xlsx")
MP_Diverts = pd.read_excel("Project/MPDiverts.xlsx")



# CLEAN THE DATA

# Drop the "Notes," "Standardized Productivity," and "Alone?" columns
Whole_Day = Whole_Day.drop(columns=["Notes", "MP Lane Diverts (Full Day)", "MP Lane Team?", "Whole Day?"], errors="ignore")

# Group stations into scanner configurations
three_scanner_stations = ["2A", "2B", "3A", "3B"]
three_scanner_lanes = ["2", "3"]

# Create new column "Scanner Count"
Whole_Day["Scanner Count"] = Whole_Day["MP Line and Side"].apply(lambda x: 3 if x in three_scanner_stations else 2)

# Define scanner configuration categories
scanner_map = {
    "Close Side and Middle": ["1A", "1B", "4A", "5A"],
    "Close Side and Far Side": ["4B", "7A", "8B", "9A"],
    "Middle and Far Side": ["5B", "6A", "6B", "7B", "8A", "9B"],
    "Close Side, Middle, and Far Side": ["2A", "2B", "3A"],
    "Middle, Middle, and Far Side": ["3B"]
}

# Function that creates new column and assigns scanner configuration to each row
def assign_position(station):
    for pos, stations in scanner_map.items():
        if station in stations:
            return pos
    return "Unknown"
# Apply the function to each row 
Whole_Day["Scanner Positioning"] = Whole_Day["MP Line and Side"].apply(assign_position)

# Ensure numeric fields are the correct dtype (don't have errors)
Whole_Day["JPH"] = pd.to_numeric(Whole_Day["JPH"], errors='coerce')
Whole_Day["MP Line Diverts"] = pd.to_numeric(Whole_Day["MP Line Diverts"], errors='coerce')
Whole_Day["Hours"] = pd.to_numeric(Whole_Day["Hours"], errors='coerce')
Whole_Day["Diverts Per Hour"] = pd.to_numeric(Whole_Day["Diverts Per Hour"], errors='coerce')

# Drop any rows with missing values
Whole_Day.dropna(subset=["JPH", "MP Line Diverts", "Hours", "LC Level"], inplace=True)
MP_Diverts.dropna(subset=["Lane", "Total Diverts", "Team?",], inplace=True)



# STREAMLIT APP
# Title and description
st.title("Analysis of Manual Palletize Lanes and Scanners")
st.write("This application analyzes the relationship between the number of scanners, their locations, and the number of jobs per hour.")
st.write("This data was collected beginning 6/21/2025 and tracks MP lines from 6:30 AM to 5:15 PM.")
st.info("The goal of this dataset and application is to determine whether three scanners per MP Lane is optimal over only two scanners. \
        Only MP lanes 2 and 3 have three scanners on each side, while the rest of them have two scanners. This application allows the user to filter by LC Level, " \
        "Diverts Per Hour, and Hours Worked to standardize the data and see a more accurate comparison between lanes with three scanners versus lanes with two scanners.")

# Display the first few rows of the dataset
st.subheader("Preview of the Dataset:")
st.dataframe(Whole_Day.head())

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Two vs. Three Scanners", "Alone vs. Teams", "DPH to JPH", "Individual Employee Performance", "Individual Lane and Side Performance", "Total Lane Diverts"])

with tab1:
    st.header('Two vs. Three Scanners Analysis')
    st.write("This section analyzes the difference in productivity between MP Lanes 2 and 3 versus all the other lanes. \
            (MP2 and MP3 are the only lanes with three scanners).")
        
    st.info("Filter the data by LC Level, Diverts Per Hour, and Hours Worked")

    # Multiselect for LC Levels
    lc_level_options = sorted(Whole_Day["LC Level"].dropna().unique())
    selected_lc_levels = st.multiselect("Select LC Levels to Include:", 
                                            options=lc_level_options, 
                                            default=lc_level_options,
                                            key = "lc_levels_count")

    # Filters
    diverts_min = int(Whole_Day["Diverts Per Hour"].min() - 1)
    diverts_max = int(Whole_Day["Diverts Per Hour"].max() + 1)  
    diverts_per_hour = st.slider(
        "Select Range of Diverts Per Hour",
        min_value=diverts_min,
        max_value=diverts_max,
        value=(diverts_min, diverts_max),
        key = "diverts_count"
    )

    hours_min = float(Whole_Day["Hours"].min())
    hours_max = float(Whole_Day["Hours"].max())
    hours_worked = st.slider(
        "Select Range of Hours Worked",
        min_value=hours_min,
        max_value=hours_max,
        value=(hours_min, hours_max),
        key = "hours_count"
        )

    # Use function to filter data according to parameters
    filtered_data = filter_data(selected_lc_levels, diverts_per_hour, hours_worked)

    # Use function to find sum of jobs and horus for weighted JPH
    sum_jobs1 = sum_jobs(filtered_data)
    sum_hours1 = sum_hours(filtered_data)

    # Use function to merge dataframes and find manual JPH
    avg_jph1 = avg_jph(sum_jobs1, sum_hours1)

    # Replace numerical values for display purposes
    avg_jph1["Scanner Count"] = avg_jph1["Scanner Count"].map({
        2: "2 Scanners (Other Lines)",
        3: "3 Scanners (Lines 2 & 3)"
    })

    # Display the number of filtered observations
    st.write("Filtered Row Counts by Scanner Count:")
    st.write(filtered_data["Scanner Count"].value_counts())

    # Plot the bar chart
    fig = px.bar(
        avg_jph1,
        x="Scanner Count",
        y="Manual JPH",
        title="Average JPH by Scanner Count",
        labels={"Manual JPH": "Average JPH", "Scanner Count": "Scanner Count"},
        text_auto=True,
        color="Scanner Count"
    )
    st.plotly_chart(fig)

    # Finding statistical significance (Permutation test)
    st.subheader("Statistical Significance of JPH Difference")

    # Create groups
    group_2_scanners = filtered_data[filtered_data["Scanner Count"] == 2]
    group_3_scanners = filtered_data[filtered_data["Scanner Count"] == 3]

    # Using permutation function to find if the difference between the weighted JPHs is statistically significant
    obs_diff, p_value = permutation_test(group_2_scanners, group_3_scanners)

    # Output results
    st.info(f"Observed Difference in JPH: {obs_diff:.3f} \n\n"
    f"Permutation test p-value: {p_value:.4f}")
    alpha = 0.05
    if p_value < alpha:
        st.success("The difference in JPH between lanes with two scanners and lanes with three scanners is statistically significant (p < 0.05).")
    else:
        st.warning("The difference in JPH between lanes with two scanners and lanes with three scanners statistically significant (p > 0.05).")
       


with tab2:
    st.header("Difference in JPH when Working Alone vs. Working with a Partner")
    st.write("This section analyzes the differences in JPH whenever an associate is working alone or \
            working with a partner. It allows the user to simultaneously view and compare how average JPH changes \
            when working alone or with a partner for lanes with two scanners and lanes with three scanners.")


    st.info("Filter the data by LC Level, Diverts Per Hour, and Hours Worked")

    # Multiselect for LC Levels
    lc_level_options = sorted(Whole_Day["LC Level"].dropna().unique())
    selected_lc_levels = st.multiselect("Select LC Levels to Include:", 
                                            options=lc_level_options, 
                                            default=lc_level_options,
                                            key = "lc_levels_alone")

    # Filters
    diverts_min = int(Whole_Day["Diverts Per Hour"].min() - 1)
    diverts_max = int(Whole_Day["Diverts Per Hour"].max() + 1)  
    diverts_per_hour = st.slider(
        "Select Range of Diverts Per Hour",
        min_value=diverts_min,
        max_value=diverts_max,
        value=(diverts_min, diverts_max),
        key = "diverts_alone"
    )

    hours_min = float(Whole_Day["Hours"].min())
    hours_max = float(Whole_Day["Hours"].max())
    hours_worked = st.slider(
        "Select Range of Hours Worked",
        min_value=hours_min,
        max_value=hours_max,
        value=(hours_min, hours_max),
        key = "hours_alone"
        )

    # Use function to filter data according to parameters
    filtered_data = filter_data(selected_lc_levels, diverts_per_hour, hours_worked)

    # Group filtered data by Alone?
    filtered_alone = filtered_data[filtered_data["Alone?"].isin(["Yes", "Mostly"])]
    filtered_team = filtered_data[filtered_data["Alone?"].isin(["No", "Sometimes"])]

    # Sum jobs and hours alone
    sum_jobs_alone = sum_jobs(filtered_alone)
    sum_hours_alone = sum_hours(filtered_alone)
    # Sum jobs and hours team
    sum_jobs_team = sum_jobs(filtered_team)
    sum_hours_team = sum_hours(filtered_team)

    # Find manual average JPH
    avg_jph_alone = avg_jph(sum_jobs_alone, sum_hours_alone)
    avg_jph_team = avg_jph(sum_jobs_team, sum_hours_team)


    # Replace numerical values with descriptions for display purposes (Team)
    avg_jph_alone["Scanner Count"] = avg_jph_alone["Scanner Count"].map({
        2: "2 Scanners (Other Lines)",
        3: "3 Scanners (Lines 2 & 3)"
    })

    avg_jph_team["Scanner Count"] = avg_jph_team["Scanner Count"].map({
    2: "2 Scanners (Other Lines)",
    3: "3 Scanners (Lines 2 & 3)"
    })

    # Label each dataset
    avg_jph_alone["Dataset"] = "Alone"
    avg_jph_team["Dataset"] = "Team"

    # Combine into one DataFrame
    avg_jph_team_comp = pd.concat([avg_jph_alone, avg_jph_team], ignore_index=True)

    # Display the number of filtered observations
    st.subheader("Scanner Count for AAs Working Alone:")
    st.write("Filtered Row Counts by Scanner Count for AAs Working Alone:")
    st.write(filtered_alone["Scanner Count"].value_counts())
    st.subheader("Scanner Count for AAs Working as a Team:")
    st.write("Filtered Count for AAs Working as a Team:")
    st.write(filtered_team["Scanner Count"].value_counts())

    # Plot
    fig = px.bar(
        avg_jph_team_comp,
        x="Scanner Count",
        y="Manual JPH",
        color="Dataset",
        barmode="group",
        title="Average JPH by Scanner Count: Working Alone vs. Working in a Team",
        labels={"Manual JPH": "Average JPH", "Scanner Count": "Scanner Count"},
        text_auto=True
    )
    st.plotly_chart(fig)

    # Finding statistical significance for AAs working as a team with 2 vs 3 scanners (Permutation test)
    st.subheader("Statistical Significance of JPH Difference for AAs Working as a Team (2 vs 3 scanners)")

    # Create groups
    group_2_scanners_team = filtered_team[filtered_team["Scanner Count"] == 2]
    group_3_scanners_team = filtered_team[filtered_team["Scanner Count"] == 3]

    # Use permutation function
    obs_diff_team, p_value_team = permutation_test(group_2_scanners_team, group_3_scanners_team)

    # Output results
    st.info(f"Observed Difference in JPH: {obs_diff_team:.3f} \n\n"
            f"Permutation test p-value: {p_value_team:.4f}")
    alpha = 0.05
    if p_value_team < alpha:
        st.success("The difference in JPH between lanes with two scanners and lanes with three scanners when AAs are working as a team is statistically significant (p < 0.05).")
    else:
        st.warning("The difference in JPH between lanes with two scanners and lanes with three scanners when AAs are working as a team is not statistically significant (p > 0.05).")


    # Finding statistical significance for AAs working as a team with 2 vs 3 scanners (Permutation test)
    st.subheader("Statistical Significance of JPH Difference for AAs Working Alone (2 vs 3 scanners)")

    # Create groups
    group_2_scanners_alone = filtered_alone[filtered_alone["Scanner Count"] == 2]
    group_3_scanners_alone = filtered_alone[filtered_alone["Scanner Count"] == 3]

    # Permutation function
    obs_diff_alone, p_value_alone = permutation_test(group_2_scanners_alone, group_3_scanners_alone)

    # Output results
    st.info(f"Observed Difference in JPH: {obs_diff_alone:.3f} \n\n"
            f"Permutation test p-value: {p_value_alone:.4f}")
    alpha = 0.05
    if p_value_alone < alpha:
        st.success("The difference in JPH between lanes with two scanners and lanes with three scanners when AAs are working alone is statistically significant (p < 0.05).")
    else:
        st.warning("The difference in JPH between lanes with two scanners and lanes with three scanners when AAs are working alone is not statistically significant (p > 0.05).")




with tab3:
    st.header("Projected JPH Based on Diverts Per Hour")
    st.write("This section analyzes the projected JPH based on how many diverts per hour a lane is receiving. " \
            "This gives a rough estimate of how many diverts a lane should be getting per hour, based on whether an " \
            "AA is working alone or not.")

    st.info("Filter the data by LC Level and Hours Worked")

    # Multiselect for LC Levels
    lc_level_options = sorted(Whole_Day["LC Level"].dropna().unique())
    selected_lc_levels = st.multiselect("Select LC Levels to Include:", 
                                            options=lc_level_options, 
                                            default=lc_level_options,
                                            key = "lc_levels_dph")

    hours_min = float(Whole_Day["Hours"].min())
    hours_max = float(Whole_Day["Hours"].max())
    hours_worked = st.slider(
        "Select Range of Hours Worked",
        min_value=hours_min,
        max_value=hours_max,
        value=(hours_min, hours_max),
        key = "hours_dph"
        )

    # Use function to filter data according to parameters
    filtered_data = Whole_Day[(Whole_Day["LC Level"].isin(selected_lc_levels)&
        (Whole_Day["Hours"] >= hours_worked[0]) &
        (Whole_Day["Hours"] <= hours_worked[1]))]

    # Filter for team and alone
    filtered_alone = filtered_data[filtered_data["Alone?"].isin(["Yes", "Mostly"])]
    filtered_team = filtered_data[filtered_data["Alone?"].isin(["No", "Sometimes"])]

    # Find weighted JPH mean in bins
    bin_edges = list(range(100, 401, 25))  # [100, 125, ..., 400)
    bin_labels = [f"{start}-{end-1}" for start, end in zip(bin_edges[:-1], bin_edges[1:])]

    jph_2_list = []
    jph_3_list = []
    bin_counts_alone = []
    bin_scanner_count_alone = []

    for start, end, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):        
        bin_df = filtered_alone[
            (filtered_alone["Diverts Per Hour"] >= start) &
            ((filtered_alone["Diverts Per Hour"] < end) if end < bin_edges[-1] else (filtered_alone["Diverts Per Hour"] <= end))
        ]
        bin_counts_alone.append(len(bin_df))

        for scanner_count in [2,3]:
            count = (bin_df["Scanner Count"] == scanner_count).sum()
            bin_scanner_count_alone.append({
                "Diverts Per Hour Bin": label,
                "Scanner Count": f"{scanner_count} Scanners",
                "Observation Count": count
            })
    
        jobs = sum_jobs(bin_df)
        hours = sum_hours(bin_df)
        jph_df = avg_jph(jobs, hours)

        # Default NaNs
        jph_2 = np.nan
        jph_3 = np.nan

        if not jph_df.empty:
            for _, row in jph_df.iterrows():
                if row["Scanner Count"] == 2:
                    jph_2 = row["Manual JPH"]
                elif row["Scanner Count"] == 3:
                    jph_3 = row["Manual JPH"]

        jph_2_list.append(jph_2)
        jph_3_list.append(jph_3)


    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        "Diverts Per Hour Bin": bin_labels,
        "2 Scanners": jph_2_list,
        "3 Scanners": jph_3_list
    })

    # Melt into long format for grouped bar chart
    plot_df_melted = plot_df.melt(id_vars="Diverts Per Hour Bin", 
                              value_vars=["2 Scanners", "3 Scanners"],
                              var_name="Scanner Count", 
                              value_name="Manual JPH")


    fig = px.bar(
        plot_df_melted,
        x="Diverts Per Hour Bin",
        y="Manual JPH",
        color="Scanner Count",
        barmode="group",
        title="Weighted JPH by Diverts Per Hour and Scanner Count (Alone)",
        labels={"Diverts Per Hour Bin": "Diverts Per Hour", "Manual JPH": "Weighted Jobs Per Hour"}
    )
    st.plotly_chart(fig, key = 'alone')





    # Plot JPH by DPH (team)
    jph_2_list_team = []
    jph_3_list_team = []
    bin_counts_team = []
    bin_scanner_count_team = []

    for start, end, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):        
        bin_df = filtered_team[
            (filtered_team["Diverts Per Hour"] >= start) &
            ((filtered_team["Diverts Per Hour"] < end) if end < bin_edges[-1] else (filtered_team["Diverts Per Hour"] <= end))
        ]
        bin_counts_team.append(len(bin_df))

        for scanner_count_team in [2,3]:
            count_team = (bin_df["Scanner Count"] == scanner_count_team).sum()
            bin_scanner_count_team.append({
                "Diverts Per Hour Bin": label,
                "Scanner Count": f"{scanner_count_team} Scanners",
                "Observation Count": count_team
            })
    
        jobs = sum_jobs(bin_df)
        hours = sum_hours(bin_df)
        jph_df = avg_jph(jobs, hours)

        # Default NaNs
        jph_2_team = np.nan
        jph_3_team = np.nan

        if not jph_df.empty:
            for _, row in jph_df.iterrows():
                if row["Scanner Count"] == 2:
                    jph_2_team = row["Manual JPH"]
                elif row["Scanner Count"] == 3:
                    jph_3_team = row["Manual JPH"]

        jph_2_list_team.append(jph_2_team)
        jph_3_list_team.append(jph_3_team)


    # Create DataFrame for plotting
    plot_df_team = pd.DataFrame({
        "Diverts Per Hour Bin": bin_labels,
        "2 Scanners": jph_2_list_team,
        "3 Scanners": jph_3_list_team
    })

    # Melt into long format for grouped bar chart
    plot_df_melted_team = plot_df_team.melt(id_vars="Diverts Per Hour Bin", 
                              value_vars=["2 Scanners", "3 Scanners"],
                              var_name="Scanner Count", 
                              value_name="Manual JPH")

    fig_team = px.bar(
        plot_df_melted_team,
        x="Diverts Per Hour Bin",
        y="Manual JPH",
        color="Scanner Count",
        barmode="group",
        title="Weighted JPH by Diverts Per Hour and Scanner Count (Team)",
        labels={"Diverts Per Hour Bin": "Diverts Per Hour", "Manual JPH": "Weighted Jobs Per Hour"}
    )
    st.plotly_chart(fig_team, key = "team")



    # Create DataFrame for dph by bin count (alone)
    bin_scanner_count_alone_df = pd.DataFrame(bin_scanner_count_alone)

    fig = px.bar(
        bin_scanner_count_alone_df,
        x = "Diverts Per Hour Bin",
        y="Observation Count",
        color="Scanner Count",
        barmode="group",
        title="Observation Counts by Diverts Per Hour Bin and Scanner Count (Alone)",
        labels={"Diverts Per Hour Bin": "Diverts Per Hour", "Observation Count": "Observation Count"}
    )
    st.plotly_chart(fig, key="alone_count")

    # Create DataFrame for dph by bin count (team)
    bin_scanner_count_team_df = pd.DataFrame(bin_scanner_count_team)

    fig = px.bar(
        bin_scanner_count_team_df,
        x = "Diverts Per Hour Bin",
        y="Observation Count",
        color="Scanner Count",
        barmode="group",
        title="Observation Counts by Diverts Per Hour Bin and Scanner Count (Team)",
        labels={"Diverts Per Hour Bin": "Diverts Per Hour", "Observation Count": "Observation Count"}
    )
    st.plotly_chart(fig, key="team_count")


    # Average DPH overall
    # Group by scanner count
    scanner_2_all = filtered_data[filtered_data["Scanner Count"] == 2]
    scanner_3_all = filtered_data[filtered_data["Scanner Count"] == 3]

    # Average DPH (overall)
    avg_dph_all_2 = scanner_2_all["Diverts Per Hour"].mean()
    avg_dph_all_3 = scanner_3_all["Diverts Per Hour"].mean()

    # Group by scanner count (alone)
    scanner_2_alone = filtered_alone[filtered_alone["Scanner Count"] == 2]
    scanner_3_alone = filtered_alone[filtered_alone["Scanner Count"] == 3]

    # Average DPH (alone)
    avg_dph_alone_2 = scanner_2_alone["Diverts Per Hour"].mean()
    avg_dph_alone_3 = scanner_3_alone["Diverts Per Hour"].mean()

    # Group by scanner count (team)
    scanner_2_team = filtered_team[filtered_team["Scanner Count"] == 2]
    scanner_3_team = filtered_team[filtered_team["Scanner Count"] == 3]

    # Average DPH (team)
    avg_dph_team_2 = scanner_2_team["Diverts Per Hour"].mean()
    avg_dph_team_3 = scanner_3_team["Diverts Per Hour"].mean()

    # Graph of average DPH team and alone
    # Label each dataset
    diverts_alone = filtered_alone.groupby("Scanner Count")["Diverts Per Hour"].mean().reset_index()
    diverts_team = filtered_team.groupby("Scanner Count")["Diverts Per Hour"].mean().reset_index()

    # Replace numerical values for display purposes
    diverts_alone["Scanner Count"] = diverts_alone["Scanner Count"].map({
        2: "2 Scanners (Other Lines)",
        3: "3 Scanners (Lines 2 & 3)"
    })
    diverts_team["Scanner Count"] = diverts_team["Scanner Count"].map({
        2: "2 Scanners (Other Lines)",
        3: "3 Scanners (Lines 2 & 3)"
    })

    diverts_alone["Dataset"] = "Alone"
    diverts_team["Dataset"] = "Team"

    diverts_comp = pd.concat([diverts_alone, diverts_team], ignore_index=True)

    fig = px.bar(
        diverts_comp,
        x="Scanner Count",
        y="Diverts Per Hour",
        color="Dataset",
        barmode="group",
        title="Average Diverts Per Hour by Scanner Count: Working Alone vs. Working in a Team",
        labels={"Diverts Per Hour": "Average Diverts Per Hour", "Scanner Count": "Scanner Count"},
        text_auto=True
    )
    st.plotly_chart(fig)



    st.info(f"The average number of Diverts Per Hour on a lane with two scanners is {avg_dph_all_2:.3f} \n\n"
            f"The average number of Diverts Per Hour on a lane with three scanners is {avg_dph_all_3:.3f}")

    # Finding statistical significance of difference in average dph on lanes with 2 vs 3 scanners (alone)
    obs_diff_all_dph = avg_dph_all_3 - avg_dph_all_2
    combined_all_dph = np.concatenate([scanner_2_all["Diverts Per Hour"].values, scanner_3_all["Diverts Per Hour"].values])
    labels_all_dph = np.array([0] * len(scanner_2_all) + [1] * len(scanner_3_all))

    # Permutations
    n_permutations = 10000
    diffs_all_dph = []

    # Permutation Test
    for _ in range(n_permutations):
        shuffled_labels_all_dph = np.random.permutation(labels_all_dph)
    
        g2_all = combined_all_dph[shuffled_labels_all_dph == 0]
        g3_all = combined_all_dph[shuffled_labels_all_dph == 1]
        diff_all_dph = g3_all.mean() - g2_all.mean()
        diffs_all_dph.append(diff_all_dph)

    # Calculate two-tailed p-value
    p_value_all_dph = (np.sum(np.abs(diffs_all_dph) >= np.abs(obs_diff_all_dph)) + 1) / (n_permutations + 1)

    # # Output results
    st.info(f"Observed Difference in Diverts Per Hour: {obs_diff_all_dph:.3f} \n\n"
    f"Permutation test p-value: {p_value_all_dph:.4f}")
    alpha = 0.05
    if p_value_all_dph < alpha:
        st.success("The difference in average Diverts Per Hour between lanes with two scanners and lanes with three scanners is statistically significant (p < 0.05).")
    else:
        st.warning("The difference in average Diverts per Hour between lanes with two scanners and lanes with three scanners is not statistically significant (p > 0.05).")
       





    # Finding average DPH when in a team vs when alone
    st.info(f"The average number of Diverts Per Hour on a lane with two scanners when an associate is working alone is {avg_dph_alone_2:.3f} \n\n"
            f"The average number of Diverts Per Hour on a lane with three scanners when an associate is working alone is {avg_dph_alone_3:.3f}")




    # Finding statistical significance of difference in average dph on lanes with 2 vs 3 scanners (alone)
    obs_diff_dph = avg_dph_alone_3 - avg_dph_alone_2
    combined_dph = np.concatenate([scanner_2_alone["Diverts Per Hour"].values, scanner_3_alone["Diverts Per Hour"].values])
    labels_dph = np.array([0] * len(scanner_2_alone) + [1] * len(scanner_3_alone))

    # Permutations
    n_permutations = 10000
    diffs_dph = []

    # Permutation Test
    for _ in range(n_permutations):
        shuffled_labels_dph = np.random.permutation(labels_dph)
    
        g2 = combined_dph[shuffled_labels_dph == 0]
        g3 = combined_dph[shuffled_labels_dph == 1]
        diff_dph = g3.mean() - g2.mean()
        diffs_dph.append(diff_dph)

    # Calculate two-tailed p-value
    p_value_dph = (np.sum(np.abs(diffs_dph) >= np.abs(obs_diff_dph)) + 1) / (n_permutations + 1)

    # # Output results
    st.info(f"Observed Difference in Diverts Per Hour: {obs_diff_dph:.3f} \n\n"
    f"Permutation test p-value: {p_value_dph:.4f}")
    alpha = 0.05
    if p_value_dph < alpha:
        st.success("The difference in average Diverts Per Hour between lanes with two scanners and lanes with three scanners is statistically significant (p < 0.05) for associates working alone.")
    else:
        st.warning("The difference in average Diverts per Hour between lanes with two scanners and lanes with three scanners is not statistically significant (p > 0.05) for associates working alone.")
       


    
    st.info(f"The average number of Diverts Per Hour on a lane with two scanners when an associate is working as a team is {avg_dph_team_2:.3f} \n\n"
            f"The average number of Diverts Per Hour on a lane with three scanners when an associate is working as a team is {avg_dph_team_3:.3f}")



    # Finding statistical significance of difference in average dph on lanes with 2 vs 3 scanners (alone)
    obs_diff_dph_team = avg_dph_team_3 - avg_dph_team_2
    combined_dph_team = np.concatenate([scanner_2_team["Diverts Per Hour"].values, scanner_3_team["Diverts Per Hour"].values])
    labels_dph_team = np.array([0] * len(scanner_2_team) + [1] * len(scanner_3_team))

    # Permutations
    n_permutations = 10000
    diffs_dph_team = []

    # Permutation Test
    for _ in range(n_permutations):
        shuffled_labels_dph_team = np.random.permutation(labels_dph_team)
    
        g2_team = combined_dph_team[shuffled_labels_dph_team == 0]
        g3_team = combined_dph_team[shuffled_labels_dph_team == 1]
        diff_dph_team = g3_team.mean() - g2_team.mean()
        diffs_dph_team.append(diff_dph_team)

    # Calculate two-tailed p-value
    p_value_dph_team = (np.sum(np.abs(diffs_dph_team) >= np.abs(obs_diff_dph_team)) + 1) / (n_permutations + 1)

    # # Output results
    st.info(f"Observed Difference in Diverts Per Hour: {obs_diff_dph_team:.3f} \n\n"
    f"Permutation test p-value: {p_value_dph_team:.4f}")
    alpha = 0.05
    if p_value_dph_team < alpha:
        st.success("The difference in average Diverts Per Hour between lanes with two scanners and lanes with three scanners is statistically significant (p < 0.05) for associates working as a team.")
    else:
        st.warning("The difference in average Diverts Per Hour between lanes with two scanners and lanes with three scanners is not statistically significant (p > 0.05) for associates working as a team.")
       
    




    with tab4:
        st.header("Individual Associate Performance")
        st.write("This section allows the user to analyze an individual associate's performance when working with two versus three scanners.")

        # Select employee
        st.info("Select an associate by entering his or her login")
        employee = st.text_input("Enter Associate Login (Email):", value="", key = "ID")
        if employee:
            filtered_aa = Whole_Day[Whole_Day["ID"] == employee]

            st.info("Filter the data by Diverts Per Hour, and Hours Worked")

            # Filters
            diverts_min = int(filtered_aa["Diverts Per Hour"].min() - 1)
            diverts_max = int(filtered_aa["Diverts Per Hour"].max() + 1)  
            if diverts_min != diverts_max:
                diverts_per_hour = st.slider(
                    "Select Range of Diverts Per Hour",
                    min_value=diverts_min,
                    max_value=diverts_max,
                    value=(diverts_min, diverts_max),
                    key = "diverts_aa")    
            else:
                st.info("There is only one instance of this employee in the dataset, so the Diverts Per Hour slider is not applicable.")

            hours_min = float(filtered_aa["Hours"].min())
            hours_max = float(filtered_aa["Hours"].max())
            if hours_min != hours_max:
                hours_worked = st.slider(
                    "Select Range of Hours Worked",
                    min_value=hours_min,
                    max_value=hours_max,
                    value=(hours_min, hours_max),
                    key = "hours_aa")
            else:
                st.info("There is only one instance of this employee in the dataset, so the Hours Worked slider is not applicable.")

            filtered_data = filtered_aa[
                    (filtered_aa["Diverts Per Hour"] >= diverts_per_hour[0]) &
                    (filtered_aa["Diverts Per Hour"] <= diverts_per_hour[1]) &
                    (filtered_aa["Hours"] >= hours_worked[0]) &
                    (filtered_aa["Hours"] <= hours_worked[1])
                ]
            
            # Group filtered data by Alone?
            filtered_alone_aa = filtered_data[filtered_data["Alone?"].isin(["Yes", "Mostly"])]
            filtered_team_aa = filtered_data[filtered_data["Alone?"].isin(["No", "Sometimes"])]

            # Alone
            sum_jobs_alone_aa = sum_jobs(filtered_alone_aa)
            sum_hours_alone_aa = sum_hours(filtered_alone_aa)
            avg_jph_alone_aa = avg_jph(sum_jobs_alone_aa, sum_hours_alone_aa)

            # Team
            sum_jobs_team_aa = sum_jobs(filtered_team_aa)
            sum_hours_team_aa = sum_hours(filtered_team_aa)
            avg_jph_team_aa = avg_jph(sum_jobs_team_aa, sum_hours_team_aa)

            # Replace numerical values for display purposes
            avg_jph_alone_aa["Scanner Count"] = avg_jph_alone_aa["Scanner Count"].map({
                2: "2 Scanners (Other Lines)",
                3: "3 Scanners (Lines 2 & 3)"
            })

            avg_jph_team_aa["Scanner Count"] = avg_jph_team_aa["Scanner Count"].map({
                2: "2 Scanners (Other Lines)",
                3: "3 Scanners (Lines 2 & 3)"
            })

            # Label each dataset
            avg_jph_alone_aa["Dataset"] = "Alone"
            avg_jph_team_aa["Dataset"] = "Team"

            # Combine into one DataFrame
            avg_jph_comp_aa = pd.concat([avg_jph_alone_aa, avg_jph_team_aa], ignore_index=True)

            # Display the number of filtered observations
            st.subheader(f"Scanner Count for {employee} Working Alone:")
            st.write(f"Filtered Row Counts by Scanner Count for {employee} Working Alone:")
            st.write(filtered_alone_aa["Scanner Count"].value_counts())
            st.subheader(f"Scanner Count for {employee} Working in a Team:")
            st.write(f"Filtered Count for {employee} Working in a Team:")
            st.write(filtered_team_aa["Scanner Count"].value_counts())

            # Plot
            fig = px.bar(
                avg_jph_comp_aa,
                x="Scanner Count",
                y="Manual JPH",
                color="Dataset",
                barmode="group",
                title=f"Average JPH by Scanner Count for {employee}: Working Alone vs. Working in a Team",
                labels={"Manual JPH": "Average JPH", "Scanner Count": "Scanner Count"},
                text_auto=True
            )
            st.plotly_chart(fig)


            # Finding statistical significance for AAs working as a team with 2 vs 3 scanners (Permutation test)
            st.subheader("Statistical Significance of JPH Difference for AAs Working as a Team (2 vs 3 scanners)")

            # Create groups
            group_2_scanners_team_aa = filtered_team_aa[filtered_team_aa["Scanner Count"] == 2]
            group_3_scanners_team_aa = filtered_team_aa[filtered_team_aa["Scanner Count"] == 3]

            # Use permutation function
            obs_diff, p_value = permutation_test(group_2_scanners_team_aa, group_3_scanners_team_aa)

            # Output results
            st.info(f"Observed Difference in JPH: {obs_diff:.3f} \n\n"
                    f"Permutation test p-value: {p_value:.4f}")
            alpha = 0.05
            if p_value < alpha:
                st.success(f"The difference in JPH for between lanes with two scanners and lanes with three scanners when working as a team is statistically significant (p < 0.05) for {employee}.")
            else:
                st.warning(f"The difference in JPH for between lanes with two scanners and lanes with three scanners when working as a team is not statistically significant (p > 0.05) for {employee}.")
        else:
            st.warning("Please enter an Employee ID (Email) to filter the data.")

    with tab5:
        # Comparing Lane Sides
        st.header("Comparing Individual Lane Side Comparison")

        # Select which MP sides to compare
        st.info("Select which two MP Lanes and Sides you wish to compare")
        side_options = sorted(Whole_Day["MP Line and Side"].dropna().unique())
        select_one = st.selectbox("Select the First MP Lane and Side:",
                               options = side_options,
                               index = 0)
        remaining_options = list(filter(lambda item: item != select_one, side_options))
        select_two = st.selectbox("Select the MP Lane and Side to Compare it to:",
                               options = remaining_options,
                               index = 0)
        
        st.info("Filter the data by LC Level, Diverts Per Hour, and Hours Worked")

        # Multiselect for LC Levels
        lc_level_options = sorted(Whole_Day["LC Level"].dropna().unique())
        selected_lc_levels = st.multiselect("Select LC Levels to Include:", 
                                            options=lc_level_options, 
                                            default=lc_level_options,
                                            key = "lc_levels_sides")

        # Filters
        diverts_min = int(Whole_Day["Diverts Per Hour"].min() - 1)
        diverts_max = int(Whole_Day["Diverts Per Hour"].max() + 1)  
        diverts_per_hour = st.slider(
            "Select Range of Diverts Per Hour",
            min_value=diverts_min,
            max_value=diverts_max,
            value=(diverts_min, diverts_max),
            key = "diverts_sides"
        )

        hours_min = float(Whole_Day["Hours"].min())
        hours_max = float(Whole_Day["Hours"].max())
        hours_worked = st.slider(
            "Select Range of Hours Worked",
            min_value=hours_min,
            max_value=hours_max,
            value=(hours_min,hours_max),
            key = "hours_sides"
        )

        # Filter the data given the parameters
        filtered_data = filter_data(selected_lc_levels, diverts_per_hour, hours_worked)

        # Create two groups by selected MP lanes and sides
        selected_lanes = [select_one, select_two]
        filtered_both = filtered_data[
        filtered_data["MP Line and Side"].isin(selected_lanes)]

        # Find average JPH (can't use function because grouping by MP Lane and Side rather than Scanner Count)
        # Sum jobs
        sum_jobs = (
            filtered_both
            .groupby("MP Line and Side")["Jobs"]
            .sum()
            .reset_index()
        )
        # Sum hours
        sum_hours = (
            filtered_both
            .groupby("MP Line and Side")["Hours"]
            .sum()
            .reset_index()
        )
        # Merge into dataframe
        avg_jph = pd.merge(sum_jobs, sum_hours, on="MP Line and Side")
        avg_jph["Manual JPH"] = avg_jph["Jobs"] / avg_jph["Hours"]
        avg_jph = avg_jph[["Manual JPH", "MP Line and Side"]]

        # Reset index for plotting
        avg_jph.reset_index()

        # Display the number of filtered observations
        st.write("Filtered Row Counts by MP Lane and Side:")
        st.write(filtered_both["MP Line and Side"].value_counts())

        # Plot the graph
        fig = px.bar(
            avg_jph,
            x = "MP Line and Side",
            y = "Manual JPH",
            title = f"JPH Comparison of MP{select_one} and MP{select_two}",
            labels = {"Manual JPH": "Average JPH", "MP Line and Side": "MP Line and Side"},
            text_auto = True,
            color = "MP Line and Side"
        )
        st.plotly_chart(fig)

        # Finding statistical significance
        st.subheader(f"Statistical Significance of JPH Difference between MP{select_one} and MP{select_two}.")

        # Create groups
        group_select_one = filtered_data[filtered_data["MP Line and Side"] == select_one]
        group_select_two = filtered_data[filtered_data["MP Line and Side"] == select_two]

        # Use permutation function
        obs_diff, p_value = permutation_test(group_select_one, group_select_two)
            # Output results
        st.info(f"Observed Difference in JPH: {obs_diff:.3f} \n\n"
                f"Permutation test p-value: {p_value:.4f}")
        alpha = 0.05
        if p_value < alpha:
            st.success(f"The difference in JPH between MP{select_one} and MP{select_two} is statistically significant (p < 0.05).")
        else:
            st.warning(f"The difference in JPH between MP{select_one} and MP{select_two} is not statistically significant (p > 0.05).")


    with tab6:
        st.header("Diverts by Lane")

        st.info("Filter the data Total Diverts (this is to remove outliers that unfairly skew the data).")

    
        # Filters
        diverts_min = int(MP_Diverts["Total Diverts"].min() - 1)
        diverts_max = int(MP_Diverts["Total Diverts"].max() + 1)  
        total_diverts = st.slider(
            "Select Range of Total Diverts",
            min_value=diverts_min,
            max_value=diverts_max,
            value=(diverts_min, diverts_max),
            key = "total_diverts"
        )

        filtered_data = MP_Diverts[
        (MP_Diverts["Total Diverts"] >= total_diverts[0]) &
        (MP_Diverts["Total Diverts"] <= total_diverts[1])
        ]

        # Group by lane and calculate average diverts
        avg_jobs = filtered_data.groupby("Lane")["Total Diverts"].mean().reset_index()

        # Sort by lane
        avg_jobs = avg_jobs.sort_values("Lane")

        # Plot
        fig, ax = plt.subplots()
        ax.bar(avg_jobs["Lane"].astype(str), avg_jobs["Total Diverts"], color="skyblue")
        ax.set_xlabel("Lane")
        ax.set_ylabel("Average Diverts")
        ax.set_title("Average Diverts by MP Lane")

        st.pyplot(fig)


        # Graph by alone and in team
        filtered_data_alone = filtered_data[filtered_data["Team?"] == "No"]
        filtered_data_team = filtered_data[filtered_data["Team?"] == "Yes"]

        # Group by lane and find average diverts only when working alone
        avg_jobs_alone = filtered_data_alone.groupby("Lane")["Total Diverts"].mean().reset_index()

        # Sort by lane
        avg_jobs_alone = avg_jobs_alone.sort_values("Lane")

        # Plot
        fig, ax = plt.subplots()
        ax.bar(avg_jobs_alone["Lane"].astype(str), avg_jobs_alone["Total Diverts"], color="skyblue")
        ax.set_xlabel("Lane")
        ax.set_ylabel("Average Diverts")
        ax.set_title("Average Diverts by MP Lane when Working Alone")

        st.pyplot(fig)


        # When working as a team
        avg_jobs_team = filtered_data_team.groupby("Lane")["Total Diverts"].mean().reset_index()

        # Sort by lane
        avg_jobs_team = avg_jobs_team.sort_values("Lane")

        # Plot
        fig, ax = plt.subplots()
        ax.bar(avg_jobs_team["Lane"].astype(str), avg_jobs_team["Total Diverts"], color="skyblue")
        ax.set_xlabel("Lane:")
        ax.set_ylabel("Average Diverts")
        ax.set_title("Average Diverts by MP Lane when Working as a Team")

        st.pyplot(fig)

        
        # Create a graph showing average total diverts by two and three scanner lanes

        # Calculate means for two and three scanner lanes
        mean_2 = filtered_data[filtered_data["Scanner Count"] == 2]["Total Diverts"].mean()
        mean_3 = filtered_data[filtered_data["Scanner Count"] == 3]["Total Diverts"].mean()

        avg_diverts_df = pd.DataFrame({
            "Scanner Count": ["Two Scanners", "Three Scanners"],
            "Average Total Diverts": [mean_2, mean_3]
        })

        fig = px.bar(
            avg_diverts_df,
            x="Scanner Count",
            y="Average Total Diverts",
            title="Average Total Diverts by Scanner Count",
            labels={"Average Total Diverts": "Average Total Diverts", "Scanner Count": "Scanner Count"},
            text_auto=True,
            color="Scanner Count"
        )
        st.plotly_chart(fig)




        group3 = [2,3]
        group2 = [lane for lane in filtered_data["Lane"].unique() if lane not in group3]

        # Calculate means
        mean_group3 = filtered_data[filtered_data["Lane"].isin(group3)]["Total Diverts"].mean()
        mean_group2 = filtered_data[filtered_data["Lane"].isin(group2)]["Total Diverts"].mean()

        st.subheader("Average Total Diverts by Two Scanner and Three Scanner Lanes")
        st.write(f"Average Total Diverts for Two Scanner Lanes: {mean_group2:.2f}")
        st.write(f"Average Total Diverts for Three Scanner Lanes: {mean_group3:.2f}")
        st.info(f"On average, lanes with three scanner stations are receiving {(mean_group3 - mean_group2):.2f} more diverts than two scanner stations.")


        # Calculate means for Alone
        mean_alone_group3 = filtered_data_alone[filtered_data_alone["Lane"].isin(group3)]["Total Diverts"].mean()
        mean_alone_group2 = filtered_data_alone[filtered_data_alone["Lane"].isin(group2)]["Total Diverts"].mean()

        st.subheader("Average Total Diverts When Working Alone")
        st.write(f"Average Total Diverts for Two Scanner Lanes: {mean_alone_group2:.2f}")
        st.write(f"Average Total Diverts for Three Scanner Lanes: {mean_alone_group3:.2f}")
        st.info(f"On average, lanes with three scanner stations are receiving {(mean_alone_group3 - mean_alone_group2):.2f} more diverts than two scanner stations.")

        # Calculate means for Team
        mean_team_group3 = filtered_data_team[filtered_data_team["Lane"].isin(group3)]["Total Diverts"].mean()
        mean_team_group2 = filtered_data_team[filtered_data_team["Lane"].isin(group2)]["Total Diverts"].mean()

        st.subheader("Average Total Diverts When Working as a Team")
        st.write(f"Average Total Diverts for Two Scanner Lanes: {mean_team_group2:.2f}")
        st.write(f"Average Total Diverts for Three Scanner Lanes: {mean_team_group3:.2f}")
        st.info(f"On average, lanes with three scanner stations are receiving {(mean_team_group3 - mean_team_group2):.2f} more diverts than two scanner stations.")




       