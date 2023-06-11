import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the data
df = pd.read_csv('./healthcare-dataset-stroke-data.csv')

# Set page config
st.set_page_config(
    page_title="Brain Stroke Analysis Dashboard",
    layout="wide"
)

# Define width and height
height = 500
width = 400

# Title
col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.image('./strokegraphic.jpg', use_column_width = True)
with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""<div style='display: flex; justify-content: center; align-items: center;'><p style='color: black; font-size: 35px; font-weight: bold; text-align: center; width: 100%;'>Brain Stroke Analysis Dashboard</p></div>""", unsafe_allow_html=True)
with col3:
    st.image('./stroke2.jpg', use_column_width = True)

st.markdown("""
    <div style='font-size: 24px; font-weight: bold;'>
    Welcome to the Brain Stroke Risk Analysis Dashboard!<br>This interactive tool allows you to explore a dataset of patients with and without stroke, 
    with the aim of identifying key risk factors and trends.
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Demographic and Lifestyle Analysis", "Medical Risk Factors Analysis", "Integrated Analysis"])

with tab1:
    st.header("Overview")

    # Show Basic Statistics
    st.subheader('Basic Statistics')

    column1, column2, column3, column4, column5 = st.columns(5)
    total_patients = df.shape[0]
    total_strokes = df[df['stroke'] == 1].shape[0]
    avg_age = round(df['age'].mean(), 2)
    avg_bmi = round(df['bmi'].mean(), 2)
    avg_glucose = round(df['avg_glucose_level'].mean(), 2)
    column1.metric('Total number of patients', total_patients)
    column2.metric('Total number of stroke cases', total_strokes)
    column3.metric('Average age of patients', avg_age)
    column4.metric('Average BMI of patients', avg_bmi)
    column5.metric('Overall Average Glucose Level of patients', avg_glucose)
    
    st.subheader('Key Demographics')

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Create pie chart for gender distribution
        gender_counts = df['gender'].value_counts()
        fig_gender = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values, hole=.3, marker_colors=['darkblue', 'lightblue'])])
        fig_gender.update_layout(title_text='Gender Distribution', showlegend=True,autosize=True, 
                        title_font=dict(size=18), 
                        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        legend=dict(font=dict(size=14)))
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        # Create pie chart for Smoking Status Distribution
        smoke_values = df['smoking_status'].value_counts()
        fig_smoke = go.Figure(data=[go.Pie(labels=smoke_values.index, values=smoke_values.values, hole=.3,marker_colors=['darkblue', 'lightblue', 'royalblue','steelblue'])])
        fig_smoke.update_layout(title_text='Smoking Status Distribution',showlegend=True,autosize=True,
                        title_font=dict(size=18), 
                        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        legend=dict(font=dict(size=14)))
        st.plotly_chart(fig_smoke, use_container_width=True)

    # Create two columns
    col3, col4 = st.columns(2)

    with col3:
        # Create histogram for age distribution
        bins = pd.cut(df['age'], [0, 20, 40, 60, 80, 100])
        age_counts = df['age'].groupby(bins).count()   
        bins= [0,30,50,70,120] # Define age groups
        labels = ['Under 30','30-50','50-70','Over 70']
        df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)   
        fig_age = px.histogram(df, x="AgeGroup") # Create histogram for age group distribution
        # Set color for each trace (bar)
        for i, trace in enumerate(fig_age.data):
            trace.marker.color = 'darkblue'  # You can choose the color you want here

        fig_age.update_layout(title_text='Age Group Distribution', xaxis_title='Age Group', yaxis_title='Count',
                            showlegend=True, autosize=True,
                            title_font=dict(size=18), 
                            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            legend=dict(font=dict(size=14)))
                
        fig_age.update_xaxes(categoryorder='array', categoryarray= labels) # Specify the category order

        st.plotly_chart(fig_age, use_container_width=True)

    with col4:
        # Create historgram for Avg Glucose Level Distribution
        fig_glucose = go.Figure(data=[go.Histogram(x=df['avg_glucose_level'], nbinsx=40,marker_color='darkblue')]) 

        fig_glucose.add_shape(
            type="line",
            x0=70, y0=0,
            x1=70, y1=1,
            yref="paper",
            line=dict(
                color="Green",
                width=3,
                dash="dashdot")
            ) # Add line for lower threshold

        fig_glucose.add_shape(
            type="line",        x0=100, y0=0,
            x1=100, y1=1,
            yref="paper",
            line=dict(
                color="Red",
                width=3,
                dash="dashdot")
            ) # Add line for upper threshold

        fig_glucose.update_layout(title_text='Avg Glucose Level Distribution', 
                                xaxis_title='Avg Glucose Level', 
                                yaxis_title='Count', 
                                showlegend=False,autosize=True, 
                                title_font=dict(size=18), 
                                xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                                yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
                                legend=dict(font=dict(size=14)))
        st.plotly_chart(fig_glucose, use_container_width=True)

    
    with st.container():
        st.markdown("<hr style='border: 1px solid black'>", unsafe_allow_html=True)
        st.text("This dashboard was created by Dany El Feghali.")
        st.text("Disclaimer: The information presented here is for informational purposes only. Always consult a healthcare professional for medical advice.")
        
with tab2:
    st.header("Demographic and Lifestyle Analysis")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Define age groups
        bins= [0,30,50,70,120]
        labels = ['Under 30','30-50','50-70','Over 70']
        df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        # Count of strokes by age group and gender
        stroke_by_age_group_gender = df[df['stroke'] == 1].groupby(['AgeGroup', 'gender']).size().unstack()

        # Plot the stroke count by age group and gender
        fig1 = px.bar(stroke_by_age_group_gender.reset_index(), 
                    x='AgeGroup', 
                    y=['Male', 'Female'], 
                    title='Stroke Prevalence by Age Group and Gender',
                    labels={'value':'Stroke Prevalence', 'variable':'Gender', 'AgeGroup':'Age Group'},
                    barmode='group', color_discrete_map={'Male':'darkblue', 'Female':'lightblue'})
        fig1.update_layout(showlegend=True,autosize=True, 
                        title_font=dict(size=18), 
                        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        legend=dict(font=dict(size=14)))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Group data by stroke and residence type
        residence_stroke = df.groupby('stroke')['Residence_type'].value_counts(normalize=True).unstack()

        # Create a bar chart for stroke incidence by residence type
        fig2 = go.Figure()
        color_map = {'Rural': 'darkblue', 'Urban': 'lightblue'}  # Create a color map for the residence types

        for residence in residence_stroke.columns:
            fig2.add_trace(go.Bar(name=residence, y=['No Stroke', 'Stroke'], x=residence_stroke[residence].values, orientation='h',marker_color=color_map[residence]))

        fig2.update_layout(barmode='group', title_text='Stroke Incidence by Residence Type', 
                        yaxis_title='Stroke Incidence', xaxis_title='Proportion of Patients', 
                        title_x=0, title_font=dict(size=18), 
                        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        legend=dict(font=dict(size=14)),
                        autosize=True, margin=dict(r=135))
        st.plotly_chart(fig2, use_container_width=True)
  
    # Create two columns
    col3, col4 = st.columns(2)

    with col3:
        # Group data by marital status and stroke
        marital_status_stroke = df.groupby('stroke')['ever_married'].value_counts(normalize=True).unstack()
        
        # Bar chart for marital status
        fig1 = go.Figure(data=[
            go.Bar(name='Not Married', x=['No Stroke', 'Stroke'], y=marital_status_stroke['No'].values, marker_color='lightblue'),
            go.Bar(name='Married', x=['No Stroke', 'Stroke'], y=marital_status_stroke['Yes'].values, marker_color='darkblue')
        ])

        # Change the bar mode and layout
        fig1.update_layout(barmode='stack', bargroupgap=0.3, title_text='Marital Status by Stroke Incidence', 
                        xaxis_title='Stroke Incidence', yaxis_title='Proportion of Patients', 
                        title_x=0, title_font=dict(size=18), 
                        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        legend=dict(font=dict(size=14)),
                        autosize=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col4:
        color_palette = px.colors.qualitative.Set1

        # Group data by work type and stroke
        work_type_stroke = df.groupby('stroke')['work_type'].value_counts(normalize=True).unstack()

        # Bar chart for work type
        fig2 = go.Figure()
        for idx, work_type in enumerate(work_type_stroke.columns):
            fig2.add_trace(go.Bar(name=work_type, 
                                x=['No Stroke', 'Stroke'], 
                                y=work_type_stroke[work_type].values,
                                marker_color=color_palette[idx % len(color_palette)]))  # Specify color for each work type

        fig2.update_layout(barmode='stack', 
                        bargroupgap=0.3, 
                        title_text='Work Type by Stroke Incidence', 
                        xaxis_title='Stroke Incidence', 
                        yaxis_title='Proportion of Patients', 
                        title_x=0, 
                        title_font=dict(size=18), 
                        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        legend=dict(font=dict(size=14)),
                        autosize=True)
        st.plotly_chart(fig2, use_container_width=True)

    with st.container():
        # Create treemap for smoking status and stroke 
        proportions_df = df.groupby(['stroke', 'smoking_status']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index(name='proportion')

        proportions_df.columns = ['stroke', 'smoking_status', 'proportion']       
        proportions_df['values'] = proportions_df['proportion'] # Set 'proportion' as values

        proportions_df['stroke'] = proportions_df['stroke'].map({0: 'No Stroke', 1: 'Stroke'})  # Map 0 and 1 to 'No Stroke' and 'Stroke'

        fig = px.treemap(proportions_df, path=['stroke', 'smoking_status'], 
                        values='values', 
                        color='proportion',
                        color_continuous_scale=px.colors.qualitative.Set1,  # Change color scale to different shades of blue
                        title='<b>Treemap of Smoking Proportions within Stroke and Non-Stroke Patients</b>',
                        hover_data=['proportion'])  # Include proportion data in the hover data

        fig.update_traces(texttemplate='%{label}<br>%{customdata[0]:.2f}%', textfont_size=20)

        fig.update_layout(
            title={
                'text': '<b>Treemap of Smoking Proportions within Stroke and Non-Stroke Patients</b>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font':dict(size=18)
                }
            )
        # Show the figure
        st.plotly_chart(fig, use_container_width=True)

    with st.container():
        st.markdown("<hr style='border: 1px solid black'>", unsafe_allow_html=True)
        st.text("This dashboard was created by Dany El Feghali.")
        st.text("Disclaimer: The information presented here is for informational purposes only. Always consult a healthcare professional for medical advice.")

with tab3:
    st.header("Medical Risk Factors Analysis")
    
    # Create two columns
    col1, col2 = st.columns(2)

    # Display the figures side by side
    with col1:
        # Calculate proportions for hypertension
        hypertension_stroke = df.groupby('stroke')['hypertension'].value_counts(normalize=True).unstack()
        # Bar chart for hypertension
        fig1 = go.Figure(data=[
            go.Bar(name='No Hypertension', x=['No Stroke', 'Stroke'], y=hypertension_stroke[0].values, marker_color='lightblue'),
            go.Bar(name='Hypertension', x=['No Stroke', 'Stroke'], y=hypertension_stroke[1].values, marker_color='darkblue')
        ])
        # Change the bar mode and layout
        fig1.update_layout(barmode='group', title_text='Hypertension Status by Stroke Incidence', 
                        xaxis_title='Stroke Incidence', yaxis_title='Proportion of Patients', 
                        title_x=0, title_font=dict(size=18), 
                        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                        legend=dict(font=dict(size=14)),autosize=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        heart_disease_stroke = df.groupby('stroke')['heart_disease'].value_counts(normalize=True).unstack()
        # Bar chart for heart disease
        fig2 = go.Figure(data=[
            go.Bar(name='No Heart Disease', x=['No Stroke', 'Stroke'], y=heart_disease_stroke[0].values, marker_color='lightblue'),
            go.Bar(name='Heart Disease', x=['No Stroke', 'Stroke'], y=heart_disease_stroke[1].values, marker_color='darkblue')
        ])

        fig2.update_layout(barmode='group', title_text='Heart Disease Status by Stroke Incidence', 
                xaxis_title='Stroke Incidence', yaxis_title='Proportion of Patients', 
                title_x=0, title_font=dict(size=18), 
                xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                legend=dict(font=dict(size=14)),autosize=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Create two columns
    col3, col4 = st.columns(2)

    # Display the figures side by side
    with col3:
        # Create bins for average glucose level
        bins = [0, 100, 150, 200, df['avg_glucose_level'].max()]
        labels = ['<100', '100-150', '150-200', '>200']

        df['avg_glucose_level_binned'] = pd.cut(df['avg_glucose_level'], bins=bins, labels=labels, include_lowest=True)

        # Calculate proportions separately for Stroke and No Stroke cases
        stroke_glucose = df[df['stroke'] == 1]['avg_glucose_level_binned'].value_counts(normalize=True).loc[labels]
        no_stroke_glucose = df[df['stroke'] == 0]['avg_glucose_level_binned'].value_counts(normalize=True).loc[labels]

        # Bar chart
        fig3 = go.Figure(data=[
            go.Bar(name='No Stroke', x=no_stroke_glucose.index, y=no_stroke_glucose.values, marker_color='lightblue'),
            go.Bar(name='Stroke', x=stroke_glucose.index, y=stroke_glucose.values, marker_color='darkblue')
        ])

        # Update the layout
        fig3.update_layout(barmode='group', title_text='Average Glucose Level by Stroke Incidence', 
                            xaxis_title='Average Glucose Level', yaxis_title='Proportion of Patients', 
                            title_x=0, title_font=dict(size=18), 
                            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            legend=dict(font=dict(size=14)),
                            autosize=True)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Create bins for BMI
        bins = [0, 18.5, 25, 30, df['bmi'].max()]
        labels = ['<18.5', '18.5-25', '25-30', '>30']

        df['bmi_binned'] = pd.cut(df['bmi'], bins=bins, labels=labels, include_lowest=True)

        # Calculate proportions separately for Stroke and No Stroke cases
        stroke_bmi = df[df['stroke'] == 1]['bmi_binned'].value_counts(normalize=True).loc[labels]
        no_stroke_bmi = df[df['stroke'] == 0]['bmi_binned'].value_counts(normalize=True).loc[labels]

        # Bar chart
        fig4 = go.Figure(data=[
            go.Bar(name='No Stroke', x=no_stroke_bmi.index, y=no_stroke_bmi.values, marker_color='lightblue'),
            go.Bar(name='Stroke', x=stroke_bmi.index, y=stroke_bmi.values, marker_color='darkblue')
        ])

        # Update the layout
        fig4.update_layout(barmode='group',title_text='BMI by Stroke Incidence', 
                            xaxis_title='BMI', yaxis_title='Proportion of Patients', 
                            title_x=0, title_font=dict(size=18), 
                            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            legend=dict(font=dict(size=14)),
                            autosize=True)
        st.plotly_chart(fig4, use_container_width=True)

    # Filters
    st.markdown("<h3 style='text-align: center; background-color: #f2f2f2; padding: 10px;'>Custom Filters</h2>", unsafe_allow_html=True)

    # Create two columns
    col5, col6 = st.columns(2)

    # Use the first column for filters and sliders
    with col5:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        # Filter for hypertension
        hypertension_mapping = {0: 'No', 1: 'Yes'}
        df['hypertension_label'] = df['heart_disease'].map(hypertension_mapping)
        hypertension_options = df['hypertension_label'].unique().tolist()
        selected_hypertension = st.multiselect('Hypertension', hypertension_options, default=hypertension_options)
        selected_hypertension = [1 if item == "Yes" else 0 for item in selected_hypertension]

        # Filter for heart disease
        heart_disease_mapping = {0: 'No', 1: 'Yes'}
        df['heart_disease_label'] = df['heart_disease'].map(heart_disease_mapping)
        heart_disease_options = df['heart_disease_label'].unique().tolist()
        selected_heart_disease = st.multiselect('Heart Disease', heart_disease_options, default=heart_disease_options)
        selected_heart_disease = [1 if item == "Yes" else 0 for item in selected_heart_disease]

        # Slider for BMI
        min_bmi, max_bmi = float(df['bmi'].min()), float(df['bmi'].max())
        selected_bmi = st.slider('BMI', min_value=min_bmi, max_value=max_bmi, value=(min_bmi, max_bmi))

        # Slider for average glucose level
        min_glucose, max_glucose = float(df['avg_glucose_level'].min()), float(df['avg_glucose_level'].max())
        selected_glucose = st.slider('Average Glucose Level', min_value=min_glucose, max_value=max_glucose, value=(min_glucose, max_glucose))

        # Filter data based on selections
        df_flt = df[df['hypertension'].isin(selected_hypertension)]
        df_flt = df_flt[df_flt['heart_disease'].isin(selected_heart_disease)]
        df_flt = df_flt[(df_flt['bmi'] >= selected_bmi[0]) & (df_flt['bmi'] <= selected_bmi[1])]
        df_flt = df_flt[(df_flt['avg_glucose_level'] >= selected_glucose[0]) & (df_flt['avg_glucose_level'] <= selected_glucose[1])]

    # Use the second column for resulting visualization
    with col6:
        # Total counts of stroke and no-stroke cases in the whole dataset
        total_stroke_counts = df['stroke'].value_counts()

        # Counts of stroke and no-stroke cases in the filtered data
        flt_stroke_counts = df_flt['stroke'].value_counts()

        # Calculate proportions
        stroke_proportions = flt_stroke_counts / total_stroke_counts

        # Adjust the index
        stroke_proportions.index = ['No Stroke' if idx == 0 else 'Stroke' for idx in stroke_proportions.index]

        # Create a plotly figure
        fig = go.Figure(data=[
            go.Bar(name='Stroke', x=stroke_proportions.index, y=stroke_proportions.values, marker_color=['darkblue', 'red'])
        ])

        # Remove the legend
        fig.update_layout(showlegend=False, bargroupgap=0.3)

        # Display the figure
        st.plotly_chart(fig)

    with st.container():
        st.markdown("<hr style='border: 1px solid black'>", unsafe_allow_html=True)
        st.text("This dashboard was created by Dany El Feghali.")
        st.text("Disclaimer: The information presented here is for informational purposes only. Always consult a healthcare professional for medical advice.")

with tab4:
    st.header("Integrated Analysis")

    # Let the user choose the risk factor
    risk_factor = st.selectbox('Choose a Risk Factor:', ['hypertension', 'heart_disease'])

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the figures side by side
    with col1:
        # Create age groups
        bins = [0, 30, 40, 50, 60, 70, 80, np.inf]
        names = ['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
        df['age_group'] = pd.cut(df['age'], bins, labels=names)

        # Create a copy of the Original DataFrame
        df1 = df.copy()

        df1[risk_factor] = df1[risk_factor].map({0: f'No {risk_factor.capitalize()}', 1: f'{risk_factor.capitalize()}'})

        # Create a grouped dataframe for stroke incidence by age group and risk factor
        grouped_df = df1.groupby(['age_group', risk_factor])['stroke'].mean().reset_index()

        # Create a bar plot
        fig1 = px.bar(grouped_df, x='age_group', y='stroke', 
                    color=risk_factor, 
                    barmode='stack', 
                    labels={'stroke': 'Stroke Incidence', 
                            'age_group': 'Age Groups',
                            risk_factor: risk_factor.capitalize()},
                    title=f'Stroke Incidence by Age and {risk_factor.capitalize()} Status',
                    color_discrete_sequence=['darkblue', 'lightblue'])

        fig1.update_layout(title_x=0, title_font=dict(size=18), 
                            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            legend=dict(font=dict(size=14)),
                            autosize=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Create another copy of the DataFrame for the second figure
        df2 = df.copy()

        df2[risk_factor] = df2[risk_factor].map({0: f'No {risk_factor.capitalize()}', 1: f'{risk_factor.capitalize()}'})
        
        df2['bmi_category'] = df2['bmi'].apply(lambda x: 'High (>25)' if x >= 25 else 'Low (<25)')
        df2['stroke_status'] = df2['stroke'].map({0: 'No Stroke', 1: 'Stroke'})

        # Calculate counts of each group
        stroke_bmi_hypertension_counts = df2.groupby(['stroke_status', risk_factor, 'bmi_category']).size()

        # Calculate proportions by dividing by total counts within each 'stroke_status' group
        stroke_bmi_hypertension = stroke_bmi_hypertension_counts / stroke_bmi_hypertension_counts.groupby(level=0).transform(sum)

        # Reset index
        stroke_bmi_hypertension = stroke_bmi_hypertension.reset_index()

        # Rename columns for clarity
        stroke_bmi_hypertension.columns = ['stroke_status', risk_factor, 'bmi_category', 'proportion']

        # Create the plot
        fig2 = px.bar(stroke_bmi_hypertension, 
                        x='stroke_status', 
                        y='proportion', 
                        color=risk_factor,
                        facet_row='bmi_category',
                        facet_row_spacing=0.1,
                        labels={'proportion': 'Proportion of Patients', 
                                'stroke_status': 'Stroke Incidence',
                                risk_factor: risk_factor.capitalize()+' Status',
                                'bmi_category': 'BMI Category'},
                        title=f'Proportion of {risk_factor.capitalize()} Status and BMI Category by Stroke Incidence',
                        barmode='group',
                        color_discrete_sequence=['darkblue', 'lightblue'])

        fig2.update_layout(title_x=0, title_font=dict(size=18),
                            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            legend=dict(font=dict(size=14)),
                            autosize=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Create two columns
    col3, col4 = st.columns(2)

    with col3:
        grouped_df = df.groupby(['smoking_status', 'gender', risk_factor])['stroke'].mean().reset_index()

        # Filter for rows where the risk factor is 1 (yes)
        grouped_df = grouped_df[grouped_df[risk_factor] == 1]

        # Separate data by gender
        male_df = grouped_df[grouped_df['gender']=='Male']
        female_df = grouped_df[grouped_df['gender']=='Female']

        # Create a diverging bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=male_df['smoking_status'],
            x=male_df['stroke'],
            orientation='h',
            name='Male',
            marker=dict(color='darkblue'),
            hovertemplate = 'Stroke: %{x:.2%}<extra></extra>'  # Custom hover template
        ))

        fig.add_trace(go.Bar(
            y=female_df['smoking_status'],
            x=-female_df['stroke'],  # Negate female values for symmetry
            orientation='h',
            name='Female',
            marker=dict(color='lightblue'),
            hovertemplate = 'Stroke: %{customdata:.2%}<extra></extra>',  # Custom hover template
            customdata=female_df['stroke']
        ))

        # Modify x-axis ticks to show absolute values
        fig.update_xaxes(tickformat=".%", 
                        tickvals=[-0.4, -0.35, -0.3, 0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                        ticktext=['40%','35%','30%','25%','20%','15%','10%', '5%', '0%','5%','10%','15%','20%', '25%','30%','35%','40%'])

        fig.update_layout(
            title_text=f'Influence of {risk_factor.capitalize()} on Stroke Incidence Differentially Impacts Male and Female Smokers',
            title_x=0, 
            title_font=dict(size=18),
            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
            legend=dict(font=dict(size=14)),
            autosize=True,
            barmode='relative',
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:
        grouped_df = df.groupby(['work_type', 'ever_married', risk_factor])['stroke'].mean().reset_index()

        # Filter for rows where the risk factor is 1 (yes)
        grouped_df = grouped_df[grouped_df[risk_factor] == 1]

        # Separate data by marital status
        married_df = grouped_df[grouped_df['ever_married']=='Yes']
        not_married_df = grouped_df[grouped_df['ever_married']=='No']
            
        # Ensure that both dataframes have the same work types
        married_df = married_df[married_df['work_type'].isin(not_married_df['work_type'])]
        not_married_df = not_married_df[not_married_df['work_type'].isin(married_df['work_type'])]

        # Create a diverging bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=married_df['work_type'],
            x=married_df['stroke'],
            orientation='h',
            name='Married',
            marker=dict(color='darkblue'),
            hovertemplate = 'Stroke: %{x:.2%}<extra></extra>'  # Custom hover template
        ))

        fig.add_trace(go.Bar(
            y=not_married_df['work_type'],
            x=-not_married_df['stroke'],  # Negate values for symmetry
            orientation='h',
            name='Not Married',
            marker=dict(color='lightblue'),
            hovertemplate = 'Stroke: %{customdata:.2%}<extra></extra>',  # Custom hover template
            customdata=not_married_df['stroke']
        ))

        # Modify x-axis ticks to show absolute values
        fig.update_xaxes(tickformat=".%", 
                        tickvals=[-0.4, -0.35, -0.3, 0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                        ticktext=['40%','35%','30%','25%','20%','15%','10%', '5%', '0%','5%','10%','15%','20%', '25%','30%','35%','40%'])

        fig.update_layout(
            title_text=f'Influence of {risk_factor.capitalize()} on Stroke Incidence Across Work Types and Marital Status',
            title_x=0, 
            title_font=dict(size=18),
            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
            legend=dict(font=dict(size=14)),
            autosize=True,
            barmode='relative',
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create two columns
    col5, col6 = st.columns(2)

    # Display the figures side by side
    with col5:       
        # Filter the dataframe based on the stroke condition
        df_no_stroke = df[df['stroke'] == 0]
        df_stroke = df[df['stroke'] == 1]

        # Create a scatter plot for patients without stroke
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_no_stroke['age'], 
            y=df_no_stroke['avg_glucose_level'], 
            mode='markers', 
            name='No Stroke', 
            marker=dict(
                color='lightblue',
                size=6
            )
        ))

        # Create a scatter plot for patients with stroke
        fig.add_trace(go.Scatter(
            x=df_stroke['age'], 
            y=df_stroke['avg_glucose_level'], 
            mode='markers', 
            name='Stroke', 
            marker=dict(
                color='red',
                size=6
            )
        ))

        # Set the title and labels
        fig.update_layout(
            title='Age vs Average Glucose Level', 
            xaxis_title='Age', 
            yaxis_title='Average Glucose Level',title_x=0, 
            title_font=dict(size=18), 
            xaxis=dict(title_font=dict(size=16), 
            tickfont=dict(size=14)), 
            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col6:
        df['bmi_category'] = df['bmi'].apply(lambda x: 'High (>25)' if x >= 25 else 'Low (<25)')
        df['stroke_status'] = df['stroke'].map({0: 'No Stroke', 1: 'Stroke'})
        # Define bins and labels for glucose level
        bins = [0, 90, 110, 125, 140, np.inf]
        names = ['(<90)', '(90-110)', '(110-125)', '(125-140)', '(>140)']

        # Create glucose_category based on average_glucose_level
        df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins, labels=names)

        # Calculate counts of each group
        stroke_bmi_glucose_counts = df.groupby(['stroke_status', 'glucose_category', 'bmi_category']).size()

        # Calculate proportions by dividing by total counts within each 'stroke' group
        stroke_bmi_glucose = stroke_bmi_glucose_counts / stroke_bmi_glucose_counts.groupby(level=0).transform(sum)

        # Reset index
        stroke_bmi_glucose = stroke_bmi_glucose.reset_index()

        # Rename columns for clarity
        stroke_bmi_glucose.columns = ['stroke_status', 'glucose_category', 'bmi_category', 'proportion']

        # Create the plot
        fig = px.bar(stroke_bmi_glucose, 
                        x='stroke_status', 
                        y='proportion', 
                        color='glucose_category',
                        facet_row='bmi_category',
                        facet_row_spacing=0.1,
                        labels={'proportion': 'Proportion of Patients', 
                                'stroke_status': 'Stroke Incidence',
                                'glucose_category': 'Glucose Level Category',
                                'bmi_category': 'BMI Category'},
                        title='Proportion of Glucose Level and BMI Category by Stroke Incidence',
                        barmode='group',
                        color_discrete_sequence=px.colors.qualitative.Set1)
            
        fig.update_layout(title_x=0, title_font=dict(size=18), 
                            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)), 
                            legend=dict(font=dict(size=14)),
                            autosize=True)
        st.plotly_chart(fig, use_container_width=True)

    with st.container():
        st.markdown("<hr style='border: 1px solid black'>", unsafe_allow_html=True)
        st.text("This dashboard was created by Dany El Feghali.")
        st.text("Disclaimer: The information presented here is for informational purposes only. Always consult a healthcare professional for medical advice.")