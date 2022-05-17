"""
This module is used to implement a gas fuel price
vision of Brazil over the years. It also uses logging,
testing and clean code practices.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def plot_and_show(datasets_, labels_, colors_, sup_title):
    """
    Receives a list of datasets, labels and colors and returns the
    figure generated
    Args:
    datasets: DataFrame[]
    labels: str[]
    colors: str[]
    suptitle: str
    Returns:
    fig: Figure
    """
    try:
        assert len(datasets_) > 0
        assert len(labels_) > 0
        assert len(colors_) > 0
    except AssertionError:
        logging.error("ERROR: None of the lists can be empty.")

    try:
        assert isinstance(sup_title, str)
    except AssertionError:
        logging.error("ERROR: Title and Suptitle must be strings.")

    logging.info(
        "SUCCESS: Plotting %i datasets related to '%s'",
        len(datasets_), sup_title)

    fig, a_x = plt.subplots(figsize=(12, 6))

    j = 0
    for i in datasets_:
        line_plot_gas_over_time(i, labels_[j], colors_[j], a_x)
        j += 1

    # Adding titles
    # ax.suptitle.set_text(sup_title, weight='bold')
    a_x.set_title(sup_title, fontdict={'weight': 'bold', 'fontsize': 16})

    # plt.xlabel('Year')
    plt.ylabel('Mean Price (R$/L)')

    # Adding footer
    a_x.annotate('©MLOPS' + ' ' * 70 +
                 'Source: National Agency of Petroleum, '
                 + 'Natural Gas and Bio fuels (ANP in Portuguese)',
                 (0, 0), (-20, -50), xycoords='axes fraction',
                 textcoords='offset points', va='top',
                 backgroundcolor='#4d4d4d', color='#f0f0f0')

    # Increasing grid opacity
    a_x.grid(alpha=0.5)
    a_x.legend()
    return fig
    # ax.show()


def config_logging(path):
    """
    Configures the logging instance
    Args:
    path: str
    Returns:
    None
    """
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        filemode='w',
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def line_plot_gas_over_time(dataset, label, color, a_x):
    """
    Receives a dataframe and plots a line graph
    Args:
    dataset: DataFrame
    label: str
    color: str

    Returns:
    None
    """
    rolling_mean = 200

    try:
        assert isinstance(label, str)
    except AssertionError:
        logging.error("ERROR: Label must be a string.")

    logging.info("SUCCESS: plotting gas prices for %s", label)
    a_x.plot(dataset['final_date'],
             dataset['resell_mean'].rolling(rolling_mean).mean(),
             label=label,
             color=color,
             linewidth=2)


def clean_dataset(dataset):
    """
    Receives a DataFrame and cleans it for the problem
    Args:
    dataset: DataFrame
    Returns:
    cleaned_data: DataFrame
    """
    cleaned_data = dataset.copy()
    try:
        cleaned_data.columns = [
            "initial_date",
            "final_date",
            "region",
            "state",
            "product",
            "gas_stations_number",
            "measurement_unit",
            "resell_mean",
            "resell_std",
            "resell_min_price",
            "resell_max_price",
            "resell_mean_margin",
            "resell_variant_coef",
            "mean_price_dist",
            "std_dist",
            "min_price_dist",
            "max_price_dist",
            "variation_dist_coef"]

        cleaned_data['final_date'] = pd.to_datetime(cleaned_data['final_date'])
        cleaned_data.drop("initial_date", axis=1, inplace=True)
    except KeyError:
        logging.error(
            "ERROR: final_date or initial_date not found on this DataFrame")
    except ValueError:
        logging.error("ERROR: There are less columns than the specified.")

    # Removing initial_date, since it won't be used
    cleaned_data.sort_values('final_date', inplace=True)
    cleaned_data.reset_index(drop=True, inplace=True)

    # Removing accents
    cleaned_data['product'].replace(
        {'ÓLEO DIESEL': 'OLEO DIESEL'}, inplace=True)
    cleaned_data['product'].replace(
        {'ÓLEO DIESEL S10': 'OLEO DIESEL S10'}, inplace=True)

    logging.info("SUCCESS: dataframe was cleaned successfully.")

    return cleaned_data


def declare_and_execute(question):
    """
    This is the main function, reads the dataset, creates variables and plots things.
    Args:
    question: int
    Returns:
    fig: Figure
    """
    # Getting dataset and cleaning
    data = pd.read_csv('gas_2004-2021.tsv', sep='\t', lineterminator='\n')
    config_logging('./results2.log')

    data = clean_dataset(data)

    # Question 1
    data_ethanol = data[data['product'] == "ETANOL HIDRATADO"]
    data_gasoline = data[data['product'] == "GASOLINA COMUM"]
    data_diesel = data[data['product'] == "OLEO DIESEL"]
    data_glp = data[data['product'] == "GLP"]

    # Question 2
    gasoline_centro_oeste = data_gasoline[data_gasoline['region']
                                          == 'CENTRO OESTE']
    gasoline_nordeste = data_gasoline[data_gasoline['region'] == 'NORDESTE']
    gasoline_sul = data_gasoline[data_gasoline['region'] == 'SUL']
    gasoline_sudeste = data_gasoline[data_gasoline['region'] == 'SUDESTE']
    gasoline_norte = data_gasoline[data_gasoline['region'] == 'NORTE']

    # Question 3
    glp_lula = data_glp.copy()[(data_glp['final_date'].dt.year >= 2004)
                               & (data_glp['final_date'].dt.year <= 2010)]
    glp_dilma = data_glp.copy()[(data_glp['final_date'].dt.year >= 2010)
                                & (data_glp['final_date'].dt.year <= 2016)]
    glp_temer = data_glp.copy()[(data_glp['final_date'].dt.year >= 2016)
                                & (data_glp['final_date'].dt.year <= 2019)]
    glp_bolsonaro = data_glp.copy()[data_glp['final_date'].dt.year >= 2019]

    if question == 1:
        return select_question_and_plot(
            question, [data_gasoline, data_diesel, data_ethanol])
    if question == 2:
        return select_question_and_plot(question, [
            gasoline_centro_oeste,
            gasoline_nordeste,
            gasoline_norte,
            gasoline_sul,
            gasoline_sudeste])
    if question == 3:
        return select_question_and_plot(
            question, [glp_lula, glp_dilma, glp_temer, glp_bolsonaro])

    return None


def select_question_and_plot(question, data):
    """
    Receives a question number and plots its corresponding question
    Args:
    question: int
    datasets DataFrame[]
    """
    try:
        if question == 1:
            assert len(data) == 3
        elif question == 2:
            assert len(data) == 5
        elif question == 3:
            assert len(data) == 4
    except AssertionError:
        logging.error("ERROR: the dataset must have more elements")

    if question == 1:
        # Plotting Question 1 Results
        labels = ['Gasoline', 'Diesel', 'Ethanol']
        colors = ['c', 'm', 'g']
        sup_title = 'Gasoline, Diesel and Ethanol evolution in Brazil from 2004 to 2021'
        datasets = data
        return plot_and_show(datasets, labels, colors, sup_title)

    if question == 2:
        # Plotting Question 2 Results
        labels = ['Centro-Oeste', 'Nordeste', 'Norte', 'Sul', 'Sudeste']
        colors = ['orange', 'g', 'b', 'r', 'aquamarine']
        sup_title = 'Gasoline evolution in each brazilian region from 2004 to 2021'
        datasets = data
        return plot_and_show(datasets, labels, colors, sup_title)

    if question == 3:
        # Plotting Question 3 Results
        labels = ['Lula', 'Dilma', 'Temer', 'Bolsonaro']
        colors = ['r', 'orange', 'blue', 'green']
        sup_title = 'GLP Price in every Government from 2004 to 2022'
        datasets = data
        return plot_and_show(datasets, labels, colors, sup_title)
    return None


def generate_streamlit():
    """
    Generates a simple dashboard.
    """
    st.write(
        """
        # Fuel Prices from 2004 to 2021 in Brazil
        Hi, my name is Joao and I am a student at UFRN. I'm developing this
        dashboard on Streamlit as a part of the first grade in a university
        subject called MLOps, which is being developed by Ivanovitch Silva.

        For this exercise, I chose to analyze some fuel data from Brazil. In the
        past few years, we had a huge increase in fuel prices due to several factors.
        In this analysis, I made 3 questions and tried to answer them. Hope you enjoy!
        """)

    option = st.selectbox(
        'Which question do you wanna see?',
        ('Question 1', 'Question 2', 'Question 3'))

    if option in 'Question 1':
        st.pyplot(declare_and_execute(1))
    elif option in 'Question 2':
        st.pyplot(declare_and_execute(2))
    elif option in 'Question 3':
        st.pyplot(declare_and_execute(3))


generate_streamlit()
