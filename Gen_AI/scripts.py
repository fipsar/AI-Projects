import mysql.connector
import json
# import boto3
import pandas as pd
import matplotlib.pyplot as plt
# from IPython.display import HTML
import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import numpy as np
import openai
import os
from dotenv import load_dotenv
from flask import session
import psycopg2
import re
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.express as px
import colorsys
import tiktoken
import logging
import logging_config


load_dotenv()

def count_tokens(text, encoding_name="cl100k_base"):
    """
    Counts the number of tokens in a given text using a specified encoding.
    Parameters:
    - text (str): The text to tokenize.
    - encoding_name (str): The name of the encoding to use (default is "cl100k_base").
    Returns:
    - int: The number of tokens in the text.
    """
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        return token_count
    except ValueError as e:
        return f"Error: {e}"

def mysql_connection(query):
    try:
        conn = mysql.connector.connect(
            user='root',
            password='1234',
            host='localhost',
            port='3306',
            database=session.get('selected_database', '')
        )
        cursor = conn.cursor()
        cursor.execute(query)
        
        if query.strip().lower().startswith(('update', 'delete', 'insert')):
            conn.commit()
            result = f"Query executed successfully: {cursor.rowcount} rows affected."
        else:
            result = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return result
    
    except mysql.connector.Error as error:
        return f"Error: {error}"

def postgresql_connection(query):
    try:
        conn = psycopg2.connect(
            user='postgres',
            password='4563',
            host='localhost',
            port='5432',
            database=session.get('selected_database', '')
        )
        cursor = conn.cursor()
        cursor.execute(query)
        
        if query.strip().lower().startswith(('update', 'delete', 'insert')):
            conn.commit()
            result = f"Query executed successfully: {cursor.rowcount} rows affected."
        else:
            result = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return result
    
    except psycopg2.Error as error:
        return f"Error: {error}"
def connection(query, selected_datasource):
    if selected_datasource == 'mysql':
        return mysql_connection(query)
    elif selected_datasource == 'postgresql':
        return postgresql_connection(query)
    else:
        raise ValueError(f"Unsupported data source selected: {selected_datasource}")



def get_specific_tables_metadata(connection,selected_datasource, selected_database, schema_name, tables_df, num_samples=1):
    metadata = ""
    working_tables = tables_df[tables_df['Flag'] == 1]
    
    for _, table_row in working_tables.iterrows():
        table_name = table_row['Table_name']
        metadata += f"\nMetadata for table {selected_database}.{schema_name}.{table_name}:\n"
        sql = f"SHOW COLUMNS FROM {schema_name}.{table_name}"
        
        result = connection(sql,selected_datasource)
        create_table_statement = f"CREATE TABLE {schema_name}.{table_name} ("
        for row in result:
            if selected_datasource == 'mysql':
                if selected_database == 'healthcare':
                    column_name = row[0]
                    data_type = row[1]
                    create_table_statement += f"\n {column_name} {data_type},"
                if selected_database == 'pharma':
                    column_name = row[0]
                    data_type = row[0]
                    create_table_statement += f"\n {column_name} {data_type},"
        create_table_statement = create_table_statement.rstrip(',') + "\n);"
        metadata += create_table_statement + "\n"

        sample_sql = f"SELECT * FROM {schema_name}.{table_name} LIMIT {num_samples};"
        sample_rows = connection(sample_sql,selected_datasource)

        metadata += f"\nSample rows from {table_name} table:\n"
        for row in sample_rows:
            metadata += str(row) + "\n"
    
    return metadata


def meta_data(selected_database, selected_datasource):
    schema_files = {
        'healthcare': 'health_data_schema.csv',
        'pharma': 'pharma_database_schema.csv',
        'chinook': 'chinook_database_schema.csv'
    }
    
    schema_dataframes = {
        key: pd.read_csv(file) for key, file in schema_files.items()
    }
    
    if selected_datasource == 'mysql':
        if selected_database in ['healthcare', 'pharma']:
            schema_name = selected_database
            df = schema_dataframes[selected_database]
        else:
            raise ValueError(f"Unknown database selected for mysql: {selected_database}")
    elif selected_datasource == 'postgresql':
        if selected_database == 'chinook':
            schema_name = 'chinook'
            df = schema_dataframes['chinook']
        else:
            raise ValueError(f"Unknown database selected for chinook: {selected_database}")
    else:
        raise ValueError(f"Unknown datasource selected: {selected_datasource}")
    
    table_info = get_specific_tables_metadata(connection, selected_datasource, selected_database, schema_name, df)
    return table_info, schema_name


# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_openai_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Specify the model
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.5,
        top_p=0.9
    )
    
    # Extracting the generated text from the response
    model_results = response.choices[0].message['content'].strip()
    return model_results


def question_classification(Input):
    prompt_for_classify_input = f"""Given a INPUT you should classify it under one of the following question type: {Input}.

Below are the examples of the Input and its question type.

Examples:

Input: How many patients are with insurance coverage?; question type: The question requires response in natural language.
Input: How many rows are present in the doctors table?; question type: The question requires response in natural language.
Input: What is the license number of doctor Carly ?; question type: The question requires response in natural language.
Input: Which doctor prescribed this for the patient Cris ?; question type: The question requires response in natural language.
Input: What medications are prescribed for patients with high blood pressure?; question type: The question requires response in natural language.
Input: Give me the contact details of pharmacies in Springfield, IL?; question type: The question requires response in natural language.
Input: If the License Number is 112 change it as 'NULL' ; question type: The question requires response in natural language.
Input: In doctor table if the License Number is 112 change it as 'NULL' ; question type: The question requires response in natural language.
Input: Visualize the distribution of patients' genders in a pie chart.?; question type: The question requires response in visualization.
Input: Generate a bar chart showing the number of prescriptions per medication.?; question type: The question requires response in visualization.
Input: Generate a scatter plot of patients' ages against their insurance coverage.?; question type: The question requires response in visualization.
Input: Provide the list of table with the names, genders, and dates of birth of all patients.?; question type: The question requires response in table format.
Input: Give me the list of prescriptions along with patient names and medications.?; question type: The question requires response in table format.
Input: List out the appointments with patient names, doctor names, and appointment dates.?; question type: The question requires response in table format.
Input: List the appointments with patient names, doctor names, and appointment dates.?; question type: The question requires response in table format.
Input: {Input}


<instructions>
If any Input having a word 'chart', 'plot' then it should be classified under the question type: The question requires response in visualization.
If any Input having a word 'table', 'list' then it should be classified under the question type: The question requires response in table format. 
Based on the user input you need to anlayze the question and classify its question type whether it should be answered in natural language or in table format or in visualization. 
Do not strictly write the Input question and Instruction in the output. only give the classified question type.
</instructions>
"""
    input_for_classification = count_tokens(prompt_for_classify_input)
    # print(input_for_classification,'input_for_classification')
    logging.info(f"{input_for_classification} - input_for_classification")
    classification = call_openai_gpt(prompt_for_classify_input)
    output_classification = count_tokens(classification)
    # print(output_classification,'output_classification')
    logging.info(f"{output_classification} - output_classification")
    return classification.strip()

# def sql_query_generator(Input, selected_database,selected_datasource,mode):
#     if mode == 'user_mode':
#         meta_data_content, schema_name = meta_data(selected_database,selected_datasource)

#         prompt_sql_data = f"""
#         You are a SQL expert.
#         You are tasked to generate only SQL statement from the instruction provided. Never give out any additional sentences other than the SQL query requested of you.

#         **instructions**
#         Understanding the input question and referencing the database schema which has column names and its data types and sample rows of each tables,
#         generate a SQL statement that represents the question irrespective of whether the question is related to table format/visualization output just concentrate on giving out SQL query related to that question.

#         *Write the SQL query between <SQL></SQL>.

#         *Only provide SQL Query. Do not give any explanations for the generated SQL Query

#         *For this problem you can use the following table schema:

#         **table_schema**
#         {meta_data_content}

#         *Please provide the SQL Query for this question:

#         **question**
#         {Input}

#         **Additional Hints:**

#         *Provide only the SQL query, with no additional comments.
#         *The SQL query must follow the provided table schema.
#         *If the question is unrelated to the database or you cannot generate a relevant SQL statement, respond with "sorry, I am unable to help."
#         *Do not create fabricated answers.
#         *Respond with only the SQL query.
#         """

#         generated_sql = call_openai_gpt(prompt_sql_data)

#         cleaned_sql = generated_sql.replace("<SQL>", "").replace("</SQL>", "")

#         if f"{schema_name}." not in cleaned_sql:
#             cleaned_sql = cleaned_sql.replace('FROM', f'FROM {schema_name}.')
#             cleaned_sql = cleaned_sql.replace('JOIN', f'JOIN {schema_name}.')

#         query = f"{cleaned_sql}".strip()
#         col_name = []
#         select_part = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE)
#         if select_part:
#             column_string = select_part.group(1)
#             for col in column_string.split(','):
#                 # Extract alias if it exists, otherwise use the column name
#                 match = re.search(r'\s+AS\s+(\w+)', col, re.IGNORECASE)
#                 if match:
#                     col_name.append(match.group(1).strip())
#                 else:
#                     col_name.append(col.strip())
#         else:
#             print("Warning: Unable to extract column names from the SQL query.")

#         results = connection(query,selected_datasource)
#         return query, results, col_name, mode
    
#     elif mode == 'developer_mode':
#         meta_data_content, schema_name = meta_data(selected_database,selected_datasource)
#         prompt_sql_data = f"""
#         You are a SQL expert.
#         You are tasked to generate only SQL statement from the instruction provided. Never give out any additional sentences other than the SQL query requested of you.

#         **instructions**
#         Understanding the input question and referencing the database schema which has column names and its data types and sample rows of each tables,
#         generate a SQL statement that represents the question irrespective of whether the question is related to table format/visualization output just concentrate on giving out SQL query related to that question.

#         *Write the SQL query between <SQL></SQL>.

#         *Only provide SQL Query. Do not give any explanations for the generated SQL Query

#         *For this problem you can use the following table schema:

#         **table_schema**
#         {meta_data_content}

#         *Please provide the SQL Query for this question:

#         **question**
#         {Input}

#         **Additional Hints:**

#         *Provide only the SQL query, with no additional comments.
#         *The SQL query must follow the provided table schema.
#         *If the question is unrelated to the database or you cannot generate a relevant SQL statement, respond with "sorry, I am unable to help."
#         *Do not create fabricated answers.
#         *Respond with only the SQL query.
#         """

#         generated_sql = call_openai_gpt(prompt_sql_data)

#         cleaned_sql = generated_sql.replace("<SQL>", "").replace("</SQL>", "")

#         if f"{schema_name}." not in cleaned_sql:
#             cleaned_sql = cleaned_sql.replace('FROM', f'FROM {schema_name}.')
#             cleaned_sql = cleaned_sql.replace('JOIN', f'JOIN {schema_name}.')

#         query = f"{cleaned_sql}".strip()

#         col_name = []
#         select_part = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE)
#         if select_part:
#             column_string = select_part.group(1)
#             for col in column_string.split(','):
#                 # Extract alias if it exists, otherwise use the column name
#                 match = re.search(r'\s+AS\s+(\w+)', col, re.IGNORECASE)
#                 if match:
#                     col_name.append(match.group(1).strip())
#                 else:
#                     col_name.append(col.strip())
#         else:
#             print("Warning: Unable to extract column names from the SQL query.")

#         return query


def sql_query_generator(Input, selected_database, selected_datasource, mode):
    print('Entering_sql_query_generator_function')
    meta_data_content, schema_name = meta_data(selected_database, selected_datasource)
    
    prompt_sql_data = f"""
    You are a SQL expert.
    You are tasked to generate only SQL statement from the instruction provided. Never give out any additional sentences other than the SQL query requested of you.

    **instructions**
    Understanding the input question and referencing the database schema which has column names and its data types and sample rows of each tables,
    generate a SQL statement that represents the question irrespective of whether the question is related to table format/visualization output just concentrate on giving out SQL query related to that question.

    *Write the SQL query between <SQL></SQL>.

    *Only provide SQL Query. Do not give any explanations for the generated SQL Query

    *For this problem you can use the following table schema:

    **table_schema**
    {meta_data_content}

    *Please provide the SQL Query for this question:

    **question**
    {Input}

    **Additional Hints:**

    *Provide only the SQL query, with no additional comments.
    *The SQL query must follow the provided table schema.
    *If the question is unrelated to the database or you cannot generate a relevant SQL statement, respond with "sorry, I am unable to help."
    *Do not create fabricated answers.
    *Respond with only the SQL query.
    """
    input_token_generated_sql = count_tokens(prompt_sql_data)
    # print(input_token_generated_sql,'input_token_generated_sql')
    logging.info(f"{input_token_generated_sql} - input_token_generated_sql")
    generated_sql = call_openai_gpt(prompt_sql_data)
    output_token_generated_sql = count_tokens(generated_sql)
    # print(output_token_generated_sql,'output_token_generated_sql')
    logging.info(f"{output_token_generated_sql} - output_token_generated_sql")
    cleaned_sql = generated_sql.replace("<SQL>", "").replace("</SQL>", "").strip()

    if f"{schema_name}." not in cleaned_sql:
        cleaned_sql = cleaned_sql.replace('FROM', f'FROM {schema_name}.')
        cleaned_sql = cleaned_sql.replace('JOIN', f'JOIN {schema_name}.')
    
    query = cleaned_sql
    col_name = extract_column_names(query)

    if mode == 'user_mode':
        print('Sql_query_generator_for_user_mode')
        results = connection(query, selected_datasource)
        if len(col_name) != len(results[0]):
            col_name = [f'column_{i+1}' for i in range(len(results[0]))]
        return query, results, col_name, mode
    elif mode == 'developer_mode':
        print('Sql_query_generator_for_dev_mode')
        return query,col_name

def extract_column_names(query):
    col_name = []
    select_part = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE)
    if select_part:
        column_string = select_part.group(1)
        for col in column_string.split(','):
            match = re.search(r'\s+AS\s+(\w+)', col, re.IGNORECASE)
            if match:
                col_name.append(match.group(1).strip())
            else:
                col_name.append(col.strip())
    else:
        print("Warning: Unable to extract column names from the SQL query.")
    return col_name



def final_result(classification, Input,selected_database,selected_datasource,mode):
    print('Entering_final_resut')
    if any(word.lower() in classification.lower() for word in ['visualization']):
        if mode == 'user_mode':
            print('Final_resut_user_mode_for_visual')
            sql_query, out, col_name, mode_type = sql_query_generator(Input,selected_database,selected_datasource,mode)
            result = pd.DataFrame(out, columns=col_name)

            # def generate_unique_colors(n):
            #     HSV_tuples = [(x * 1.0 / n, 0.5, 0.8) for x in range(n)]
            #     RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
            #     return [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.8)' for r, g, b in RGB_tuples]
            def generate_unique_colors(n):
                HSV_tuples = [(x * 1.0 / n, 0.8, 0.6) for x in range(n)]  # Increase saturation and reduce value for darker colors
                RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
                return [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.8)' for r, g, b in RGB_tuples]
            
            def generate_bar_chart(Input, result):
                if isinstance(result, list):
                    if len(result) > 1:
                        result = pd.DataFrame(result[1:], columns=result[0])
                    else:
                        result = pd.DataFrame()

                if not isinstance(result, pd.DataFrame):
                    raise ValueError("result should be a pandas DataFrame or convertible to one.")

                cols = result.columns.tolist()
                
                if len(cols) < 2:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Bar Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 75%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['Not enough data'];
                            const data = [1];
                            const uniqueColors = ['rgba(128, 128, 128, 0.8)'];

                            const config = {{
                                type: 'bar',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Not enough data',
                                        data: data,
                                        backgroundColor: uniqueColors,
                                        borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    maintainAspectRatio : true,
                                    animation: false,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: false }}
                                    }},
                                    scales: {{
                                        y: {{
                                            beginAtZero: true,
                                            title: {{
                                                display: true,
                                                text: 'Value',
                                                font: {{ size: 14 }}
                                            }}
                                        }},
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: 'Category',
                                                font: {{ size: 14 }}
                                            }},
                                            ticks: {{
                                                maxRotation: 90,
                                                minRotation: 90
                                }}
                                            
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                result = result.dropna(subset=cols)

                labels = result[cols[0]].tolist()
                data = result[cols[1]].tolist()
                
                unique_colors = generate_unique_colors(len(labels))

                if result.empty:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Bar Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 75%;display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['No values'];
                            const data = [1];
                            const uniqueColors = ['rgba(128, 128, 128, 0.8)'];

                            const config = {{
                                type: 'bar',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'No values',
                                        data: data,
                                        backgroundColor: uniqueColors,
                                        borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    maintainAspectRatio : true,
                                    animation: false,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: false }}
                                    }},
                                    scales: {{
                                        y: {{
                                            beginAtZero: true,
                                            title: {{
                                                display: true,
                                                text: 'Value',
                                                font: {{ size: 14 }}
                                            }}
                                        }},
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: '{cols[0]}',
                                                font: {{ size: 14 }}
                                            }},
                                            ticks: {{
                                    maxRotation: 90,
                                    minRotation: 90
                                }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Interactive Bar Chart</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div style="width: 75%; margin: auto; display: ruby-text;">
                        <canvas id="myChart"></canvas>
                    </div>
                    <script>
                        const uniqueColors = {json.dumps(unique_colors)};
                        const labels = {json.dumps(labels)};
                        const data = {json.dumps(data)};

                        const config = {{
                            type: 'bar',
                            data: {{
                                labels: labels,
                                datasets: [{{
                                    label: '{cols[1]}',
                                    data: data,
                                    backgroundColor: uniqueColors,
                                    borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                    borderWidth: 1,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio : true,
                                animation: false,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: '{Input}',
                                        font: {{ size: 18 }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                    }},
                                    legend: {{ display: false }}
                                }},
                                scales: {{
                                    y: {{
                                        beginAtZero: true,
                                        title: {{
                                            display: true,
                                            text: '{cols[1]}',
                                            font: {{ size: 14 }}
                                        }}
                                    }},
                                    x: {{
                                        title: {{
                                            display: true,
                                            text: '{cols[0]}',
                                            font: {{ size: 14 }}
                                        }},
                                        ticks: {{
                                    maxRotation: 90,
                                    minRotation: 90
                                }}
                                    }}
                                }}
                            }}
                        }};

                        const ctx = document.getElementById('myChart').getContext('2d');
                        new Chart(ctx, config);
                    </script>
                </body>
                </html>
                """

                with open('static/chart.html', 'w') as f:
                    f.write(html_content)
            
            def generate_pie_chart(Input, result):
                if isinstance(result, list):
                    if len(result) > 1:
                        result = pd.DataFrame(result[1:], columns=result[0])
                    else:
                        result = pd.DataFrame()

                if not isinstance(result, pd.DataFrame):
                    raise ValueError("result should be a pandas DataFrame or convertible to one.")

                cols = result.columns.tolist()
                
                if len(cols) < 2:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Pie Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 50%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['Not enough data'];
                            const data = [1];
                            const uniqueColors = ['rgba(128, 128, 128, 0.8)'];

                            const config = {{
                                type: 'pie',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Not enough data',
                                        data: data,
                                        backgroundColor: uniqueColors,
                                        borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    maintainAspectRatio: true,
                                    animation: false,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: true, 
                                        position: 'right',
                                        labels: {{
                                            title: 'Legend',
                                            titleAlign: 'center',
                                        }}, layout: {{
                                            padding: {{
                                                left: 50,
                                                right: 50,
                                                top: 0,
                                                bottom: 0
                                            }}
                                        }} 
                                    }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                result = result.dropna(subset=cols)

                labels = result[cols[0]].tolist()
                data = result[cols[1]].tolist()
                
                unique_colors = generate_unique_colors(len(labels))

                if result.empty:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Pie Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 50%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['No values'];
                            const data = [1];
                            const uniqueColors = ['rgba(128, 128, 128, 0.8)'];

                            const config = {{
                                type: 'pie',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'No values',
                                        data: data,
                                        backgroundColor: uniqueColors,
                                        borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    maintainAspectRatio: true,
                                    animation: false,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: true, position: 'right', 
                                        labels: {{
                                            title: 'Legend',
                                            titleAlign: 'center',
                                        }}, 
                                        layout: {{
                                            padding: {{
                                                left: 50,
                                                right: 50,
                                                top: 0,
                                                bottom: 0
                                            }}
                                         }} 
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Interactive Pie Chart</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div style="width: 50%; display: ruby-text; margin: auto;">
                        <canvas id="myChart"></canvas>
                    </div>
                    <script>
                        const uniqueColors = {json.dumps(unique_colors)};
                        const labels = {json.dumps(labels)};
                        const data = {json.dumps(data)};

                        const config = {{
                            type: 'pie',
                            data: {{
                                labels: labels,
                                datasets: [{{
                                    label: '{cols[1]}',
                                    data: data,
                                    backgroundColor: uniqueColors,
                                    borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                    borderWidth: 1,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: true,
                                animation: false,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: '{Input}',
                                        font: {{ size: 18 }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                    }},
                                    legend: {{ display: true, position: 'right',
                                    labels: {{
                                        title: 'Legend',
                                        titleAlign: 'center',
                                    }}, layout: {{
                            padding: {{
                                left: 50,
                                right: 50,
                                top: 0,
                                bottom: 0
                            }}
                        }}
                                }}
                                }}
                            }}
                        }};

                        const ctx = document.getElementById('myChart').getContext('2d');
                        new Chart(ctx, config);
                    </script>
                </body>
                </html>
                """

                with open('static/chart.html', 'w') as f:
                    f.write(html_content)

            
            def generate_scatter_plot(Input, result):
                if isinstance(result, list):
                    if len(result) > 1:
                        result = pd.DataFrame(result[1:], columns=result[0])
                    else:
                        result = pd.DataFrame()

                if not isinstance(result, pd.DataFrame):
                    raise ValueError("result should be a pandas DataFrame or convertible to one.")

                cols = result.columns.tolist()
                
                if len(cols) < 2:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Scatter Plot</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['Not enough data'];
                            const data = [{x: 0, y: 1}];
                            const uniqueColors = ['rgba(128, 128, 128, 0.8)'];

                            const config = {{
                                type: 'scatter',
                                data: {{
                                    datasets: [{{
                                        label: 'Not enough data',
                                        data: data,
                                        backgroundColor: uniqueColors,
                                        borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    maintainAspectRatio: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{
                                                size: 20,
                                                weight: 'bold'
                                            }},
                                            padding: {{
                                                top: 10,
                                                bottom: 30
                                            }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{
                                            position: 'right',
                                            align: 'center',
                                            labels: {{
                                                boxWidth: 15,
                                                font: {{
                                                    size: 12
                                                }},
                                                padding: 15
                                            }}
                                        }}
                                    }},
                                    layout: {{
                                        padding: {{
                                            left: 20,
                                            right: 20,
                                            top: 20,
                                            bottom: 20
                                        }}
                                    }},
                                    scales: {{
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: 'X-axis',
                                                font: {{ size: 14 }}
                                            }}
                                        }},
                                        y: {{
                                            beginAtZero: true,
                                            title: {{
                                                display: true,
                                                text: 'Y-axis',
                                                font: {{ size: 14 }}
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                result = result.dropna(subset=cols)

                data = [{'x': x, 'y': y} for x, y in zip(result[cols[0]], result[cols[1]])]
                unique_colors = generate_unique_colors(len(data))

                if result.empty:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Scatter Plot</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['No values'];
                            const data = [{x: 0, y: 1}];
                            const uniqueColors = ['rgba(128, 128, 128, 0.8)'];

                            const config = {{
                                type: 'scatter',
                                data: {{
                                    datasets: [{{
                                        label: 'No values',
                                        data: data,
                                        backgroundColor: uniqueColors,
                                        borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    maintainAspectRatio: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{
                                                size: 20,
                                                weight: 'bold'
                                            }},
                                            padding: {{
                                                top: 10,
                                                bottom: 30
                                            }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{
                                            position: 'right',
                                            align: 'center',
                                            labels: {{
                                                boxWidth: 15,
                                                font: {{
                                                    size: 12
                                                }},
                                                padding: 15
                                            }}
                                        }}
                                    }},
                                    layout: {{
                                        padding: {{
                                            left: 20,
                                            right: 20,
                                            top: 20,
                                            bottom: 20
                                        }}
                                    }},
                                    scales: {{
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: 'X-axis',
                                                font: {{ size: 14 }}
                                            }}
                                        }},
                                        y: {{
                                            beginAtZero: true,
                                            title: {{
                                                display: true,
                                                text: 'Y-axis',
                                                font: {{ size: 14 }}
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Interactive Scatter Plot</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div style="width: 85%; display: ruby-text; margin: auto;">
                        <canvas id="myChart"></canvas>
                    </div>
                    <script>
                        const uniqueColors = {json.dumps(unique_colors)};
                        const data = {json.dumps(data)};

                        const config = {{
                            type: 'scatter',
                            data: {{
                                datasets: [{{
                                    label: '{cols[1]}',
                                    data: data,
                                    backgroundColor: uniqueColors,
                                    borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                    borderWidth: 1,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: true,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: '{Input}',
                                        font: {{
                                            size: 20,
                                            weight: 'bold'
                                        }},
                                        padding: {{
                                            top: 10,
                                            bottom: 30
                                        }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                    }},
                                    legend: {{
                                        position: 'right',
                                        align: 'center',
                                        labels: {{
                                            boxWidth: 15,
                                            font: {{
                                                size: 12
                                            }},
                                            padding: 15
                                        }}
                                    }}
                                }},
                                layout: {{
                                    padding: {{
                                        left: 20,
                                        right: 20,
                                        top: 20,
                                        bottom: 20
                                    }}
                                }},
                                scales: {{
                                    x: {{
                                        title: {{
                                            display: true,
                                            text: '{cols[0]}',
                                            font: {{ size: 14 }}
                                        }}
                                    }},
                                    y: {{
                                        beginAtZero: true,
                                        title: {{
                                            display: true,
                                            text: '{cols[1]}',
                                            font: {{ size: 14 }}
                                        }}
                                    }}
                                }}
                            }}
                        }};

                        const ctx = document.getElementById('myChart').getContext('2d');
                        new Chart(ctx, config);
                    </script>
                </body>
                </html>
                """

                with open('static/chart.html', 'w') as f:
                    f.write(html_content)

            def generate_column_chart(Input, result):
                # Check if result is a list or tuple and convert it to DataFrame
                if isinstance(result, (list, tuple)):
                    result = list(result)  # Convert tuple to list if necessary
                    if len(result) > 1:
                        # Assuming the first row is the header and the rest are data
                        result = pd.DataFrame(result[1:], columns=result[0])
                    else:
                        # Handle the case where the result list is empty or has no data rows
                        result = pd.DataFrame()

                # Now result should be a DataFrame
                if not isinstance(result, pd.DataFrame):
                    raise ValueError("result should be a pandas DataFrame or convertible to one.")

                cols = result.columns.tolist()
                
                if len(cols) < 2:
                    # Handle the case where there are not enough columns
                    print('Not enough columns to generate chart')
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Column Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['Not enough data'];
                            const data = [1];
                            const uniqueColors = ['rgba(128, 128, 128, 0.8)'];

                            const config = {{
                                type: 'bar',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Not enough data',
                                        data: data,
                                        backgroundColor: uniqueColors,
                                        borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                        borderWidth: 1
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: false }}
                                    }},
                                    scales: {{
                                        y: {{
                                            beginAtZero: true,
                                            title: {{
                                                display: true,
                                                text: 'Value',
                                                font: {{ size: 14 }}
                                            }}
                                        }},
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: 'Category',
                                                font: {{ size: 14 }}
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                result = result.dropna(subset=cols)

                labels = result[cols[0]].tolist()
                data = result[cols[1]].tolist()
                
                unique_colors = generate_unique_colors(len(labels))

                if result.empty:
                    print('came for no values')
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Column Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['No values'];
                            const data = [1];
                            const uniqueColors = ['rgba(128, 128, 128, 0.8)'];

                            const config = {{
                                type: 'bar',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'No values',
                                        data: data,
                                        backgroundColor: uniqueColors,
                                        borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: false }}
                                    }},
                                    scales: {{
                                        y: {{
                                            beginAtZero: true,
                                            title: {{
                                                display: true,
                                                text: 'Value',
                                                font: {{ size: 14 }}
                                            }}
                                        }},
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: '{cols[0]}',
                                                font: {{ size: 14 }}
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Interactive Column Chart</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div style="width: 85%; display: ruby-text; margin: auto;">
                        <canvas id="myChart"></canvas>
                    </div>
                    <script>
                        const uniqueColors = {json.dumps(unique_colors)};
                        const labels = {json.dumps(labels)};
                        const data = {json.dumps(data)};

                        const config = {{
                            type: 'bar',
                            data: {{
                                labels: labels,
                                datasets: [{{
                                    label: '{cols[1]}',
                                    data: data,
                                    backgroundColor: uniqueColors,
                                    borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                    borderWidth: 1,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: '{Input}',
                                        font: {{ size: 18 }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                    }},
                                    legend: {{ display: false }}
                                }},
                                scales: {{
                                    y: {{
                                        beginAtZero: true,
                                        title: {{
                                            display: true,
                                            text: '{cols[1]}',
                                            font: {{ size: 14 }}
                                            }}
                                        }},
                                    x: {{
                                        title: {{
                                            display: true,
                                            text: '{cols[0]}',
                                            font: {{ size: 14 }}
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                        const ctx = document.getElementById('myChart').getContext('2d');
                        new Chart(ctx, config);
                    </script>
                </body>
                </html>
                """

                with open('static/chart.html', 'w') as f:
                    f.write(html_content)
        
            def generate_area_chart(Input, result):
                if isinstance(result, list):
                    if len(result) > 1:
                        result = pd.DataFrame(result[1:], columns=result[0])
                    else:
                        result = pd.DataFrame()

                if not isinstance(result, pd.DataFrame):
                    raise ValueError("result should be a pandas DataFrame or convertible to one.")

                cols = result.columns.tolist()
                
                if len(cols) < 2:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Area Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['Not enough data'];
                            const data = [1];
                            const backgroundColor = 'rgba(128, 128, 128, 0.5)';
                            const borderColor = 'rgba(128, 128, 128, 1)';

                            const config = {{
                                type: 'line',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Not enough data',
                                        data: data,
                                        backgroundColor: backgroundColor,
                                        borderColor: borderColor,
                                        fill: true,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: true }}
                                    }},
                                    scales: {{
                                        x: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[0] if len(cols) > 0 else "X Axis"}'
                                            }}
                                        }},
                                        y: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[1] if len(cols) > 1 else "Y Axis"}'
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                result = result.dropna(subset=cols)

                labels = result[cols[0]].tolist()
                data = result[cols[1]].tolist()
                
                if result.empty:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Area Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['No values'];
                            const data = [1];
                            const backgroundColor = 'rgba(128, 128, 128, 0.5)';
                            const borderColor = 'rgba(128, 128, 128, 1)';

                            const config = {{
                                type: 'line',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'No values',
                                        data: data,
                                        backgroundColor: backgroundColor,
                                        borderColor: borderColor,
                                        fill: true,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: true }}
                                    }},
                                    scales: {{
                                        x: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[0] if len(cols) > 0 else "X Axis"}'
                                            }}
                                        }},
                                        y: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[1] if len(cols) > 1 else "Y Axis"}'
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Interactive Area Chart</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div style="width: 85%; display: ruby-text; margin: auto;">
                        <canvas id="myChart"></canvas>
                    </div>
                    <script>
                        const labels = {json.dumps(labels)};
                        const data = {json.dumps(data)};
                        const backgroundColor = 'rgba(75, 192, 192, 0.5)';
                        const borderColor = 'rgba(75, 192, 192, 1)';

                        const config = {{
                            type: 'line',
                            data: {{
                                labels: labels,
                                datasets: [{{
                                    label: '{cols[1]}',
                                    data: data,
                                    backgroundColor: backgroundColor,
                                    borderColor: borderColor,
                                    fill: true,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: '{Input}',
                                        font: {{ size: 18 }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                    }},
                                    legend: {{ display: true }}
                                }},
                                scales: {{
                                    x: {{
                                        display: true,
                                        title: {{
                                            display: true,
                                            text: '{cols[0]}'
                                        }}
                                    }},
                                    y: {{
                                        display: true,
                                        title: {{
                                            display: true,
                                            text: '{cols[1]}'
                                        }}
                                    }}
                                }}
                            }}
                        }};

                        const ctx = document.getElementById('myChart').getContext('2d');
                        new Chart(ctx, config);
                    </script>
                </body>
                </html>
                """

                with open('static/chart.html', 'w') as f:
                    f.write(html_content)

            def generate_line_chart(Input, result):
                if isinstance(result, list):
                    if len(result) > 1:
                        result = pd.DataFrame(result[1:], columns=result[0])
                    else:
                        result = pd.DataFrame()

                if not isinstance(result, pd.DataFrame):
                    raise ValueError("result should be a pandas DataFrame or convertible to one.")

                cols = result.columns.tolist()

                if len(cols) < 2:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Line Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['Not enough data'];
                            const data = [1];
                            const backgroundColor = 'rgba(128, 128, 128, 0.5)';
                            const borderColor = 'rgba(128, 128, 128, 1)';

                            const config = {{
                                type: 'line',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Not enough data',
                                        data: data,
                                        backgroundColor: backgroundColor,
                                        borderColor: borderColor,
                                        fill: false,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: true }}
                                    }},
                                    scales: {{
                                        x: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[0] if len(cols) > 0 else "X Axis"}'
                                            }}
                                        }},
                                        y: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[1] if len(cols) > 1 else "Y Axis"}'
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                result = result.dropna(subset=cols)

                labels = result[cols[0]].tolist()
                data = result[cols[1]].tolist()
                
                if result.empty:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Line Chart</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['No values'];
                            const data = [1];
                            const backgroundColor = 'rgba(128, 128, 128, 0.5)';
                            const borderColor = 'rgba(128, 128, 128, 1)';

                            const config = {{
                                type: 'line',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'No values',
                                        data: data,
                                        backgroundColor: backgroundColor,
                                        borderColor: borderColor,
                                        fill: false,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: true }}
                                    }},
                                    scales: {{
                                        x: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[0] if len(cols) > 0 else "X Axis"}'
                                            }}
                                        }},
                                        y: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[1] if len(cols) > 1 else "Y Axis"}'
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                unique_colors = generate_unique_colors(len(labels))

                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Interactive Line Chart</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div style="width: 85%; display: ruby-text; margin: auto;">
                        <canvas id="myChart"></canvas>
                    </div>
                    <script>
                        const uniqueColors = {json.dumps(unique_colors)};
                        const labels = {json.dumps(labels)};
                        const data = {json.dumps(data)};
                        const backgroundColor = 'rgba(75, 192, 192, 0.5)';
                        const borderColor = 'rgba(75, 192, 192, 1)';

                        const config = {{
                            type: 'line',
                            data: {{
                                labels: labels,
                                datasets: [{{
                                    label: '{cols[1]}',
                                    data: data,
                                    backgroundColor: backgroundColor,
                                    borderColor: borderColor,
                                    fill: false,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: '{Input}',
                                        font: {{ size: 18 }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                    }},
                                    legend: {{ display: true }}
                                }},
                                scales: {{
                                    x: {{
                                        display: true,
                                        title: {{
                                            display: true,
                                            text: '{cols[0]}'
                                        }}
                                    }},
                                    y: {{
                                        display: true,
                                        title: {{
                                            display: true,
                                            text: '{cols[1]}'
                                        }}
                                    }}
                                }}
                            }}
                        }};

                        const ctx = document.getElementById('myChart').getContext('2d');
                        new Chart(ctx, config);
                    </script>
                </body>
                </html>
                """

                with open('static/chart.html', 'w') as f:
                    f.write(html_content)

            def generate_count_plot(Input, result):
                if isinstance(result, list):
                    if len(result) > 1:
                        result = pd.DataFrame(result[1:], columns=result[0])
                    else:
                        result = pd.DataFrame()

                if not isinstance(result, pd.DataFrame):
                    raise ValueError("result should be a pandas DataFrame or convertible to one.")

                cols = result.columns.tolist()

                if len(cols) < 1:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Count Plot</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['Not enough data'];
                            const data = [1];
                            const backgroundColor = 'rgba(128, 128, 128, 0.5)';
                            const borderColor = 'rgba(128, 128, 128, 1)';

                            const config = {{
                                type: 'bar',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Not enough data',
                                        data: data,
                                        backgroundColor: backgroundColor,
                                        borderColor: borderColor,
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: true }}
                                    }},
                                    scales: {{
                                        x: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[0] if len(cols) > 0 else "X Axis"}'
                                            }}
                                        }},
                                        y: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: 'Count'
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                count_data = result[cols[0]].value_counts().reset_index()
                count_data.columns = [cols[0], 'count']

                labels = count_data[cols[0]].tolist()
                data = count_data['count'].tolist()

                if count_data.empty:
                    html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Interactive Count Plot</title>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div style="width: 85%; display: ruby-text; margin: auto;">
                            <canvas id="myChart"></canvas>
                        </div>
                        <script>
                            const labels = ['No values'];
                            const data = [1];
                            const backgroundColor = 'rgba(128, 128, 128, 0.5)';
                            const borderColor = 'rgba(128, 128, 128, 1)';

                            const config = {{
                                type: 'bar',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'No values',
                                        data: data,
                                        backgroundColor: backgroundColor,
                                        borderColor: borderColor,
                                        borderWidth: 1,
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        title: {{
                                            display: true,
                                            text: '{Input}',
                                            font: {{ size: 18 }}
                                        }},
                                        tooltip: {{
                                            mode: 'index',
                                            intersect: false,
                                        }},
                                        legend: {{ display: true }}
                                    }},
                                    scales: {{
                                        x: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: '{cols[0] if len(cols) > 0 else "X Axis"}'
                                            }}
                                        }},
                                        y: {{
                                            display: true,
                                            title: {{
                                                display: true,
                                                text: 'Count'
                                            }}
                                        }}
                                    }}
                                }}
                            }};

                            const ctx = document.getElementById('myChart').getContext('2d');
                            new Chart(ctx, config);
                        </script>
                    </body>
                    </html>
                    """
                    with open('static/chart.html', 'w') as f:
                        f.write(html_content)
                    return

                unique_colors = generate_unique_colors(len(labels))

                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Interactive Count Plot</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div style="width: 85%; display: ruby-text; margin: auto;">
                        <canvas id="myChart"></canvas>
                    </div>
                    <script>
                        const uniqueColors = {json.dumps(unique_colors)};
                        const labels = {json.dumps(labels)};
                        const data = {json.dumps(data)};

                        const config = {{
                            type: 'bar',
                            data: {{
                                labels: labels,
                                datasets: [{{
                                    label: 'Count',
                                    data: data,
                                    backgroundColor: uniqueColors,
                                    borderColor: uniqueColors.map(color => color.replace('0.8', '1')),
                                    borderWidth: 1,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                plugins: {{
                                    title: {{
                                        display: true,
                                        text: '{Input}',
                                        font: {{ size: 18 }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                    }},
                                    legend: {{ display: true }}
                                }},
                                scales: {{
                                    x: {{
                                        display: true,
                                        title: {{
                                            display: true,
                                            text: '{cols[0]}'
                                        }}
                                    }},
                                    y: {{
                                        display: true,
                                        title: {{
                                            display: true,
                                            text: 'Count'
                                        }}
                                    }}
                                }}
                            }}
                        }};

                        const ctx = document.getElementById('myChart').getContext('2d');
                        new Chart(ctx, config);
                    </script>
                </body>
                </html>
                """

                with open('static/chart.html', 'w') as f:
                    f.write(html_content)

            if "bar chart" in Input.lower():
                generate_bar_chart(Input, result)
            elif 'pie chart' in Input.lower():
                generate_pie_chart(Input, result)
            elif 'scatter chart' in Input.lower():
                generate_scatter_plot(Input, result)
            elif 'column chart' in Input.lower():
                generate_column_chart(Input, result)
            elif 'area chart' in Input.lower():
                generate_area_chart(Input, result)
            elif 'line chart' in Input.lower():
                generate_line_chart(Input, result)
            elif 'count chart' in Input.lower():
                generate_count_plot(Input, result)
            else:
                print('No specific chart mentioned')
            

            response = {
                "input": Input,
                "sql_query": sql_query,
                "output": "Visualization Generated",
                "chart_filename": "/static/chart.html"
            }

            return sql_query, None, response, selected_database, selected_datasource, mode
        
        elif mode == 'developer_mode':
            print('Final_resut_dev_mode_for_visual')
            sql_query,col_name = sql_query_generator(Input,selected_database,selected_datasource,mode)
            return sql_query, col_name,mode
            
    elif any(word in classification.lower() for word in ['table']):
        if mode == 'user_mode':
            print('Final_resut_user_mode_for_table')
            sql_query, out,col_name, mode_ = sql_query_generator(Input,selected_database,selected_datasource,mode)
            d_f = pd.DataFrame(out,columns=col_name)
            return sql_query, d_f, None, selected_database,selected_datasource, mode
        elif mode == 'developer_mode':
            print('Final_resut_dev_mode_for_table')
            sql_query, col_name = sql_query_generator(Input,selected_database,selected_datasource,mode)
            return sql_query, col_name,mode
    
    else:
        if mode == 'user_mode':
            print('Final_resut_user_mode_for_NL')
            sql_query, out,col_name,mode = sql_query_generator(Input,selected_database,selected_datasource,mode)

            prompt_1 = f"""

    Instructions:
    Given a question {Input} and the corresponding answer you should reframe the answer in natural language.
    Don't use words outside the scope which are not there in either question or answer. Never give any additional content in the response.

    Below are the example Questions and their Answers and how you should Respond.

    Example:

    Question: What is the average age of the patients?;
    Answer: 30;
    Your Response: The average age of patients is 30.

    Question: How many rows are present in the doctor table?;
    Answer: 20;
    Your Response: There are 20 rows present in the doctor's table.

    Question: Delete the phone numbers for the doctor whose specialization is Cardiology;
    Answer: Query executed successfully: 4 rows affected.
    Your Response: The phone number for the doctor whose specialization is Cardiology has been deleted.

    Question: Update the phone numbers for the doctor whose specialization is Neurology;
    Answer: Query executed successfully: 10 rows affected.
    Your Response: The phone number for the doctor whose specialization is Neurology has been updated.

    Question: Replace the phone numbers for the doctor whose specialization is Oncology;
    Answer: Query executed successfully: 15 rows affected.
    Your Response: The phone number for the doctor whose specialization is Oncology has been replaced.

    Question: Insert the phone number for the doctor whose specialization is Oncology where it is null;
    Answer: Query executed successfully: 0 rows affected.
    Your Response: The phone number for the doctor whose specialization is Oncology has been inserted where it was null.

    Question: {Input};
    Answer: {out};

    additional_hints

    You should only give the {'Your Response'} part. Don't give any other comments in the response.
    """
            input_token_nl_response_user_mode = count_tokens(prompt_1)
            # print(input_token_nl_response_user_mode,'input_token_nl_response_user_mode')
            logging.info(f"{input_token_nl_response_user_mode} - input_token_nl_response_user_mode")
            nl_response = call_openai_gpt(prompt_1)
            output_token_nl_response_user_mode = count_tokens(nl_response)
            # print(output_token_nl_response_user_mode,'output_token_nl_response_user_mode')
            logging.info(f"{output_token_nl_response_user_mode} - output_token_nl_response_user_mode")


            return sql_query, nl_response, None, selected_database, selected_datasource, mode
        elif mode == 'developer_mode':
            print('Final_resut_dev_mode_for_NL')
            sql_query,col_name = sql_query_generator(Input,selected_database,selected_datasource,mode)

            return sql_query,col_name, mode
