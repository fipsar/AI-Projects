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
from scripts import connection,call_openai_gpt,extract_column_names,count_tokens
import colorsys
import logging
import logging_config



def ai_result(Input, classification_dev, edited_sql, selected_datasource, selected_database, mode):
    col_name = extract_column_names(edited_sql)
    result = connection(edited_sql, selected_datasource)
    if any(word.lower() in classification_dev.lower() for word in ['visualization']):
        if mode == 'developer_mode':
            print('Handling_visual_res_for_dev_mode')
            def generate_unique_colors(n):
                HSV_tuples = [(x * 1.0 / n, 0.5, 0.8) for x in range(n)]
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
                        <div style="width: 80%; margin: auto;">
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
                                        borderSkipped: false,
                                        borderRadius: 5,
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
                        <div style="width: 80%; margin: auto;">
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
                                        borderSkipped: false,
                                        borderRadius: 5,
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
                    <title>Interactive Bar Chart</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div style="width: 80%; margin: auto;">
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
                                    borderSkipped: false,
                                    borderRadius: 5,
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
                        <div style="width: 80%; margin: auto;">
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
                        <div style="width: 80%; margin: auto;">
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
                    <div style="width: 80%; margin: auto;">
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
            if "pie chart" in Input.lower():
                generate_pie_chart(Input, result)
            else:
                print('No specific chart mentioned')
            
            response = {
                "input": Input,
                "ai_res": "Visualization Generated",
                "chart_filename": "/static/chart.html"
            }

            return None, response, selected_database, selected_datasource, mode

        
    if any(word in classification_dev.lower() for word in ['table']):
        if mode == 'developer_mode':
            print('Handling_table_res_for_dev_mode')
            result = connection(edited_sql, selected_datasource)
            d_f = pd.DataFrame(result,columns=col_name)
            return d_f, None, selected_database, selected_datasource, mode
        
    else:
        if mode == 'developer_mode':
            print('Handling_NL_res_for_dev_mode')
            result = connection(edited_sql, selected_datasource)

            prompt_1 = f"""
    This function handles the developer mode, where there may be a chance to edit the generated SQL query. If the user does not edit the SQL query {edited_sql}, 
    follow Instructions_1. If the user edits the generated SQL query {edited_sql}, follow Instructions_2.

**Instructions_1:**
Do not execute when {edited_sql} contains SQL.
Given a question {Input} and the corresponding answer, reframe the answer in natural language. Do not use words outside the scope of the question or answer. 
Never add any additional content in the response.

**Examples:**
- **Question:** What is the average age of the patients?  
  **Answer:** 30  
  **Your Response:** The average age of patients is 30.
  
- **Question:** How many rows are present in the doctor table?  
  **Answer:** 20  
  **Your Response:** There are 20 rows present in the doctor's table.
  
- **Question:** What is the distinct name of the patients?  
  **Answer:** Ram, Raja, Siva, Kumar  
  **Your Response:** The distinct name of the patients is Ram, Raja, Siva, Kumar.
  
- **Question:** Delete the phone numbers for the doctor whose specialization is Cardiology.  
  **Answer:** Query executed successfully: 4 rows affected.  
  **Your Response:** The phone number for the doctor whose specialization is Cardiology has been deleted.
  
- **Question:** Update the phone numbers for the doctor whose specialization is Neurology.  
  **Answer:** Query executed successfully: 10 rows affected.  
  **Your Response:** The phone number for the doctor whose specialization is Neurology has been updated.
  
- **Question:** Replace the phone numbers for the doctor whose specialization is Oncology.  
  **Answer:** Query executed successfully: 15 rows affected.  
  **Your Response:** The phone number for the doctor whose specialization is Oncology has been replaced.
  
- **Question:** Insert the phone number for the doctor whose specialization is Oncology where it is null.  
  **Answer:** Query executed successfully: 0 rows affected.  
  **Your Response:** The phone number for the doctor whose specialization is Oncology has been inserted where it was null.

- **Question:** {Input}  
  **Answer:** {result}

**Additional Hints:** You should only provide the {'Your Response'} part. Do not give any other comments in the response.

**Instructions_2:**
Must execute when {edited_sql} contains SQL.
Given the edited SQL query {edited_sql} and the corresponding answer based on {result}, reframe the answer in natural language and strictly do not give the previous 
question's answer.
    """     
            input_token_prompt_for_ai_res = count_tokens(prompt_1)
            # print(input_token_prompt_for_ai_res,'input_token_prompt_for_ai_res')
            logging.info(f"{input_token_prompt_for_ai_res} - input_token_prompt_for_ai_res")
            nl_response = call_openai_gpt(prompt_1)
            dev_mode_output_token_count_nl_response = count_tokens(nl_response)
            # print(dev_mode_output_token_count_nl_response,'dev_mode_output_token_count_nl_response')
            logging.info(f"{dev_mode_output_token_count_nl_response} - dev_mode_output_token_count_nl_response")
            return nl_response, None, selected_database,selected_datasource, mode
