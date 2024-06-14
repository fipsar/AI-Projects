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


load_dotenv()

def connection(query):
    try:
        conn = mysql.connector.connect(
            user='root',
            password='1234',
            host='localhost',
            port='3306',
            database=session.get('selected_database', 'healthcare')
        )
        
        print('Connected to database:', conn.database)
        
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
        print(f"Error occurred: {error}")
        return f"Error: {error}"



def get_specific_tables_metadata(connection, selected_database, schema_name, tables_df, num_samples=2):
    print('get_specific_tables_metadata database name', selected_database)
    metadata = ""
    working_tables = tables_df[tables_df['Flag'] == 1]
    
    for _, table_row in working_tables.iterrows():
        table_name = table_row['Table_name']
        metadata += f"\nMetadata for table {selected_database}.{schema_name}.{table_name}:\n"
        sql = f"SHOW COLUMNS FROM {schema_name}.{table_name}"
        
        result = connection(sql)
        create_table_statement = f"CREATE TABLE {schema_name}.{table_name} ("
        for row in result:
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
        sample_rows = connection(sample_sql)

        metadata += f"\nSample rows from {table_name} table:\n"
        for row in sample_rows:
            metadata += str(row) + "\n"
    
    return metadata


def meta_data(selected_database):
    healthcare_schema_df = pd.read_csv('health_data_schema.csv')
    pharma_schema_df = pd.read_csv('pharma_database_schema.csv')

    if selected_database == 'healthcare':
        schema_name = 'healthcare'
        df = healthcare_schema_df
    elif selected_database == 'pharma':
        schema_name = 'pharma'
        df = pharma_schema_df
    else:
        raise ValueError(f"Unknown database selected: {selected_database}")

    table_info = get_specific_tables_metadata(connection, selected_database, schema_name, df)
    return table_info, schema_name

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_openai_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",  # Specify the model
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
    print('From que_class------>',Input)
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

    classification = call_openai_gpt(prompt_for_classify_input)
    print('clasified as----->',classification)
    return classification.strip()

def sql_query_generator(Input, selected_database):
    meta_data_content, schema_name = meta_data(selected_database)
    print('From sql query generator, schema name:', schema_name)
    print('From sql query generator, meta_data_content name:', meta_data_content)


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

    generated_sql = call_openai_gpt(prompt_sql_data)

    cleaned_sql = generated_sql.replace("<SQL>", "").replace("</SQL>", "")

    if f"{schema_name}." not in cleaned_sql:
        cleaned_sql = cleaned_sql.replace('FROM', f'FROM {schema_name}.')
        cleaned_sql = cleaned_sql.replace('JOIN', f'JOIN {schema_name}.')

    query = f"{cleaned_sql}".strip()
    print("query :", query)

    results = connection(query)
    print("result :", results)
    return query, results

def final_result(classification, Input,selected_database):
    if any(word.lower() in classification.lower() for word in ['visualization']):
        sql_query, out = sql_query_generator(Input,selected_database)
        result = pd.DataFrame(out)

        def generate_bar_chart(Input, result):
            cols = result.columns.tolist()
            result = result.dropna(subset=cols)
            plt.figure(figsize=(16,8))

            x_ticks = np.arange(len(result[cols[0]]))
            plt.bar(x_ticks, result[cols[1]])
            plt.xticks(x_ticks, result[cols[0]], rotation='vertical')
            plt.xlabel(cols[0])
            plt.ylabel(cols[1])
            plt.title(f"Bar Chart: {Input}", fontweight='bold')
            plt.tight_layout()
            plt.savefig('static/chart.png')
            plt.close()

        def generate_pie_chart(Input, result):
            cols = result.columns.tolist()
            result = result.dropna(subset=cols)
            plt.figure(figsize=(16,8))

            wedges, labels, _ = plt.pie(result[cols[1]], autopct='%1.1f%%', pctdistance=0.85)

            num_labels = len(labels)
            cmap = plt.get_cmap('tab20')
            colors = cmap(np.linspace(0,1, num_labels))
            legend_labels = result[cols[0]]
            legend_handles = []
            for i, label in enumerate(legend_labels):
                legend_handles.append(plt.Rectangle((0,0),0.5,0.5, color = colors[i], label=label))
            plt.legend(handles = legend_handles, loc ='right', bbox_to_anchor = (1.5,0.5),title= cols[0])

            for i, wedge in enumerate(wedges):
                wedge.set_facecolor(colors[i])

            plt.title(f"Pie Chart: {Input}", fontweight='bold')
            plt.savefig('static/chart.png')
            plt.close()

        if "bar chart" in Input.lower():
            generate_bar_chart(Input, result)
        elif 'pie chart' in Input.lower():
            generate_pie_chart(Input, result)
        else:
            print('No specific chart mentioned')
           

        response = {
            "input": Input,
            "sql_query": sql_query,
            "output": "Visualization Generated",
            "chart_filename": "/static/chart.png"
        }

        return sql_query, None, response, selected_database
    
    elif any(word in classification.lower() for word in ['table']):
        sql_query, out = sql_query_generator(Input,selected_database)
        d_f = pd.DataFrame(out)
        return sql_query, d_f, None, selected_database
    
    else:
        sql_query, out = sql_query_generator(Input,selected_database)

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
        nl_response = call_openai_gpt(prompt_1)

        return sql_query, nl_response, None, selected_database


## handle empty value for bar chart 

# import matplotlib.pyplot as plt
# import numpy as np

# def generate_pie_chart(Input, result):
#     cols = result.columns.tolist()
#     result = result.dropna(subset=cols)

#     if result.empty:
#         # Create a bar chart with "No values" message
#         fig, ax = plt.subplots(figsize=(16, 8))
#         ax.barh(['No values'], [1], color='gray')
#         ax.set_yticks([])
#         ax.set_xlim(0, 1)
#         ax.text(0.5, 0, 'No values', ha='center', va='center', fontsize=15, color='red')
#         plt.title(f"Pie Chart: {Input}", fontweight='bold')
#         plt.savefig('static/chart.png')
#         plt.close()
#     else:
#         plt.figure(figsize=(16, 8))
#         wedges, labels, _ = plt.pie(result[cols[1]], autopct='%1.1f%%', pctdistance=0.85)

#         num_labels = len(labels)
#         cmap = plt.get_cmap('tab20')
#         colors = cmap(np.linspace(0, 1, num_labels))
#         legend_labels = result[cols[0]]
#         legend_handles = []
#         for i, label in enumerate(legend_labels):
#             legend_handles.append(plt.Rectangle((0, 0), 0.5, 0.5, color=colors[i], label=label))
#         plt.legend(handles=legend_handles, loc='right', bbox_to_anchor=(1.5, 0.5), title=cols[0])

#         for i, wedge in enumerate(wedges):
#             wedge.set_facecolor(colors[i])

#         plt.title(f"Pie Chart: {Input}", fontweight='bold')
#         plt.savefig('static/chart.png')
#         plt.close()
