from flask import Flask, request, render_template, jsonify, url_for, session, redirect, abort
from scripts import question_classification, final_result,call_openai_gpt
from datetime import datetime
import pandas as pd
import csv
import os
import copy
import os


# username = os.getenv('username')
# password = os.getenv('password')
# schema_name = os.getenv('schema_name')
# database_name = os.getenv('database_name')
app = Flask(__name__)

chat_history = []


def combine_question_for_followup(chat_history):
    print('This is the chat history questions from the combine ques follow up functn:-->', chat_history)
    prompt_for_followup = f"""Given the {chat_history} which contains list of questions, you need to rephrase those questions to be a standalone question.
<instructions>
Given a Chat History where the user has asked a series of questions sequentially, provide a rephrased question to the most last question. 
If there is any conflict between the previous questions and the last question, prioritize the last question.
Restrict your output to number of tokens required for saying just the rephrased question of the specified Chat History.
Do not strictly write the prompt detail in the output.
</instructions>

Below are the examples of the Chat History list and the Rephrased Question.

<example>
Chat History : ['Generate a bar chart for the count of patient sex ', 'show this as a Pie chart instead']
Rephrased Question : Generate a pie chart for the count of patient sex.
</example>

<example>
Chat History : ['What is the prescription date for the patient Cris ?', 'which doctor prescribed this?']
Rephrased Question : Which doctor prescribed this for the patient Cris ?.
</example>

<chat_history>
Chat History: {chat_history}
</chat_history>
"""
    
    question_response = call_openai_gpt(prompt_for_followup)
    return question_response.strip()

def handle_follow_up_question(chat_history):
    classification_for_follow_up = question_classification(chat_history)
    print('This is classification of Rephrased Question--->', classification_for_follow_up)
    sql_query, output, visualization_data = final_result(classification_for_follow_up, chat_history)

    if isinstance(output, pd.DataFrame):
        output = output.to_html(index=False, header=False)
        print(sql_query)
        print(output)

    if visualization_data is not None:
        print(sql_query)
        print(output)
        print(visualization_data)
        response = visualization_data
    else:
        print(sql_query)
        print(output)
        response = {'input':chat_history, "sql_query":sql_query,"output":output}
        if response['sql_query']:
            response['sql_query'] = response['sql_query'].replace('\n','<br>')
    return response

def handle_new_question(chat_history):
    print('This is the chat history from handle new ques funct:--->',chat_history[0])
    classification = question_classification(chat_history[0])
    print('This is the classification variable history:--->',classification)
    sql_query, output, visualization_data = final_result(classification, chat_history[0])

    if isinstance(output, pd.DataFrame):
        output = output.to_html(index=False, header=False)
        print(sql_query)
        print(output)
    if visualization_data is not None:
        print(sql_query)
        print(output)
        print(visualization_data)
        response = visualization_data
    else:
        print(sql_query)
        print(output)
        response = {'input':chat_history[0], "sql_query":sql_query,"output":output}
        if response['sql_query']:
            response['sql_query'] =response['sql_query'].replace('\n','<br>')
    return response

@app.route('/')
def index():
    chat_type = 'new'
    return render_template('index.html', chat_type=chat_type)

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    if request.method =='POST':
        Input= request.form['chatInput']
        chat_type = request.form['chat_type']
        print('chat type for initalization',chat_type)
        selected_database = request.form['databaseSelect']
        print('selected_database for initalization',selected_database)
        
        def process_chat(chat_type,Input):
            global chat_history

            try:
                original_chat_history = copy.deepcopy(chat_history)

                if chat_type =='new':
                    chat_history = [Input]
                    response = handle_new_question(chat_history)
                else:
                    chat_history.append(Input)
                    combined_question = combine_question_for_followup(chat_history)
                    print('This is the combined [Rephrased] ques from follow up:--->',combined_question)
                    chat_history = [combined_question]
                    print('This is the chat history of combined ques:--->',chat_history)
                    response = handle_follow_up_question(chat_history[0])
                return response
            except Exception as e:
                print("error happended--->" , e)
                chat_history = original_chat_history
                raise

        try:
            print('This is 1st try---->')
            response = process_chat(chat_type, Input)
            print('This is 1st try input---->',Input)
        except Exception as e:
            try:
                print('This is 2nd try----->')
                response = process_chat(chat_type, Input)
                print('This is 2nd try Input----->',Input)
            except Exception as e:
                error_message = " Sorry, I don't understand your question. Please rephrase and try again."
                response = {
                    'error_message' : error_message
                }

        return render_template('index.html', response= response, chat_type=chat_type,selected_database=selected_database)
    
@app.route('/record_feedback', methods = ['POST'])
def record_feedback():
    feedback_data = request.json
    classify = 'Natural Language Response' if '.' in feedback_data['aiResponse'] else ('Visual Response' if not feedback_data['aiResponse'] else 'Table Format Response')
    
    user_question = feedback_data['userQuestion']
    response_type = classify
    sql_query = feedback_data['sqlQuery']
    ai_response =feedback_data['aiResponse']
    feedback = feedback_data['feedback']
    feedback_type = feedback_data['feedbackType']
    comment = feedback_data['comment']

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_info = ''

    with open('feedback.csv','a',newline='') as csvfile:
        fieldnames = ['Date','User Info', 'User Question', 'Response Type', 'SQL Query', 'AI Response', 'Feedback', 'Feedback Type', 'Comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'Date': current_datetime,
            'User Info': user_info,
            'User Question': user_question,
            'Response Type' : response_type,
            'SQL Query' : sql_query,
            'AI Response' : ai_response,
            'Feedback' : feedback,
            'Feedback Type': feedback_type,
            'Comment':comment
        })
    return jsonify({'message':'Feedback recorded successfully'})

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 

