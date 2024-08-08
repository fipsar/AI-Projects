from flask import Flask, request, render_template, jsonify, url_for, session, redirect, abort
from scripts import question_classification, final_result,call_openai_gpt,count_tokens
from datetime import datetime
import pandas as pd
import csv
import os
import copy
import os
from scripts_for_edited_sql import ai_result
import logging
import logging_config


app = Flask(__name__)
app.secret_key = 'fipsar'

users = {
    'rdb_user': 'fipsar@123',
    'rdb_test': 'fipsar@456'
}

selected_database = ''
selected_datasource = ''
chat_history = []

def combine_question_for_followup(chat_history):
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
    input_token_combine_question_for_followup = count_tokens(prompt_for_followup)
    # print(input_token_combine_question_for_followup,'input_token_combine_question_for_followup')
    logging.info(f"{input_token_combine_question_for_followup} - input_token_combine_question_for_followup")
    question_response = call_openai_gpt(prompt_for_followup)
    output_token_question_classification = count_tokens(question_response)
    # print(output_token_question_classification,'output_token_question_classification')
    logging.info(f"{output_token_question_classification} - output_token_question_classification")
    return question_response.strip()

def handle_follow_up_question(chat_history,selected_database,selected_datasource, mode):
    print('Entering_handle_follow_up')
    classification_for_follow_up = question_classification(chat_history)
    if mode == 'user_mode':
        print('handling_follow_ques_for_user_mode_condition')
        sql_query, output, visualization_data,database,datasource,mode_type  = final_result(classification_for_follow_up, chat_history,selected_database,
                                                                                            selected_datasource,mode)

        if isinstance(output, pd.DataFrame):
            output.insert(0, 'S.No', range(1, len(output) + 1))
            output = output.to_html(index=False, header=True)
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
    elif mode == 'developer_mode':
        print('handling_followup_for_dev_mode_condition')
        sql_query,col_name, mode = final_result(classification_for_follow_up, chat_history,selected_database,selected_datasource,mode)
        print(sql_query)
             
        response = {'input':chat_history, "sql_query":sql_query, 'mode':mode}
        return response

def handle_new_question(chat_history,selected_database,selected_datasource,mode):
    print('Entering_handle_new_question')
    classification = question_classification(chat_history[0])
    if mode == 'user_mode':
        print('handling_new_ques_for_user_mode_condition')
        sql_query, output, visualization_data,database,datasource,mode_type  = final_result(classification, chat_history[0],
                                                                                            selected_database,selected_datasource,mode)
        if isinstance(output, pd.DataFrame):
            output.insert(0, 'S.No', range(1, len(output) + 1))
            output = output.to_html(index=False, header=True)
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
                response['sql_query'] = response['sql_query'].replace('\n','<br>')
        return response
    
    elif mode == 'developer_mode':
        print('handling_new_ques_for_dev_mode_condition')
        sql_query,col_name, mode = final_result(classification, chat_history[0],selected_database,selected_datasource,mode)
        print(sql_query)
             
        response = {'input':chat_history[0], "sql_query":sql_query, 'mode':mode}
    return response

def handle_new_question_gen_ai(chat_history, edited_sql, selected_database, selected_datasource, mode):
    print('Entering_handle_new_question_gen_ai')
    classification = question_classification(chat_history[0])
    ai_res, visualization_data, database, datasource, mode_type = ai_result(chat_history[0], classification, edited_sql, selected_datasource, selected_database, mode)
    if isinstance(ai_res, pd.DataFrame):
        ai_res.insert(0, 'S.No', range(1, len(ai_res) + 1))
        ai_res = ai_res.to_html(index=False, header=True)
        print(ai_res)
    if visualization_data is not None:
        print('Came to view vis Res')
        print(ai_res)
        print(visualization_data)
        response = visualization_data
    else:
        print('Came to view AI Res')
        print(ai_res)
        response = {'input': chat_history[0], "ai_res": ai_res, 'mode': mode}
    return response

def handle_follow_up_question_gen_ai(chat_history, edited_sql, selected_database, selected_datasource, mode):
    print('Entering_handle_followup_question_gen_ai')
    classification_for_follow_up = question_classification(chat_history)
    ai_res, visualization_data, database, datasource, mode_type  = ai_result(chat_history, classification_for_follow_up, edited_sql, 
                                                                             selected_datasource, selected_database, mode)

    if isinstance(ai_res, pd.DataFrame):
        ai_res.insert(0, 'S.No', range(1, len(ai_res) + 1))
        ai_res = ai_res.to_html(index=False, header=True)
        print(ai_res)

    if visualization_data is not None:
        print(ai_res)
        print(visualization_data)
        response = visualization_data
    else:
        print(ai_res)
        response = {'input':chat_history, "ai_res":ai_res}
    return response

@app.route('/')
def index():
    chat_type = 'new'
    mode = 'user_mode'
    if 'username' not in session:
        return redirect(url_for('login'))   
    return render_template('index.html', chat_type=chat_type, mode=mode, selected_database=selected_database, 
                           selected_datasource=selected_datasource)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message='Invalid username or password')

    return render_template('login.html', message=None)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/edit_query', methods=['POST'])
def edit_query():
    data = request.json
    Input = data['chatInput']
    chat_type = data['chat_type']
    mode = data['mode']
    selected_database = data['databaseSelect']
    selected_datasource = data['datasourceSelect']
    edited_sql = data['hidden_sql']
    def process_chat_get_edited_query(chat_type, Input):
            global chat_history          
            original_chat_history = copy.deepcopy(chat_history)
            try:
                if mode == 'developer_mode' and edited_sql and chat_type == 'new':
                    print('satisfing_dev_edited_query__chat_type_new')
                    chat_history = [Input]
                    response = handle_new_question_gen_ai(chat_history,edited_sql,selected_database, selected_datasource,mode)
                elif mode == 'developer_mode' and edited_sql and chat_type == 'follow_up':
                    print('satisfing_dev_edited_query__chat_type_follow_up')
                    chat_history.append(Input)
                    combined_question = combine_question_for_followup(chat_history)
                    chat_history = [combined_question]
                    response = handle_follow_up_question_gen_ai(chat_history[0],edited_sql,selected_database, selected_datasource, mode)
                return response
            except Exception as e:
                print("error happened--->", e)
                chat_history = original_chat_history
                raise

    try:
        print('First_try_edit_query_route')
        response = process_chat_get_edited_query(chat_type, Input)
    except Exception as e:
        try:
            print('Second_try_edit_query_route')
            response = process_chat_get_edited_query(chat_type, Input)
        except Exception as e:
            error_message = " Sorry, I don't understand your question. Please rephrase and try again."
            response = {
                'error_message': error_message
            }
    
    return jsonify(response)


@app.route('/chat', methods=['POST','GET'])
def chat():
    global chat_history, selected_database, selected_datasource
    if request.method == 'GET':
        Input = session.get('chatInput')
        chat_type = session.get('chat_type')
        mode = session.get('mode')
        selected_database = session.get('selected_database')
        selected_datasource = session.get('selected_datasource')
        edited_sql = session.get('hidden_sql')
        def process_chat_get_method(chat_type, Input):
            global chat_history
            original_chat_history = copy.deepcopy(chat_history)
            try:
                if mode == 'developer_mode' and edited_sql:
                    if chat_type == 'new':
                        chat_history = [Input]
                        response = handle_new_question_gen_ai(chat_history,edited_sql,
                                                              selected_database, selected_datasource,mode)
                    else:
                        chat_history.append(Input)
                        combined_question = combine_question_for_followup(chat_history)
                        print('This is the combined [Rephrased] ques from follow up:--->', combined_question)
                        chat_history = [combined_question]
                        print('This is the chat history of combined ques:--->', chat_history)
                        response = handle_follow_up_question_gen_ai(chat_history[0],
                                                                    selected_database, selected_datasource, mode)
                    return response
            except Exception as e:
                print("error happened--->", e)
                chat_history = original_chat_history
                raise

        try:
            print('First_try_chat_route_GET')
            response = process_chat_get_method(chat_type, Input)
        except Exception as e:
            try:
                print('Second_try_chat_route_GET')
                response = process_chat_get_method(chat_type, Input)
            except Exception as e:
                error_message = " Sorry, I don't understand your question. Please rephrase and try again."
                response = {
                    'error_message': error_message
                }

        return render_template('index.html', response=response, chat_type=chat_type,
                               mode=mode, selected_database=selected_database,
                                selected_datasource =selected_datasource)
    if request.method == 'POST':
        if 'username' not in session:
            return jsonify({'response': 'User not logged in'}), 401
        Input = request.form['chatInput']
        chat_type = request.form['chat_type']
        mode = request.form['mode']
        selected_database = request.form.get('databaseSelect')
        selected_datasource = request.form.get('datasourceSelect')

        session['selected_database'] = selected_database 
        session['selected_datasource'] = selected_datasource
        
        def process_chat_post_method(chat_type, Input):
            global chat_history
                        
            original_chat_history = copy.deepcopy(chat_history)
            try:

                if chat_type == 'new':
                    chat_history = [Input]
                    response = handle_new_question(chat_history,selected_database, selected_datasource,mode)
                else:
                    chat_history.append(Input)
                    combined_question = combine_question_for_followup(chat_history)
                    chat_history = [combined_question]
                    response = handle_follow_up_question(chat_history[0],selected_database, selected_datasource, mode)
                return response
            except Exception as e:
                print("error happened--->", e)
                chat_history = original_chat_history
                raise

        try:
            print('First_try_chat_route_POST')
            response = process_chat_post_method(chat_type, Input)
        except Exception as e:
            try:
                print('Second_try_chat_route_POST')
                response = process_chat_post_method(chat_type, Input)
            except Exception as e:
                error_message = " Sorry, I don't understand your question. Please rephrase and try again."
                response = {
                    'error_message': error_message
                }

        return render_template('index.html', response=response, chat_type=chat_type,mode=mode, selected_database=selected_database,
                                selected_datasource =selected_datasource)
    
@app.route('/record_feedback', methods = ['POST'])
def record_feedback():
    feedback_data = request.json
    classify = 'Natural Language Response' if '.' in feedback_data['aiResponse'] else ('Visual Response' if not feedback_data['aiResponse'] else 'Table Format Response')
    
    user_question = feedback_data['userQuestion']
    response_type = classify
    chat_type = feedback_data['chat_type']
    sql_query = feedback_data['sqlQuery']
    ai_response =feedback_data['aiResponse']
    feedback = feedback_data['feedback']
    feedback_type = feedback_data['feedbackType']
    comment = feedback_data['comment']
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_info = ''

    with open('audit_data.csv','a',newline='') as csvfile:
        fieldnames = ['Date','User Info', 'User Question','Chat Type' ,'Response Type', 'SQL Query', 'AI Response', 'Feedback', 'Feedback Type', 'Comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'Date': current_datetime,
            'User Info': user_info,
            'User Question': user_question,
            'Chat Type':chat_type,
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

