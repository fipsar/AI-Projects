import pandas as pd
import re
import openpyxl

# def parse_log_file(log_file):
#     """
#     Parses the log file to extract relevant information.
    
#     Parameters:
#     - log_file (str): The path to the log file.
    
#     Returns:
#     - dict: Extracted data with labels and values.
#     """
#     data = {}
#     with open(log_file, 'r') as file:
#         for line in file:
#             # Customize parsing based on your log format
#             if 'example_function' in line:
#                 parts = line.split(' - ')
#                 timestamp, log_level, message = parts[0], parts[2], parts[3].strip()
#                 data[timestamp] = message
#     return data

# def save_to_excel(data, filename):
#     """
#     Saves data to an Excel file.
    
#     Parameters:
#     - data (dict): The data to save.
#     - filename (str): The name of the Excel file.
#     """
#     df = pd.DataFrame(data.items(), columns=['Timestamp', 'Message'])
#     df.to_excel(filename, index=False)

# # Example usage
# log_file = 'app.log'
# data = parse_log_file(log_file)
# save_to_excel(data, 'logs.xlsx')

def parse_log_file(log_file):
    data = []
    with open(log_file, 'r') as file:
        for line in file:
            # Regex to match token-related logs
            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - root - INFO - (\d+) - (.+)', line)
            if match:
                timestamp, token_count, message = match.groups()
                data.append((timestamp, message, int(token_count)))
    print("Parsed Data:", data)  # Debug print
    return data

# def save_to_csv(data, filename):
#     if not data:
#         print("No data to save.")
#         return
#     df = pd.DataFrame(data.items(), columns=['Timestamp', 'Message'])
#     df.to_csv(filename, index=False)
#     print(f"Data saved to {filename}")
def save_to_csv(data, filename):
    """
    Saves data to an Excel file.
    
    Parameters:
    - data (list of tuples): The data to save.
    - filename (str): The name of the Excel file.
    """
    if not data:
        print("No data to save.")
        return
    
    df = pd.DataFrame(data, columns=['Timestamp', 'Message', 'Token Count'])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

log_file = 'Gen_AI/app.log'
data = parse_log_file(log_file)
save_to_csv(data, 'Gen_AI/logs.csv')




