#code to be run in google colab
!pip install pandas openpyxl --upgrade
import pandas as pd
from datetime import datetime, timedelta

# Excel file ka path
excel_file = '/content/Running_Contract_Details.xlsx'

# Pandas se data read karo
df = pd.read_excel(excel_file)

# "Action to be Taken From" column ko datetime objects me convert karo, mixed formats ko handle karo
df['Action to be Taken From'] = pd.to_datetime(df['Action to be Taken From'], format='mixed', errors='coerce')

# Get indices of rows with NaT values
nat_indices = df[df['Action to be Taken From'].isnull()].index

# Convert original values at NaT indices to numeric, handling errors
for index in nat_indices:
  try:
    df.loc[index, 'Action to be Taken From'] = pd.to_datetime(float(df.loc[index, 'Action to be Taken From']), unit='D', origin='1899-12-30', errors='coerce')
  except (ValueError, TypeError):
    pass  # Skip if conversion fails


# Table ko sort karo "Action to be Taken From" column se latest to farthest date ke hisab se
df = df.sort_values(by=['Action to be Taken From'], ascending=True, na_position='last')

# Aaj ki date
today = datetime.now()

# HTML me convert karo, full height and width ke liye style add karo, aur rows ko highlight karo
def format_row(row):
    if pd.notna(row['Action to be Taken From']) and (row['Action to be Taken From'] - today) <= timedelta(days=15):
        return 'background-color: #ffdddd; font-weight: bold;'  # Light red background aur bold font
    else:
        return ''

html_table = df.style.apply(lambda row: [format_row(row)] * len(row), axis=1) \
                   .format({'Action to be Taken From': lambda date: date.strftime("%Y-%m-%d") if pd.notna(date) else ''}) \
                   .to_html()  # Use to_html() instead of render()

html_table = f"""
<!DOCTYPE html>
<html>
<head>
<style>
body, html {{
  height: 100%;
  margin: 0;
}}

table {{
  width: 100%;
  height: 100%;
  border-collapse: collapse;
}}

td, th {{
  border: 1px solid black;
  padding: 8px;
}}
</style>
</head>
<body>

{html_table}

</body>
</html>
"""

# HTML file me save karo
with open('table.html', 'w') as f:
    f.write(html_table)

print("Data 'table.html' naam ke ek static HTML page me convert aur save ho gaya hai, jisme dates formatting ke saath hain.")
