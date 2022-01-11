# Adam Gardell & Ludde Lahrl
#
# Filters answers in reddit data by matching id in manually stripped question file.

import pandas as pd
import io
import csv

df = pd.read_csv('reddit_questions.csv', error_bad_lines=False, usecols=['id'], sep=';')

id = df['id'].tolist()

df2 = pd.read_csv('reddit_answers.csv',error_bad_lines=False, sep=';')

df2 = df2[df2['q_id'].isin(id)]

df2.to_csv('filteredAnswers.csv', sep=';')





