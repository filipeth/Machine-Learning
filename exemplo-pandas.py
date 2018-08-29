import numpy as np
import pandas as pd
import sqlite3

# pais_nome = np.array(['Great Britain', 'China', 'Russia', 'United States', 'Korea', 'Japan', 'Germany'])
# vetor = pd.Series(pais_nome)
#
# print(vetor)

# dici = pd.Series([29, 38, 24, 46, 13, 7, 11], index=['Great Britain', 'China', 'Russia', 'United States', 'Korea', 'Japan', 'Germany'])
# print(dici.loc['China'])
#
# print(dici.iloc[0])
#
#
#
# primeiro = pd.Series([1,2,3,4], index=['a', 'b', 'c', 'd'])
# segundo = pd.Series([10,20,30,40], index=['a', 'b', 'c', 'd'])
# total = primeiro + segundo
#
# print(total)
#
# lista = {'cidade': ['toronto', 'bh'], 'ano':[2000, 2010]}
# ve = pd.DataFrame(lista)
#
# print(ve.describe())

#
# participantes = pd.Series([200, 201, 202], index=['a', 'b', 'c'])
# doideri = pd.Series(['eu', 'vc', 'tu'], index=['a', 'b', 'c'])
# ser = pd.DataFrame({'olha que legal': participantes, 'vamo ve':doideri})
# print(ser)


# vetor = np.array([2010, 2012, 2015])
# dic = {'ano': vetor}
# df = pd.DataFrame(dic)
# print(df.head(1))
#
# print(df.tail(1))
#
# print(df.index)

# def standardize(test):
#     return (test - test.mean()/ test.std())
#
# def standardize_scores(dataf):
#     return (dataf.apply(standardize))
#
# teste = pd.DataFrame({'teste1': [95, 84, 73, 88, 82, 61], 'teste2': [74, 85, 82, 73, 77, 79]}, index=['jack', 'lewis', 'patrick', 'rich', 'kelly', 'paula'])
# # print(teste.sort_values('teste1'))
#
# # print(standardize(teste['teste1']))
# print(standardize_scores(teste))

create_table = """CREATE TABLE student_score
(Id INTEGER, Name VARCHAR(20), Math REAL, Science REAL);"""

insertSQL = [(10, 'Jack', 85,92),(29,'Tom', 73, 89), (65, 'Ram', 65.5, 77),
             (5, 'Steve',55, 91)]
insert_stat = "Insert into student_score values (?,?,?,?)"


executeSQL = sqlite3.connect(':memory:')
executeSQL.execute(create_table)
executeSQL.executemany(insert_stat, insertSQL)
executeSQL.commit()

SQL_query = executeSQL.execute('select * from student_score')
resultset = SQL_query.fetchall()
print(resultset)
df_stutend = pd.DataFrame(resultset, columns=list(zip(*SQL_query.description))[0])
print(df_stutend)