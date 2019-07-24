#imports
import requests
import csv  
from bs4 import BeautifulSoup as bs

#creating an object and getting contents of the main file
html = "https://karki23.github.io/Weather-Data/"
page = requests.get(html+"assignment.html")
soup = bs(page.content,'html.parser')

#searches for all hyperlinks(city names) and stores it in cities
l = soup.find_all('a')  
cities = []
for i  in l:
    cities.append(str(i).split("\"")[1])

#Scrapping from every city's page one by one and storing row contents in outputrows
outputrows = []
for x in cities:
    print(x)
    count =0 
    pg = requests.get(html+x)
    sp = bs(pg.content,'html.parser') 
    table = sp.find('table')
    every = table.find_all('tr')
    for tr in every: #checking for every row content
        columns = tr.find_all('td')
        row = []
        for column in columns:
            row.append(column.text)
            #print(column.text)
        if not count:
            count=1
        else:
            outputrows.append(row)
            
#writing the data to weather.csv
file = 'weather'
with open(file+'.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([['Date','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am',"Temp3pm","RainToday","RISK_MM","RainTomorrow"]])
    writer.writerows(outputrows)
