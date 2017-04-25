#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

count = 0
c = 0
persons_of_interest = []
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
for k in enron_data:
	if enron_data[k]['salary'] != 'NaN':
		#print enron_data[k]['email_address']
		c += 1
	if enron_data[k]['poi'] == True:
		count += 1
		persons_of_interest.append(k)
		if 'fastow'.upper() in k:
			pass#print enron_data[k]
#print enron_data['COLWELL']
print count

#print persons_of_interest

#print enron_data["LAY KENNETH L"]["total_payments"]
#print c
#print sum([1 for key in enron_data.keys() if enron_data[key]['salary'] != 'NaN'])
#print sum([1 for key in enron_data.keys() if enron_data[key]['email_address'] != 'NaN'])
print sum([1 for key in enron_data.keys() if enron_data[key]['total_payments'] == 'NaN'])
#print len(enron_data)
print sum([1 for key in enron_data.keys() if enron_data[key]['poi'] == True and enron_data[key]['total_payments'] == 'NaN'])