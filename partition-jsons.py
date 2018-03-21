import json


with open('data/dev-v1.1.json') as dev_json:
	data = json.load(dev_json)
print(data['data'][0]['paragraphs'][0]['qas'][0])


for i in range(len(data['data'])):
	for j in range(len(data['data'][i]['paragraphs'])):
		data['data'][i]['paragraphs'][j]['qas'] = list(filter(lambda x: 'how' in x['question'].lower().split(), data['data'][i]['paragraphs'][j]['qas']))


		
with open('data/dev-v1.1_how.json', 'w') as outfile:
	json.dump(data, outfile)

