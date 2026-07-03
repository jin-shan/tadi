import json
from sklearn.model_selection import train_test_split

with open('train_original.json') as f:
    data = json.load(f)

train, val = train_test_split(data, test_size=0.1, random_state=42)

print("Training data size: ", len(train))
print("Validation data size: ", len(val))

with open('test.json') as file:
    test = json.load(file)

print("Test data size: ", len(test))

print
with open('train.json', 'w') as file:
    json.dump(train, file, indent=4)

with open('val.json', 'w') as file:
    json.dump(val, file, indent=4)

print("Training data saved to train.json")
print("Validation data saved to val.json")