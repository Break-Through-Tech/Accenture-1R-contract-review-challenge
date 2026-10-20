import json
import statistics

with open("data/cuad/train_separate_questions.json", "r") as file:
    dataset = json.load(file)

contracts = dataset["data"]

print("Number of contracts:", len(contracts))

print("\nChecking for missing data:")

missing_titles = 0
missing_paragraphs = 0
missing_context = 0

for contract in contracts:
    if not contract.get("title"):
        missing_titles += 1

    if not contract.get("paragraphs"):
        missing_paragraphs += 1

    for paragraph in contract.get("paragraphs", []):
        if not paragraph.get("context"):
            missing_context += 1

print("Missing titles:", missing_titles)
print("Contracts with missing paragraphs:", missing_paragraphs)
print("Paragraphs with missing contract text:", missing_context)

contract_lengths = []

for contract in contracts:
    text = ""

    for paragraph in contract["paragraphs"]:
        text += paragraph["context"]

    contract_lengths.append(len(text))

print("\nContract Length Summary:")
print("Shortest contract:", min(contract_lengths))
print("Longest contract:", max(contract_lengths))
print("Average contract:", sum(contract_lengths) / len(contract_lengths))

median_length = statistics.median(contract_lengths)

q1 = statistics.quantiles(contract_lengths, n=4)[0]
q3 = statistics.quantiles(contract_lengths, n=4)[2]

iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

print("\nOutlier Check:")
print("Median contract length:", median_length)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

outliers = []

for i, length in enumerate(contract_lengths):
    if length < lower_limit or length > upper_limit:
        outliers.append((contracts[i]["title"], length))

print("Number of outliers:", len(outliers))

for title, length in outliers:
    print(title, "-", length)

print("\nChecking for duplicate contracts:")

titles = []

for contract in contracts:
    titles.append(contract["title"])

duplicate_count = len(titles) - len(set(titles))

print("Number of duplicate contract titles:", duplicate_count)

print("\nContract Word Count:")

word_counts = []

for contract in contracts:
    text = ""

    for paragraph in contract["paragraphs"]:
        text += " " + paragraph["context"]

    words = text.split()
    word_counts.append(len(words))

print("Shortest contract in words:", min(word_counts))
print("Longest contract in words:", max(word_counts))
print("Average contract in words:", sum(word_counts) / len(word_counts))
print("Median contract in words:", statistics.median(word_counts))

print("\nQuestion Categories:")

categories = {}

for contract in contracts:
    for paragraph in contract["paragraphs"]:
        for qa in paragraph["qas"]:
            category = qa["id"].split("__")[1].rsplit("_", 1)[0]

            if category in categories:
                categories[category] += 1
            else:
                categories[category] = 1

print("Number of categories:", len(categories))

for category, count in categories.items():
    print(category, "-", count)

most_common = max(categories, key=categories.get)
least_common = min(categories, key=categories.get)

print("\nMost common category:", most_common, "-", categories[most_common])
print("Least common category:", least_common, "-", categories[least_common])

print("\nAnswer Analysis:")

total_questions = 0
questions_with_answers = 0
questions_without_answers = 0

for contract in contracts:
    for paragraph in contract["paragraphs"]:
        for qa in paragraph["qas"]:
            total_questions += 1

            if len(qa["answers"]) > 0:
                questions_with_answers += 1
            else:
                questions_without_answers += 1

print("Total questions:", total_questions)
print("Questions with answers:", questions_with_answers)
print("Questions without answers:", questions_without_answers)