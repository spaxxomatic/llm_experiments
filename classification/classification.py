from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
import transformers
transformers.logging.set_verbosity_info()

oracle = pipeline(model="deepset/roberta-base-squad2")

def classify(sequence, labels):
    output = classifier(sequence, labels, multi_label=True)
    print()
    print(res['sequence'])
    for index in range(len(res['labels'])):
        print(f"  IS {res['labels'][index]}  SCORE: {res['scores'][index]}")
    print_classification(output)

def answer_question(context, question):
    output = oracle(question=question, context="My name is Wolfgang and I live in Berlin")    
    print (context, "\n", question, response['answer'], response['score'])


answer_question("My name is Wolfgang and I live in Berlin", "Where do I live?")

sequence_to_classify = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU. Sie ist der Kanzler Deutschland"
candidate_labels = ["politics", "economy", "entertainment", "environment", "travel"]
classify(sequence_to_classify, candidate_labels)

answer_question(sequence_to_classify, "Who is Merkel?")
answer_question(sequence_to_classify, "Wer ist Merkel?")

sequence_to_classify = "I want to spend my vacation in Roma"
candidate_labels = ["politics", "travel", "city", "environment", "intent"]
classify(sequence_to_classify, candidate_labels)

answer_question(sequence_to_classify, "Which city?")

sequence_to_classify = "Book it for friday"
candidate_labels = ["date", "question", "intent", "education", "order"]
classify(sequence_to_classify, candidate_labels)

question="Which city?"
sequence_to_classify = "Next time i want to visit Munich"
answer_question(question=question, context=sequence_to_classify)

output = classifier(sequence_to_classify, candidate_labels, multi_label=True)
print_classification(output)


sequence_to_classify = "How long does it take to get to the airport? "
candidate_labels = ["travel", "question", "duration", "education", "order"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=True)
print_classification(output)

question = sequence_to_classify
output = oracle(question=question, context="Airport of Frankfurt")
print_oracle(question, output)

sequence_to_classify = "Wie lange brauche ich zum Flughafen? "
candidate_labels = ["travel", "question", "duration", "education", "order", "instruction"]

output = classifier(sequence_to_classify, candidate_labels)
print_classification(output)

sequence_to_classify = "Vieviel kostet eine Buchung für ein Tag?"
candidate_labels = ["politics", "question", "price", "duration", "booking intent"]
output = classifier(sequence_to_classify, candidate_labels)
print_classification(output)

question="What is the intent?"
output = oracle(question=question, context=sequence_to_classify)
print_oracle(question, output)

question="How long to stay?"
output = oracle(question=question, context=sequence_to_classify)
print_oracle(question, output)

question="what is the question?"
output = oracle(question=question, context="What room types do you have?")
print_oracle(question, output)
