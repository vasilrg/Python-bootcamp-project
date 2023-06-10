""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T21 Semantic Similarity 
Compulsory task 

semantic.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

Comparison of different words with the help of Spacy.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
import spacy 
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print()

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Upon observing the output from the above for loop, I was quite surprised to see that cat and apple would have any smililarity at all
# It is also quite interesting to see that, knowingly cat and monkey are both animals, the similarity coefficient exceeds the 0.5 marker,
# which often in math gets rounded by the higher, which would mean 1, i.e both animals
# Compared to the monkey, the cat seems to not have much similarity neither with apple nor banana

# The example that I will use this model with includes person, tiger, passion fruit, mango

w1 = nlp("person")
w2 = nlp("tiger")
w3 = nlp("watermelon")
w4 = nlp("cucumber")

print()
print(w1.similarity(w2))
print(w2.similarity(w4))
print(w3.similarity(w1))
print(w4.similarity(w3))
print()

# With the example above I was keen to see if spacy will categorise person as kind of animal and show higher similarity
# between person and tiger, like between cat and monkey, however I am now finding out there is barely a comparison between the two
# I was also quite interested to see what the similarity between a fruit and a vegetable would be, which is quite high!

tokens_example = nlp('person tiger watermelon cucumber')

for token1 in tokens_example:
    for token2 in tokens_example:
        print(token1.text, token2.text, token1.similarity(token2))
       
print()

# According to Spacy, tiger seems to be more similar to cucumber rather than person

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
      similarity = nlp(sentence).similarity(model_sentence)
      print(sentence + " - ",  similarity)

# When I ran the example file using both models core_web_sm and core_web_md, I have noticed the following differences: 
# Compared to the assembled sections, each corresponding coefficient seems to be lower in core_web_sm, compared to web_md
# I have also noticed that my terminal prompted me a message, advising core_web_sm does not have any word vectors loaded, 
# which to me means it is a much simpler model, complared to the web_md one