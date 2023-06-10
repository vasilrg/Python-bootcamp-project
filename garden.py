""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T20 Introduction to NLP
Compulsory task 

garden.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following code uses the library Spacy to load a pipeline with the use of which it performs:

- tokenization of each sentence within the created list
- entity recognition for each sentence within the created list
- entity explanation for two identified entities from the performed analysis.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Begin of program

# Import of library to use
import spacy 
# Delaration of variable nlp that contains the core web pipeline
nlp = spacy.load('en_core_web_sm')

# I have used the following link to obtain garden path sentences:
# https://www.youtube.com/watch?v=QdS4vB5pSvw&ab_channel=suzyjstyles

gardenpathSentences = [
    "The old man the boats.",
    "Since Jay always jogs a mile seems like a short distance."
]

# To add sentences to the list above, let's use the following function:

def add_sentence(sentence):
    gardenpathSentences.append(sentence)

new_sentence = "Mary gave the child a Band-Aid."
add_sentence(new_sentence)

new_sentence = "That Jill is never here hurts."
add_sentence(new_sentence)

new_sentence = "The cotton clothing is made of grows in Mississippi."
add_sentence(new_sentence)

# Ensuring that the new function works as expected
print(gardenpathSentences)
print()

# For loop declaration that will go through each sentence and apply the NLP pipeline
for sentence in gardenpathSentences:
   
    doc = nlp(sentence)

    # Tokenize each sentence including punctuation - wanted to see if Spacy will recognice Band-Aid as company but did not:
    # tokens = [token.text for token in doc]
    # print("Tokens:", tokens)

    # Tokenize each sentence with exact text of the token (orth_) and avoiding punctuation
    print("\n" + "New sentence analysis below:" + "\n")
    tokens = [token.orth_ for token in doc if not token.is_punct]
    print("Tokens:", tokens)
    
    # Perform named entity recognition
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    print("Named Entities:", entities)

    # I have also decided to run analysis for each word and check their type:
    print([(w.text, w.pos_) for w in doc])
    print()

    # Identification of stop words: 
    print(f"The identified stop words in the sentence '{sentence}' are:" + "\n")
    for word in doc: 
        if word.is_stop == True:
            print("-", word)
            
print()
#Use of explain function to define entity types:
print("Definition of abbreviation 'GPE':")
print(spacy.explain("GPE"))

print()

print("Definition of abbreviation 'PERSON':")
print(spacy.explain("PERSON"))
print()
# I expected Spacy to recognize the brand Band-Aid:
# Upon researching I found that should I add inc next to the name, the pipeline will categorize the brand

# The first entity I looked up was GPE and have found the following as explanation from Spacy: 
# Countries, cities, states
# I expected Spacy to categorize Mississippi as a river as well, but upon research have found this is due to en_core_wem_sm model

# The second entity I looked up was PERSON and have found the following as explanation from Spacy: 
# People, including fictional
# To my surprise the pipeline can even recognize fictional people such as Peter-Pan, Harry Potter, etc

# End