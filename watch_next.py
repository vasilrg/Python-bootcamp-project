""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T21 Semantic Similarity 
Compulsory task 2

watch_next.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following python code represents a function that compares the similarity of a given movie and description to a text file. 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Begin of program

def compared_movies():
    import spacy

    nlp = spacy.load('en_core_web_md')

    # Reading through the file and printing its content
    with open('movies.txt', 'r') as file:
        lines = file.readlines()

    # Movie description to compare
    movie_to_compare = "Planet Hulk, Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator"
    movie_name = "Planet Hulk"

    model_sentence = nlp(movie_to_compare)
    print()
    print(f"Because you watched '{movie_name}'")
    print()

    # Tokenize and compare each movie in the file
    for line in lines:

        # i removed the new line code form the txt file 
        sentence = line.strip()  
        doc = nlp(sentence)

        tokens = [token.orth_ for token in doc if not token.is_punct]

        # I have identified some tokens to use in my final print statement
        token1 = doc[0].text
        token2 = doc[1].text
        #print(tokens)

        # Compute similarity between the movie description and the current movie in the file
        similarity = model_sentence.similarity(doc)

        print(f"{sentence}\n{token1} {token2} is {round(similarity, 2)} (out of 1) similar to '{movie_name}'.")
        print()

# Call the function
compared_movies()

