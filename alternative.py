""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T11 String Handling
Compulsory task 

alternative.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following python code will display the sentence "Hello World" by "HeLlO WoRlD".

As well, the program will then display the sentence "I am learning to code" by "i AM learning TO code".

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Begin of program 

# Declaration of two string variables - phrase which contains original wording and alternative, containing updated wording
phrase = "Hello World"
alternative = " "

for x in range(len(phrase)): 

    # If statement to check if every second index in the string (converted into integer) is even 
    if x % 2 == 1:

        alternative += phrase[x].lower()
    
    else: 

        alternative += phrase[x].upper()

print("\n" + alternative)

# Declaration of two strings and a list as well as split method in order to separate each words in variable statement
statement = "I am learning to code"
new_statement = statement.split(" ")
update_sentence = []

for z in range(len(new_statement)):

    if z % 2 == 1:

        update_sentence.append(new_statement[z].upper())
    
    else:

        update_sentence.append(new_statement[z].lower())

# Use of join function in order to bring back the updated list 
final_wording =" ".join(update_sentence)

print("\n" + final_wording)

