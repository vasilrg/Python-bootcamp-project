name = str(input("Please type your first and last names: ")) #declaration of variable type string name which contains method input() to invite the user to type their names
print() #empty row to for better appereance
age = int(input("Now enter your age: ")) #declaration of variable age of type integer which will take only numbers as value, I have tested trying to put letters and got and error. 
# the variable age contains the intput() method which will act as system prompt to the user
print()
print("{} is {} years old".format(name,age)) #tried this using the examples file and it looks great!
print()
print("This is your name: ",name) #This is how I was originally planning to print the vairables 
print("This is your age: ",age)
print("Hello world!") #print statement containing string 