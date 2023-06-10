""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T03 Beginner Control Structures - If, Elif, Else & the Boolean Data type
Compulsory task 

full_name.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following code is asking the user to enter their name. 
Depending on the requirements in the compulsory task and what the user types, the program will display different messages. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# python code for a program to check the user has entered a full name

# declaraiton of a variable userName containing method input() to trigger the user to type their f&l names.
# I have also used escape function in the method to display the input user result on new line
# useful way of strip() is to clear any spacing typed by mistake from user -- I learned about this from the w3schools website: https://www.w3schools.com/python/ref_string_strip.asp
userName = input("Please enter your first and last names: \n").strip()

# declaration of a second variable called fNamelName that will take as value the variable userName encapuslated with method split() within len(), as we want to check if the user has typed more than one name by separating with " ".
# I am using method split () in order to separate any values typed by user, https://www.w3schools.com/python/ref_string_split.asp
fNamelName = len(userName.split())

# declaration of a second variable Message that contains thank you message if all conditions the program are respected.
Message = '"Thank you for entering your name."'
print()

# opening on the if function - first condition using len() method to check if the user has typed anything using operator equals to equals to

if len(userName) == 0:

    # if the above if statement is true, then value Message would take the following value 
    Message = '"You haven`t entered anything. \nPlease enter your full name"'
    #print statement returning  a message that nothing has been typed by user
    print(Message)

# elif statement: condition 2 using lenght method to check if the user has typed less than 4 characters with the use of less than operator
elif len(userName) < 4:

    # variable Message would take the below value, should 1st elif statement is true 
    Message = '"You have entered less than 4 characters. \nPlease make sure that you have entered your name and surname."'
    # print statement returning a message that the name is too short (less than 4 characters but in two statements such as "j s")
    print(Message)

# elif statement that will check if the information typed by user consists of two separate statements
elif fNamelName < 2:

     # variable Message would take the below value, should above elif statement is true 
    Message = '"Please make sure that you insert your first name and surname."'
    # print statement returning a message that the name is without a surname 
    print(Message)

# elif statement using lenght method to check if the user has typed more than 25 characters with the help of bigger than operator
elif len(userName) > 25:

    # variable Message would take the below value, should 2nd elif statement is true 
    Message = '"You have entered more than 25 characters. \nPlease make sure that you have only entrered your full name."'
    # print statement returning a message that the name is too long (more than 25 characters)
    print(Message)

# end of the if statement, meaning the avobe are the only requirements set for the program to check 
else:
    # print statement returning successfully entered name
    print(Message)

# end of program
