""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T07 Beginner Control Structures - While Loop
Compulsory task 

while.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following program code will continiously keep asking the user to insert a number of their choice. 

The above functionaility will be repeated until the user submits -1 as number. 

Then the program will stop the cycle and will calculate the average of all numbers inserted, exlcuding -1.

Should the user submit data type different from an integer - the program will return an error message. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Begin of program

# Definition of vatiables needed for the program
user_input = int(input("Please enter any number to start the program. Enter number -1 to exit the program: "))
score = 0 

# I decided to use list to store all the integers entered by the user
array = []
array.append(user_input)


while user_input != -1: 

    # The program will keep calculating the total sum of numbers the user enters, as long as they differ from -1
    # The program will also keep updating the list with all numbers entered by the user 
    # The program will keep looping, prompting the user for another number, until they enter -1
    score += user_input 
    user_input = int(input("Please enter any number to continue. Enter -1 to calculate average sum: "))
    array.append(user_input)

    if user_input == -1:
        
        # As the number we wish not to include will always be the last one, pop() is great use in the case
        array.pop()
        average = score / len(array)
        print()
        print(f"""You endered the nubmer -1, which will now close the program and callculate the average sum of the number you entered.""")
        print(f"The average sum for the entered numbers {list(array)} is {average}.")
        print()

        # Defining another variable that would keep the user's choice for another calculation
        session = input("Would you like to start another session? (yes/no): ").lower()
        print()

        if session == "yes":

            user_input = int(input("Please enter any number to start the program. Enter number -1 to exit the program: "))

        elif session != "yes":

            print("Thank you for using our services! Goodbye!")
            break
        
# End of program
    