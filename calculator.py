""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T09 Towards Defensive Programming II - Exception Handling
Compulsory task 

calculator.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The code is used to create an application, which asks the user for two numbers and the  operation they wish to action.
The program then stores the user input info into a text file and returns the equation answer.

Upon completion, the code promtps the user should they wish to conduct another operation or have all data displayed on screen.
Should user enters unkown name of file, the program will return an error. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Begin of program
# For the purpose of this application, I have decided to define the all the code into one function named simple_calculator()


def simple_calculator():
    while True: 

        print("Welcome to the simple calculator!")
        print("Using this tool, you will be able to perform simple arithmetic operations with two numbers.")

        # Defensive programming against invalid data typed by user
        try:
            a = float(input("Please enter first number: ").strip())
            b = float(input("Please enter second number: ").strip())
            operation = input("Please enter operation (e.g +, -, x, / : ")
            
            if operation == "+":
                result = a + b
                
            elif operation == "-":
                result = a - b

            elif operation == "x": 
                result = a * b
            
            elif operation == "/":
                if b == 0:
                    print("\n" + "Cannot divide by ze8ro. Start over!" + "\n")
                    continue
                result = a / b

            elif operation not in ("+", "-", "x", "/"):

                # Upon testing, variable result displays the below string into the text file too
                result = "not calculated, due to invalid data entered."
                print("\n" + "Invalid input entered for operation!")

        except ValueError:
            print("Invalid format entered for number!\nStart over!"+ "\n")
            continue
            
        print(f"Your result is {result}")
        print("Your result history is being saved in a text file called 'calculator'.")

        # Declaration of variable which will be used to write and update the text file called calculator
        equation = f'{a} {operation} {b} = {result}'

        # Use of the append method to create and update the text file, as it is expected that the user will type more than one calculation
        file = open('calculator.txt', 'a')
        file.write(equation + "\n")
        file.close()

        print("\n" + "Enter 1 for another calculation, or 2 to see your equations from a text file.")
        prompt = int(input("Please enter your choice: "))    
        if prompt == 1:
            print()
            continue

        # Defensive programming against invalid name of text file typed by user
        elif prompt == 2:
            while True:

                    filename = input("Please enter file name: ")

                    if filename == "calculator":

                        file = open('calculator.txt', 'r')
                        print(file.read() + "\n")
                        file.close()
                        break

                    else:
                        
                        print("The file you are trying to open does not exist.")
                    continue
        break

# Call of the defined function simple_calculator()
simple_calculator()