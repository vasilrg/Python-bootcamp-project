""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T05 Capstone Project I - Variables and Control Structures
Compulsory task 

finance_calculators.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following code asks the user to either type 'investment' or 'bond'.
If typed otherwise, an error message would be displayed and the program would restart displaying greeting message.

If typed 'investment':
    - the program will ask the user to type values for money, rate, years (if typed otherwise - error message + restart)
    - the program will ask the user to either type 'simple' or 'compound' (if typed otherwise - error message + restart)
    - depending on user selection, the program will calculate total amount of investment and will display the vlaue 

If typed 'bond':
    - the program will ask the user to type values for house, rate and months 
    - the program will calculate monthly repayment amount and will display it for the user

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Begin of program

# Call of Library
import math

# Declaration of a loop for the program to check the following conditions:
while True:

    # Print statements matching the displayed message in the task
    print("""investment - to calculate the amount of interest you'll earn on your investment\nbond\t   - to calculate the amount you'll have to pay on a home loan""")
    print()

    # Declaration of variables that will be called throught the loop / code 
    prompt = ("Please enter either 'investment' or 'bond'.")

    info = ("Thank you for your selection!\nIn order to proceed we need the following: \n")

    # By the call of lower() method, the program will always accept and convert the typed values by the user, irrerevantly of caps type; strip() to remove any spacing from input
    selection = input("Enter either 'investment' or 'bond' from the menu above to proceed: ").lower().strip()
    print()

    if selection == "investment":

        print(info)

        # Variables required to calculate total value 
        # Input will be divided by 100 to convert into %
        # Use of Try / Except to check if the values are matching the data types configured
        try:
            money = float(input("Please enter the amount of money you wish to deposit: ").strip())
            rate = float(input("Please insert the percentage of the interest rate as number only (without the '%'): ").strip()) / 100
            years = int(input("Number of years you plan to invest: ").strip())
            interest = str(input("Would you like to assign 'simple' or 'compound' interest rate: ").strip().lower())
         
            # For value and repayment results below I am using round() to only display two numbers past decimal point - e.g 3.14
            if interest == "simple":

                #Value calculated using the above values with simple interest formula
                value = round((money * (1 + rate * years)), 2)
                print()
                print(f"The total amound you would get in return for investing {money} for {years} years is: £{value}")
                print()

                # Prompt message to ask user for another calculation
                another_calculation = input("Would you like to make another calculation?: ").lower()
                
                if another_calculation == "yes":
                    print()
                    continue

                elif another_calculation != "yes":
                    print()
                    print("Thank you for using our services!")
                break

            elif interest == "compound":

                # Value of investment with compound interest formula
                value = round((money * math.pow((1 + rate), years)), 2)
                print()
                print(f"The total amound you would get in return for investing {money} for {years} years is: £{value}")
                print()

                another_calculation = input("Would you like to make another calculation?: ").lower()
            
                if another_calculation == "yes":
                    print()
                    continue

                elif another_calculation != "yes":
                    print()
                    print("Thank you for using our services!")
                break

            elif interest != "simple" or interest != "compound":

                # Error message if interest is neither of the above - the loop would restart
                print("""
                
 Please ensure you enter either 'simple' or 'compound' for interest.\nPlease start over!

 """)

        # The program would print the error message should the user inserts data types different from the configured
        # The loop would restart again
        except ZeroDivisionError():
                print("""

 It looks like you have provided non-numerical information for one of the following:\n
 * amount of money you wish to invest\n\n* interest rate\n\n* years of investment\n\nPlease start over!

 """)
                         
        
    elif selection == "bond":

        print(info)

        try:
            # Variables requied for the program to calculate repayments formula
            # Input will be divided by 100 to convert into % - the result will be then divided by 12
            house = float(input("Please enter value of the house (e.g 100000): ").strip())
            rate_bond = int(input("Please insert the percentage of the interest rate as number only (e.g 7): ").strip()) / 100
            rate_annual = rate_bond / 12
            months = int(input("Number of months you plan to replay the bond (e.g 120): ").strip())

            # Repayment formula calculations
            repayment = round(((rate_annual * house) / (1 - (1 + rate_annual)**(-months))), 2)
            print(f"The monthly repayment for this bond is: £{repayment}")
            print()


            another_calculation = input("Would you like to make another calculation?: ").lower()
                
            if another_calculation == "yes":
                    print()
                    continue

            elif another_calculation != "yes":
                print()
                print("Thank you for using our services!")
                break

        except ZeroDivisionError():
            print(""" 

 It looks like you have provided non-numerical information for one of the following:\n
 * value of property you wish to bond\n\n* interest rate\n\n* months of repayment\n\nPlease start over!

 """)

    # Error message if user types different from either 'investment' or 'bond' - the loop restarts
    else:
        print(prompt)
        
# End of program    



