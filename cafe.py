""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T12 Beginner Begginer Data Structures - Lists and Dictionaries
Compulsory task 

cafe.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following program code will involves an example for cafe menu consisting of 4 items,
including fuctionalities with lists and dictionaires in order to calculate and display total stock worth in the cafe.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Begin of program

# Declaration of list called menu with four variables in it.
menu = ['Cappucino', 'Espresso', 'Double Espresso', 'French Croissant']

# First dictionary called stock with all available units per item in list. 
stock = {
    'Cappucino': 160,
    'Espresso': 580,
    'Double Espresso': 600,
    'French Croissant': 250
}

# Second dictionary called price with the correspoding final cost for client
price = {
    'Cappucino': 3.45,
    'Espresso': 2.99,
    'Double Espresso': 3.20,
    'French Croissant': 3.99
}

# As this task involves numbers, I believe this is the simples way to find the total value.


total_stock = 0

for item in menu: 
     
    worth = stock[item] * price[item]
    print(f'Total price for {item} is {worth}.')
    total_stock += worth

print( "\n" + f'The total stock worth in the cafe is {total_stock}.')

# End of program