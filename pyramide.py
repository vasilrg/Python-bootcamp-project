""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T08 Beginner Control Structures - For Loop
Compulsory task 

task3.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The program code below displays the following:

*
**
***
****
*****

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Begin of program

#Declaration of string variable that holds null value
asterix = " "

# The iteration variable i will loop five times and for each loop it will add the star sign to the asterix variable(which has null value)
for i in range(0, 5):

    asterix = asterix + "*"
    print(asterix)
