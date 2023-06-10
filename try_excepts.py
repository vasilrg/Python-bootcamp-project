""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T04 Logical Programming - Operators 
Compulsory task 

task3.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The folloging program checks and displays if a triathlon contestant wins a prize. 
The code asks the user to enter their times each for the three parts of the thriathlon, which are swimming, cycling and running. 

The qualifying time is 100 minutes and depending if within 5, 10 or more minutes, the code displays the relevant prize / no price message.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Python program that displays award based on timed performance that is submitted by the user


# Originally the below comment lines contain the code that I had in mind before trying try and except, which is more convenient

#declaration of three variables that have a value of int data type (numerical), which will be submitted by contestant (minutes), use of input ()
#swimmingResult = int(input("Please submit your swimming time in minutes only and press enter: \n"))
#print()

#cyclingResult = int(input("Please submit your cycling time in minutes only and press enter: \n"))
#print()

#runningResult = int(input("Please submit your running time in minutes only and press enter to check if you witn a award from the triathlon: \n"))
#print()

# Declaration of try and except to run through the data typed by contestant and check if numerical. I researched about the exit() that terminates the loop immediately
try:
    swimming_time = int(input("Please submit your swimming time in minutes only: \n"))
    cycling_time = int(input("Please submit your cycling time in minutes only: \n"))
    running_time = int(input("Please submit your running time in minutes only: \n"))
except:
    print("Please submit your result in numbers only for each event time.")

#definition of a calculating method, use of def () - upon reading about this, setting a calculating method once can be called at any point, should the code is long and requires repetitive actions
def totalResult(swimming_time, cycling_time, running_time):
    result = swimming_time + cycling_time + running_time
    return result

#declaration of qualifying time named variable of type integer
qualifyingTime = 100

#award variable is declared empty as it will take different values in between if and elif switch statements
award = " "

#Declaration of switch statements containing the requirements set in compulsory task
if totalResult(swimming_time, cycling_time, running_time) <= qualifyingTime:
    award = "within the qualifying time. \nYour award is Provincial Colours."

elif totalResult(swimming_time, cycling_time, running_time) <= qualifyingTime + 5:
    award = "within 5 minutes of qualifying time. \nYour award is Provincial Half Colours."

elif totalResult(swimming_time, cycling_time, running_time) <= qualifyingTime + 10:
    award = "within 10 minutes of qualifying time. \nYour award is Provincial Scroll."

else: 
    award = "exceeding the qualifying time for an award. \nYou do not get an award on this triathlon."

# print statement that displays the total result and the respective award, use of f- function, rather than .format() method here.
# the totalResult() is listed rather than the result variable as the user is yet to submit values to the three events
print(f"Your total time on this triathlon is {totalResult(swimming_time, cycling_time, running_time)} minutes, which is {award}")

#End of Program
