""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T14 Beginner Programming with Functions - Defining Your Own Functiuons
Compulsory task 

holiday.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following python code will calculate the total cost of an user's holiday, which would include the price for the flights,
hotel stay and car rental. 

The proram will prompt the user to input a destination of their choice, number of nights they wish to spend at a hotel, 
and how many days they would like to rent a vehicle for. 

This python code is written with use of functions. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Begin of program

# declaration of a denined function plane_cost that has pre-defined values for ticket price 
def plane_cost(city_flight):
    
    if city_flight == "amsterdam":
        plane_price = 250
        
    elif city_flight == "new york":
        plane_price = 650
       
    elif city_flight == "miami" or city_flight == "toronto":
        plane_price = 700
       
    elif city_flight == "istanbul":
        plane_price = 400
    
    else: 
        print("We don't offer flights to this desitnation.")
        plane_price = 0 

    return plane_price

# declaration of a denined function hotel_cost with pre-defined values for hotel price 
def hotel_cost(num_nights):

    if num_nights == 7:
        hotel_price = 700

    elif num_nights == 5:
        hotel_price = 500
            
    elif num_nights == 10:
        hotel_price = 1000
            
    elif num_nights == 3:
        hotel_price = 450
            
    else:
        print("\n" + "We do not have availability!")
        hotel_price = 0
            
    return hotel_price

# declaration of a denined function car_rental with pre-defined values for rental car price 
def car_rental(rental_days):

    if rental_days == 7:
        rental_price = 1400

    elif rental_days == 5:
        rental_price = 1000
            
    elif rental_days == 10:
        rental_price = 2000
            
    elif rental_days == 3:
        rental_price = 950
            
    else:
        print("\n" + "We do not have availability!")
        rental_price = 0
            
    return rental_price

# declaration of a denined function holiday_cost that calculates the above 
def holiday_cost(plane_cost, hotel_cost, car_rental):

    total_cost = plane_price + hotel_price + rental_price
    return total_cost


while True:
    
    try:
        city_flight = input("Please type the destination you wish to fly to (letters only):" + "\n").strip().lower()

        #Check if only letters 
        if not city_flight.isalpha():
            print("Please enter only letters")
            continue 
        
        #call of the method plane_cost
        plane_price = plane_cost(city_flight)

        print("\n" + "Do you need a hotel room?")
        user_hotel = input("Yes or No?:" + "\n").strip().lower()
         
        if user_hotel not in ("yes", "no") :
            print("\n" + "Hmm.. We cannot recognize the information you typed.\n")
            continue
        
        if user_hotel == "no":
            hotel_price = 0

        elif user_hotel == "yes": 

            num_nights = int(input("\n" + "Number of hotel overnights (numbers only):" + "\n"))

            #call for method hotel_cost
            hotel_price = hotel_cost(num_nights)
            print("\n" + "Would you like to rent a car?")

        user_car = input("Yes or No?:" + "\n" ).strip().lower()

        if user_car not in ("yes", "no"):
            print("Hmm.. We cannot recognize the information you typed.\n")
            continue
        
        if user_car == "no":
            rental_price = 0

        if user_car == "yes":

            rental_days = int(input("\n" + "Number of days you rent a vehicle for (numbers only):" + "\n"))

            #call for car_rental function
            rental_price = car_rental(rental_days)

        break_price = holiday_cost(plane_cost, hotel_cost, car_rental)
        
        print(f"The total cost for your holiday will be : £{break_price}")

        another_search = input("\n" + "Would you like to perform another holiday calculation?:" + "\n").strip().lower()
        if another_search =="yes":
            continue
        
        else: 
            print("\n" + "Thank you! Good bye!")
            break

    except ValueError:
        print("\n" + "Hmm.. We cannot recognize the information you typed.\nPlease start over!")

# End of Program
            
