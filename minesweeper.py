""" 
Bootcamp: Skills Bootcamp in Data Science (Fundamentals)
Institute: HyperionDev

Student: Vasil Georgiev
Student ID: VG22110006759

DS T15 Introduction to Python - Data Structures - 2D Lists
Compulsory task 

minesweeper.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Case:

The following python code presents a minesweeper field under the form of 2d lists. 

It then uses the values assigned to the parameters in the minesweeper grid and returns a calculation, which indicates the number
of mines immediately adjacent to the spot. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Begin of program

def mine_sweeper(grid):

    # By defining rows and cols, we set the base on how I would like python to use to create the grid 
    rows = len(grid)

    # As the grid is rectangular, here I want to check what is the lenght of the first column
    cols = len(grid[0])

    # Declaration of list directions that contains all criteria needed so that calculations can be performed 
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    # Declaration of a 2D list named result that will contain the calculated grid from any input, when this method is called  
    result = [[0] * cols for _ in range(rows)]

    # So that claculations can be performed and values assigned, I need to declare two for loops, in which the inner loop iterates over the outer loop and depending on condition, it assigns a value
    # As recommended, I am using the enumerate function on the grid 
    for i, row in enumerate(grid):

        for j, spot in enumerate(row):

            if spot == '#':
                result[i][j] = '#'
                continue

            count = 0

            #Declaration of a loop that uses vairables dx, dy to calculate and store changes as the two loops above iterate over each other
            for dx, dy in directions:

                ni, nj = i + dx, j + dy
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == '#':
                    count += 1
            result[i][j] = str(count)

    return result

# The example grid from the task description 
input_grid = [["-", "-", "-", "#", "#"],
              ["-", "#", "-", "-", "-"],
              ["-", "-", "#", "-", "-"],
              ["-", "#", "#", "-", "-"],
              ["-", "-", "-", "-", "-"]]

for row in input_grid:
    print(row)
  
print()

#Call of the method I created above
output_grid = mine_sweeper(input_grid)

for row in output_grid:
    print(row)