# Link con distribución de archivos de python.
# https://www.jetbrains.com/help/pycharm/configuring-project-structure.html

import math

# ** Create a function that grabs the email website domain from a string in the form: **
# user @ domain.com
# ** So for example, passing "user@domain.com" would return: domain.com **
def returnDomain( email ):
    return email[email.index('@'):len(email)]

# ** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about
# edge cases like a punctuation being attached to the word dog, but do account for capitalization. **

def checkWords( text="", lookedWord=""):
    return True if (text.lower().find(lookedWord.lower()) > 0) else False

# ** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **
def countWords( text="", lookedWord=""):
    return text.lower().count(lookedWord.lower())

def excercisesMain():
# ** What is 7 to the power of 4?**
    print(math.pow(7, 4))

    # ** Split this string:**
    s = "Hi there Sam!"
    print(s.split())

    # ** Given the variables:**

    planet = "Earth"
    diameter = 12742

    # ** Use .format() to print the following string: **
    # The diameter of Earth is 12742 kilometers.

    print("The diameter of {0} is {1} kilometers.".format(planet, diameter))

    print("The diameter of {planeta} is {diametro} kilometers. (2)".format(planeta=planet, diametro=diameter))

    # ** Given this nested list, use indexing to grab the word "hello" **
    lst = [1, 2, [3, 4], [5, [100, 200, ['hello']], 23, 11], 1, 7]
    print('Impresión contenido lista')

    print(lst[3][1][2])

    # ** Given this nest dictionary grab the word "hello". Be prepared, this will be annoying/tricky **
    print('Impresión contenido diccionario')
    dictionary = {'k1': [1, 2, 3, {'tricky': ['oh', 'man', 'inception', {'target': [1, 2, 3, 'hello']}]}]}
    print(dictionary['k1'][3]['tricky'][3]['target'][3])

    print (returnDomain('user@domain.com'))

    print(checkWords("this is a text with the word dog", "dog"))
    print(checkWords("this is a text with the word cat", "dog"))


    print(countWords("this is a text with the word dog,dog dog,dogdogdog", "dog"))
    print(countWords("this is a text with the word cat", "dog"))

### Final Problem
"""**You are driving a little too fast, and a police officer stops you. Write a function """
"""  to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". """
"""  If your speed is 60 or less, the result is "No Ticket". If speed is between 61 """
"""  and 80 inclusive, the result is "Small Ticket".
     If speed is 81 or more, the result is "Big Ticket".
     Unless it is your birthday (encoded as a boolean value in the parameters of the function)
      -- on your birthday, your speed can be 5 higher in all """
"""  cases. ** """

def speedTest( speed=0, birthday=False):
    if (not birthday):
        top = 0
    else:
        top = 5

    if ( speed <= 60 + top ):
        return "No ticket"
    elif( speed >= 61 + top and speed <= 80 + top):
        return "Small ticket"
    elif( speed > 80+top ):
        return "Big ticket"

