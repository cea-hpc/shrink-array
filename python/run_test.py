#!/usr/bin/env python3

import subprocess

def lire_scalaire_depuis_fichier(file_name):
    try:
        # Open the file in read mode
        with open(file_name, 'r') as fichier:
            # Read the scalar from the file
            scalaire = float(fichier.readline())
            return scalaire
    except FileNotFoundError:
        print("The specified file does not exist.")
        return None
    except ValueError:
        print("The file does not contain a valid scalar.")
        return None

def run_program():

    env_variable = {"OMP_NUM_THREADS": "8"}
    
    result = subprocess.run(["????/cmake-build-optim/test/test1"], capture_output=True, text=True, env=env_variable)
    
    # Récupérer la sortie du programme
    output = result.stdout
    
    print(output)

    file_name = "memory_mo.txt"
    scalaire = lire_scalaire_depuis_fichier(file_name)

    return scalaire

def average_output(runs):
    total_output = 0
    
    for _ in range(runs):
        total_output += run_program()
    
    average = total_output / runs
    return average

def main():
    num_runs = int(input("Enter the number of times you want to run the program : "))
    avg = average_output(num_runs)

    formatted_nombre = "{:.2f}".format(avg)

    print("The average output of the program on", num_runs, "runs is :", formatted_nombre, " Mo")

if __name__ == "__main__":
    main()
