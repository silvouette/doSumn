#If csv dataset already available, don't run this code. Labelling is done manually.
import io
import os
import csv

path = os.path.dirname(os.getcwd())
csv_path = os.path.join(path, "labels.csv")
txt_path = os.path.join(path, "data_txt/")

def addToCsv():
    print("Appending to csv...")
    with open(csv_path, "w+") as of_out:
        csvwriter = csv.writer(of_out, delimiter=",", quotechar='"')
        out = []
        for txt_file in os.listdir(txt_path):
            print(txt_file+"\n")
            with open(txt_path + txt_file, "r") as of_txt:
                for line in of_txt:
                    out.append((txt_file, line)) 

        for item in out:
            csvwriter.writerow(item)
                    
if __name__ == "__main__":
    addToCsv()