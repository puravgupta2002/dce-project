import os

def collect_data():
    os.system("python collect_data.py")

def train_model():
    os.system("python train_model.py")

def run_detection():
    os.system("python detect_signs.py")

def menu():
    while True:
        print("\nSign Language Detection - Main Menu")
        print("1. Collect Gesture Data")
        print("2. Train Model")
        print("3. Run Real-Time Detection")
        print("4. Exit")

        choice = input("Enter your choice [1-4]: ")

        if choice == '1':
            collect_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_detection()
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid input. Please select a valid option.")

if __name__ == "__main__":
    menu()
