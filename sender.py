from sys import exit

from send.display import display_init, display_msg, display_end
from send.frames import msg_to_frames, msg_to_enc, enc_msg_to_frames
from settings import N_BITS, N_ROWS, N_COLS, TIMEIN_SENDER, TIMEOUT_SENDER


def main():
    display_init()
    while True:
        print("Options:", "1. Show grid", "2. Display from message",
              "3. Display from an edited message", "4. Quit", sep="\n")
        choice = input("Enter choice: ")
        if choice == "1":
            frames = msg_to_frames("", N_ROWS, N_COLS, N_BITS)
            display_msg(frames, TIMEIN_SENDER, TIMEOUT_SENDER, True)
        elif choice == "2":
            message = input("Message: ")
            frames = msg_to_frames(message, N_ROWS, N_COLS, N_BITS)
            display_msg(frames, TIMEIN_SENDER, TIMEOUT_SENDER)
        elif choice == "3":
            message = input("Message: ")
            enc = msg_to_enc(message, N_ROWS, N_COLS, N_BITS)
            print(f"Encoded message: {enc}")
            print(f"Length of the encoded message = {len(enc)}")
            flip_bit = int(input(f"Index of the bit to flip: "))
            enc = enc[:(flip_bit - 1)] + str(1 - int(enc[flip_bit - 1])) + enc[
                flip_bit:]
            frames = enc_msg_to_frames(enc, N_ROWS, N_COLS)
            display_msg(frames, TIMEIN_SENDER, TIMEOUT_SENDER)
        elif choice == "4":
            break
        else:
            print("Invalid choice. Try again.")
    display_end()
    exit()


if __name__ == "__main__":
    main()
