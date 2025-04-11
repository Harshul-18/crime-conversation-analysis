import time
import shutil

# Function to print a message and then overwrite it
def print_and_overwrite(message, sleep_time=0.2):
    # Clear the line to ensure no leftover characters
    print('\x1b[2K\x1b[0G' + ' ' * terminal_width, end='\x1b[2K\x1b[0G', flush=True)
    # Print the new message
    print(message, end='', flush=True)
    time.sleep(sleep_time)  # Wait a bit for the user to see the message

# Get the width of the terminal window
terminal_width, _ = shutil.get_terminal_size()

# Messages to print and overwrite
message = "apology is for mistakes not for crimes rightly said sir he should be punished hand over this drug dealer to police just get him arrested please sir i am sorry i will not do this again just shut up ".split()
messages = []
r = ""
for i in message:
    r += i + ' '
    messages.append(r)

print(messages)

# # Loop through the messages and print them
# for msg in messages:
#     # Truncate the message if it's longer than the terminal width
#     truncated_msg = msg if len(msg) <= terminal_width else msg[:terminal_width]
#     print_and_overwrite(truncated_msg)

# # Finally, print the last message without overwriting it
# # Ensure the message fits in the terminal window
# final_msg = messages[-1] if len(messages[-1]) <= terminal_width else messages[-1][:terminal_width]
# print('\x1b[2K\x1b[0G' + final_msg)
