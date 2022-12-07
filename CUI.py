import tkinter
from tkinter import *
import Dialogue_Manager

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    print(msg)
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        intent = Dialogue_Manager.interpreter.parse(msg)["intent"]
        print(Dialogue_Manager.interpreter.parse(msg))
        #Main.send_message(msg, intent["name"])
        res = Dialogue_Manager.respond(msg, intent["name"])

        ChatBox.insert(END, "Bot: " + res + '\n\n')

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial", )

ChatBox.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b", fg='#000000',
                    command=send)

# Create the box to enter message
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()