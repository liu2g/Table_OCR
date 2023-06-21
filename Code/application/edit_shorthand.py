import tkinter as tk
from tkscrolledframe import ScrolledFrame
import file_manager as fm

global root, title_frame, button_frame, scroll_frame, edit_frame
global rows, shorthand_list, shorthand

def delete_entry(entry1, entry2, button1):
    global root, title_frame, button_frame, scroll_frame, edit_frame
    global rows, shorthand_list
    index = 0
    for e1, e2, b1 in shorthand_list:
        if ((e1 == entry1) and (e2 == entry2) and (b1 == button1)):
            e1.destroy()
            e2.destroy()
            b1.destroy()
            shorthand_list.pop(index)
            rows -= 1

        index += 1

def create_entry():
    global root, title_frame, button_frame, scroll_frame, edit_frame
    global rows, shorthand_list
    row = rows
    entry1=tk.Entry(edit_frame, font=("Arial", 10), width=24)
    entry2=tk.Entry(edit_frame, font=("Arial", 10), width=24)
    button1=tk.Button(edit_frame, text="  X  ", font=("Arial", 10), command=lambda :delete_entry(entry1, entry2, button1))
    shorthand_list.append([entry1, entry2, button1])
    shorthand_list[row][0].grid(row=rows, column=0, sticky='nsew')
    shorthand_list[row][1].grid(row=rows, column=1, sticky='nsew')
    shorthand_list[row][2].grid(row=rows, column=2, sticky='nsew')
    edit_frame.rowconfigure(rows, weight=1)
    rows += 1

def initialize():
    global root, title_frame, button_frame, scroll_frame, edit_frame
    global rows, shorthand_list, shorthand
    shorthand = {}
    shorthand_list = []
    rows = 0
    shorthand = fm.load_shorthand()
    for short, long in shorthand.items():
        create_entry()
        row = rows - 1
        shorthand_list[row][0].insert(0,short)
        shorthand_list[row][1].insert(0,long)

def close():
    global root, title_frame, button_frame, scroll_frame, edit_frame
    global shorthand, shorthand_list
    shorthand.clear()
    for e1, e2, b1 in shorthand_list:
        if((e1.get() != "") and (e2.get() != "")):
            shorthand[e1.get()] = e2.get()
    fm.save_shorthand(shorthand)
    root.grab_release()
    root.destroy()


def run():
    global root, title_frame, button_frame, scroll_frame, edit_frame
    root = tk.Toplevel()
    root.grab_set()

    root.title("Edit Shorthand")
    root.rowconfigure(3, weight=1)
    root.columnconfigure(2, weight=1)

    title_frame = tk.Frame(root)
    title_frame.rowconfigure(0, weight=1)
    title_frame.columnconfigure(1, weight=1)
    title_frame.grid(row=0, column=0, sticky='nsew')

    button_frame = tk.Frame(root)
    button_frame.rowconfigure(0, weight=1)
    button_frame.columnconfigure(0, weight=1)
    button_frame.grid(row=2, column=0, sticky='nsew')

    scroll_frame = ScrolledFrame(root)
    scroll_frame.grid(row=1, column=0, sticky='nsew')
    scroll_frame.bind_arrow_keys(root)
    scroll_frame.bind_scroll_wheel(root)
    edit_frame = scroll_frame.display_widget(tk.Frame)
    edit_frame.rowconfigure(1, weight=1)
    edit_frame.columnconfigure(3, weight=1)

    tk.Label(title_frame, text="     Shorthand     ", font=("Arial", 15)).grid(row=0, column=0, sticky='nsew')
    tk.Label(title_frame, text="    Expands To     ", font=("Arial", 15)).grid(row=0, column=1, sticky='nsew')
    tk.Label(title_frame, text="     ", font=("Arial", 15)).grid(row=0, column=2, sticky='nsew')
    tk.Button(button_frame, text="Add Shorthand Entry", font=("Arial", 15), command=create_entry).grid(row=0, column=0, pady=10, padx=10) 

    initialize()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()

