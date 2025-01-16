import tkinter as tk
from tkinter import messagebox, scrolledtext
import base64
import json

# Function to decode JWT
def decode_jwt():
    """
    This function takes the JWT input from the user,
    splits it into its three parts (header, payload, and signature),
    and decodes the base64-encoded header and payload.
    """
    jwt_token = entry.get().strip()
    
    try:
        # Split the token into its three parts
        header_b64, payload_b64, signature = jwt_token.split('.')
        
        # Decode Base64 and add padding if necessary
        header_json = base64.urlsafe_b64decode(header_b64 + '==').decode('utf-8')
        payload_json = base64.urlsafe_b64decode(payload_b64 + '==').decode('utf-8')
        
        # Format the JSON output
        header_dict = json.loads(header_json)
        payload_dict = json.loads(payload_json)
        
        header_text.delete(1.0, tk.END)
        payload_text.delete(1.0, tk.END)
        signature_text.delete(1.0, tk.END)
        
        header_text.insert(tk.END, json.dumps(header_dict, indent=4))
        payload_text.insert(tk.END, json.dumps(payload_dict, indent=4))
        signature_text.insert(tk.END, signature)
    
    except Exception as e:
        messagebox.showerror("Error", "Invalid JWT Format! Make sure the token is correctly structured.")

# UI Setup
root = tk.Tk()
root.title("JWT Parser")
root.geometry("600x500")
root.resizable(False, False)

# Input field for JWT
tk.Label(root, text="Enter JWT Token:").pack(pady=5)
entry = tk.Entry(root, width=80)
entry.pack(pady=5)

tk.Button(root, text="Decode", command=decode_jwt).pack(pady=10)

# Header Output
tk.Label(root, text="Header:").pack()
header_text = scrolledtext.ScrolledText(root, width=70, height=5)
header_text.pack()

# Payload Output
tk.Label(root, text="Payload:").pack()
payload_text = scrolledtext.ScrolledText(root, width=70, height=5)
payload_text.pack()

# Signature Output
tk.Label(root, text="Signature:").pack()
signature_text = scrolledtext.ScrolledText(root, width=70, height=3)
signature_text.pack()

# Run the application
root.mainloop()
