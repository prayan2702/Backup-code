import streamlit as st
import os
import shutil
from datetime import datetime
import json
import time

# Page configuration
st.set_page_config(
    page_title="Common Tools",
    page_icon="🛠️",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .stTextArea [data-baseweb=base-input] {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .stButton button {
        border-radius: 10px;
        font-weight: bold;
    }
    .save-btn button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .clear-btn button {
        background-color: #f44336 !important;
        color: white !important;
    }
    .code-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    .file-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for clipboard
if 'text_entries' not in st.session_state:
    st.session_state.text_entries = []
if 'clear_text' not in st.session_state:
    st.session_state.clear_text = False
if 'save_clicked' not in st.session_state:
    st.session_state.save_clicked = False
if 'clear_clicked' not in st.session_state:
    st.session_state.clear_clicked = False

# Shared Folder for Uploaded Files
UPLOAD_FOLDER = "shared_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to save clipboard entries
def save_entries():
    with open("clipboard_entries.json", "w") as f:
        json.dump(st.session_state.text_entries, f)

# Function to load clipboard entries
def load_entries():
    try:
        with open("clipboard_entries.json", "r") as f:
            st.session_state.text_entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.text_entries = []

# Load entries when the app starts
load_entries()

# Main app layout
st.title("🛠️ Common Tools")
st.markdown("Combine clipboard sharing and file sharing in one app")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["📋 Shared Clipboard", "📂 File Sharing"])

with tab1:
    # Clipboard functionality
    st.markdown("Type text on one device and copy it from any other device")
    
    # Text input section
    def handle_form_submission():
        if st.session_state.save_clicked:
            user_text = st.session_state.user_input_text
            if user_text:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.text_entries.insert(0, {
                    "text": user_text,
                    "time": timestamp
                })
                # Keep only the last 20 entries
                st.session_state.text_entries = st.session_state.text_entries[:20]
                save_entries()
                st.session_state.clear_text = True
        elif st.session_state.clear_clicked:
            st.session_state.clear_text = True

    # Create form
    with st.form("input_form"):
        # Text area
        user_input = st.text_area(
            "Type your text here:", 
            height=150, 
            key="user_input_text",
            value="" if st.session_state.clear_text else st.session_state.get("last_input", "")
        )
        
        col1, col2 = st.columns([1,1])
        with col1:
            # Save button
            save_pressed = st.form_submit_button("Save to Shared Clipboard")
            if save_pressed:
                st.session_state.save_clicked = True
                st.session_state.clear_clicked = False
                st.session_state.last_input = user_input
                handle_form_submission()
                
        with col2:
            # Clear button
            clear_pressed = st.form_submit_button("Clear Text")
            if clear_pressed:
                st.session_state.clear_clicked = True
                st.session_state.save_clicked = False
                st.session_state.last_input = ""
                handle_form_submission()
        
        # Handle the clear flag
        if st.session_state.clear_text:
            st.session_state.clear_text = False
            st.session_state.save_clicked = False
            st.session_state.clear_clicked = False
            st.rerun()

    # Display section
    st.subheader("Shared Clipboard Contents")

    if not st.session_state.text_entries:
        st.info("Clipboard is empty. Add some text above.")
    else:
        latest_entry = st.session_state.text_entries[0]
        
        st.markdown("**Latest Entry:**")
        st.markdown(f'<div class="timestamp">Last updated: {latest_entry["time"]}</div>', unsafe_allow_html=True)
        st.code(latest_entry["text"], language="text")
        
        if st.button("Copy Latest Text"):
            st.code(latest_entry["text"], language="text")
            st.success("Text ready to copy from the box above!")
        
        # Show history (optional)
        with st.expander("View History (Last 20 entries)"):
            for i, entry in enumerate(st.session_state.text_entries[1:], 1):
                st.markdown(f"**Entry {i}** ({entry['time']})")
                st.code(entry["text"], language="text")

    # Clear all entries button
    st.markdown("---")
    if st.button("🚨 Clear ALL Clipboard Entries", key="clear_all"):
        st.session_state.text_entries = []
        save_entries()
        st.success("All clipboard entries cleared!")
        time.sleep(1)
        st.rerun()

with tab2:
    # File sharing functionality
    st.markdown("Upload and download files between devices")
    
    # Upload Section
    uploaded_file = st.file_uploader("Upload File:", type=["png", "jpg", "pdf", "txt", "csv", "xlsx"])

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

        # Save file in shared folder
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ File `{uploaded_file.name}` successfully uploaded!")
        st.rerun()

    # Show List of Available Files for Download
    st.subheader("Available Files for Download")

    files = os.listdir(UPLOAD_FOLDER)
    if files:
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file)
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            col1, col2 = st.columns([4, 1])
            with col1:
                st.download_button(
                    label=f"⬇️ Download {file}",
                    data=file_bytes,
                    file_name=file,
                    mime="application/octet-stream",
                    key=f"download_{file}"
                )
            with col2:
                if st.button(f"🗑️ Delete", key=f"delete_{file}"):
                    os.remove(file_path)
                    st.success(f"File {file} deleted!")
                    time.sleep(1)
                    st.rerun()

        # Add Delete All Button
        if st.button("🚨 Delete ALL Files", key="delete_all_files"):
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            st.warning("All files have been deleted!")
            time.sleep(1)
            st.rerun()
    else:
        st.info("No files uploaded yet. Upload a file to see it here.")

# Auto-refresh every 15 seconds
st.markdown("""
    <script>
    setTimeout(function(){
        window.location.reload(1);
    }, 15000);
    </script>
    """, unsafe_allow_html=True)
