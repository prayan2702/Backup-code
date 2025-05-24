import streamlit as st
import os
import shutil
from datetime import datetime
import json
import time
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Common Tools",
    page_icon="🛠️",
    layout="wide"
)

# Custom CSS
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

# Initialize clipboard state
if 'text_entries' not in st.session_state:
    try:
        with open("clipboard_entries.json", "r") as f:
            st.session_state.text_entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.text_entries = []

if 'clear_text' not in st.session_state:
    st.session_state.clear_text = False
if 'save_clicked' not in st.session_state:
    st.session_state.save_clicked = False
if 'clear_clicked' not in st.session_state:
    st.session_state.clear_clicked = False

# Upload folder
UPLOAD_FOLDER = "shared_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_entries():
    try:
        with open("clipboard_entries.json", "w") as f:
            json.dump(st.session_state.text_entries, f, indent=4)
    except Exception as e:
        st.error(f"Error saving clipboard entries: {e}")

st.title("🛠️ Common Tools")
st.markdown("Combine clipboard sharing and file sharing in one app")

tab1, tab2 = st.tabs(["📋 Shared Clipboard", "📂 File Sharing"])

with tab1:
    st.markdown("Type text on one device and copy it from any other device")
    # 🔄 Add a Refresh button to reload clipboard data
    col_refresh, col_empty = st.columns([1, 4])
    if col_refresh.button("🔄 Refresh Clipboard"):
        try:
            with open("clipboard_entries.json", "r") as f:
                st.session_state.text_entries = json.load(f)
            st.success("Clipboard refreshed!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error refreshing clipboard: {e}")

    def handle_form_submission():
        if st.session_state.save_clicked:
            user_text = st.session_state.user_input_text
            if user_text and user_text.strip():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.text_entries.insert(0, {
                    "text": user_text.strip(),
                    "time": timestamp
                })
                st.session_state.text_entries = st.session_state.text_entries[:20]
                save_entries()
                st.session_state.clear_text = True
        elif st.session_state.clear_clicked:
            st.session_state.clear_text = True

    with st.form("input_form", clear_on_submit=False):
        user_input = st.text_area(
            "Type your text here:",
            height=150,
            key="user_input_text",
            value="" if st.session_state.clear_text else st.session_state.get("last_input", "")
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            save_pressed = st.form_submit_button("Save to Shared Clipboard")
            if save_pressed:
                st.session_state.save_clicked = True
                st.session_state.clear_clicked = False
                st.session_state.last_input = user_input
                handle_form_submission()

        with col2:
            clear_pressed = st.form_submit_button("Clear Text")
            if clear_pressed:
                st.session_state.clear_clicked = True
                st.session_state.save_clicked = False
                st.session_state.last_input = ""
                handle_form_submission()

        if st.session_state.clear_text:
            st.session_state.clear_text = False
            st.session_state.save_clicked = False
            st.session_state.clear_clicked = False
            st.rerun()

    st.subheader("Shared Clipboard Contents")

    if not st.session_state.text_entries:
        st.info("Clipboard is empty. Add some text above.")
    else:
        latest_entry = st.session_state.text_entries[0]
        st.markdown("**Latest Entry:**")
        st.markdown(f'<div class="timestamp">Last updated: {latest_entry["time"]}</div>', unsafe_allow_html=True)
        st.code(latest_entry["text"], language="text")

        # ✅ Latest entry copy button with "Copied!" feedback
        components.html(f"""
            <input type="text" value="{latest_entry['text']}" id="copyText_latest" style="position:absolute; left:-1000px;">
            <button id="copyBtn_latest" onclick="
                navigator.clipboard.writeText(document.getElementById('copyText_latest').value);
                var btn = document.getElementById('copyBtn_latest');
                btn.innerHTML = '✅ Copied!';
                setTimeout(function(){{btn.innerHTML='📋 Copy Text';}}, 2000);
            ">📋 Copy Text</button>
        """, height=50)

        with st.expander("View History (Last 20 entries)"):
            for i, entry in enumerate(st.session_state.text_entries[1:], 1):
                st.markdown(f"**Entry {i}** ({entry['time']})")
                st.code(entry["text"], language="text")

                # ✅ History entry copy button with "Copied!" feedback
                components.html(f"""
                    <input type="text" value="{entry['text']}" id="copyText{i}" style="position:absolute; left:-1000px;">
                    <button id="copyBtn{i}" onclick="
                        navigator.clipboard.writeText(document.getElementById('copyText{i}').value);
                        var btn = document.getElementById('copyBtn{i}');
                        btn.innerHTML = '✅ Copied!';
                        setTimeout(function(){{btn.innerHTML='📋 Copy Text';}}, 2000);
                    ">📋 Copy Text</button>
                """, height=50)

    st.markdown("---")
    if st.button("🚨 Clear ALL Clipboard Entries", key="clear_all"):
        st.session_state.text_entries = []
        save_entries()
        st.success("All clipboard entries cleared!")
        time.sleep(1)
        st.rerun()

with tab2:
    st.markdown("Upload and download files between devices")

    # ✅ Allow any file, up to 2GB
    uploaded_file = st.file_uploader("Upload File (Any Type, Max 2GB):", type=None)

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        if not os.path.exists(file_path):  # Avoid duplicate writes on rerun
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ File `{uploaded_file.name}` successfully uploaded!")
        else:
            st.info(f"📁 File `{uploaded_file.name}` already uploaded.")
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

        if st.button("🚨 Delete ALL Files", key="delete_all_files"):
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            st.warning("All files have been deleted!")
            time.sleep(1)
            st.rerun()
    else:
        st.info("No files uploaded yet. Upload a file to see it here.")
