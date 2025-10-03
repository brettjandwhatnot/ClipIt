import webview
from webview.menu import Menu, MenuAction, MenuSeparator
from backend import Api, app as flask_app
import signal
import sys

# Global variable for the window, so menu functions can access it
window = None

# --- Menu Functions ---
# These functions handle the actions for the top menu bar.

def get_window():
    """Helper to safely get the main window instance."""
    return webview.windows[0] if webview.windows else None

def about():
    if window:
        window.evaluate_js('alert("Clip It! - Version 1.7.7")')

def save_session():
    """Triggers the save session flow in JavaScript."""
    window = get_window()
    if window:
        window.evaluate_js('window.initiateSave()')

def load_session():
    """Triggers the load session flow in JavaScript."""
    window = get_window()
    if window:
        window.evaluate_js('window.initiateLoad()')

def toggle_log():
    """Calls the JavaScript function to toggle the log's visibility."""
    window = get_window()
    if window:
        window.evaluate_js('window.toggleLogVisibility()')

def open_prompt_settings():
    """Calls the JavaScript function to open the AI prompt settings modal."""
    window = get_window()
    if window:
        window.evaluate_js('window.openPromptSettings()')

def quit_app():
    if window:
        window.destroy()

# --- Main Application Setup ---

if __name__ == '__main__':
    # Define the signal handler for graceful shutdown (e.g., Ctrl+C)
    def signal_handler(sig, frame):
        print('Shutting down gracefully...')
        quit_app()
        sys.exit(0)

    # Set up signal handlers in the main thread
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create an instance of your backend API
    api_instance = Api()

    # Define the menu structure
    menu_items = [
        Menu(
            'Clip It!',
            [
                MenuAction('About', about),
                MenuSeparator(),
                MenuAction('Quit', quit_app),
            ]
        ),
        Menu(
            'File',
            [
                MenuAction('Save Session', save_session),
                MenuAction('Load Session', load_session),
                MenuSeparator(),
                MenuAction('AI Prompt Settings', open_prompt_settings),
                MenuSeparator(),
                MenuAction('Close', quit_app),
            ]
        ),
        Menu(
            'View',
            [
                MenuAction('Toggle Log', toggle_log),
            ]
            
        )
        
    
    ]

    # Create the application window
    window = webview.create_window(
        'Clip It!',
        flask_app,
        width=1280,
        height=720,
        resizable=True,
        min_size=(1024, 720),
        background_color='#2c2c2c',
        js_api=api_instance, # Connects your backend.py to the frontend
        text_select=False,
    )

    # Pass the created window object to the API instance
    # so backend functions can control the window
    api_instance.window = window
    
    # Start the application
    webview.start(debug=False, menu=menu_items)