import webview
from backend import Api, app as flask_app
from webview.menu import Menu, MenuAction

"""
This is the main entry point for the hybrid desktop application.
"""

def get_window():
    """Helper to safely get the main window instance."""
    return webview.windows[0] if webview.windows else None

# FIX: Use lambda functions for menu actions to ensure the window context is correctly captured.
# This makes the menu calls more robust.
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


if __name__ == '__main__':
    api_instance = Api()

    menu_items = [
        Menu(
            'File',
            [
                MenuAction('Save Session', save_session),
                MenuAction('Load Session', load_session),
            ]
        ),
        Menu(
            'View',
            [
                MenuAction('Toggle Log', toggle_log),
            ]
        )
    ]

    window = webview.create_window(
        'Clip It!',
        flask_app,
        js_api=api_instance,
        width=1200,
        height=850,
        min_size=(1000, 700),
        menu=menu_items
    )
    
    api_instance.set_window(window)
    webview.start(debug=False)

