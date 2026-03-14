# ==============================================================================
# TuNet PyInstaller Entry Point
#
# Single entry point that dispatches to different modules based on arguments.
# When frozen (exe), sys.executable points to this exe. The UI launches
# sub-scripts via [sys.executable, 'train.py', ...], so this dispatcher
# intercepts .py arguments and runs the corresponding bundled script.
# ==============================================================================

import sys
import os


def get_base_dir():
    """Get the base directory where scripts and data are located."""
    if getattr(sys, 'frozen', False):
        # PyInstaller --onedir: _MEIPASS is the _internal directory
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def main():
    # When called with a .py file as first argument, dispatch to that script.
    # This handles subprocess calls like: [sys.executable, 'train.py', '--config', ...]
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        script_name = os.path.basename(first_arg)

        if script_name.endswith('.py'):
            # Find the bundled script
            base_dir = get_base_dir()

            # Check in base dir first, then in subdirectories (exporters/, utils/)
            script_path = os.path.join(base_dir, script_name)
            if not os.path.isfile(script_path):
                # Try preserving the relative path (e.g. exporters/flame_exporter.py)
                for prefix in [base_dir, os.path.dirname(sys.executable)]:
                    # Try the directory from the argument path
                    candidate = os.path.join(prefix, os.path.basename(os.path.dirname(first_arg)), script_name)
                    if os.path.isfile(candidate):
                        script_path = candidate
                        break
                    # Also search known subdirectories
                    for subdir in ['exporters', 'utils']:
                        candidate = os.path.join(prefix, subdir, script_name)
                        if os.path.isfile(candidate):
                            script_path = candidate
                            break
                else:
                    # Try the original path as-is (might be an absolute path into _MEIPASS)
                    if os.path.isfile(first_arg):
                        script_path = first_arg

            if os.path.isfile(script_path):
                # Shift argv so the script sees itself as argv[0]
                sys.argv = sys.argv[1:]
                sys.argv[0] = script_path

                # Execute the script with __name__ = '__main__' so
                # if __name__ == '__main__': blocks run correctly
                with open(script_path, 'r', encoding='utf-8') as f:
                    code = compile(f.read(), script_path, 'exec')
                exec(code, {
                    '__name__': '__main__',
                    '__file__': script_path,
                    '__builtins__': __builtins__,
                })
                return

    # Default: launch the UI
    # Check for updates on startup (frozen builds only)
    if getattr(sys, 'frozen', False):
        try:
            from tunet_updater import install_or_update
            install_or_update(headless=False)
        except Exception as e:
            print(f"Update check skipped: {e}")

    from ui_app import MainWindow
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    # PyInstaller multiprocessing support
    import multiprocessing
    multiprocessing.freeze_support()
    main()
