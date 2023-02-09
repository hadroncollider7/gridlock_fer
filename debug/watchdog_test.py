import sys
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
import os

if __name__ == "__main__":
    """
    A simple program to moniotr the current directory recursively for
    file system changes and log them to the console.
    """
    os.system("cls")        # Clear the console
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s - %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S')
    # Define the path to be monitored recuresivley
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    # Implement a subclass of watchdog.events.FileSystemEventHandler (LoggingEventHandler already implements the subclass)
    event_handler = LoggingEventHandler()
    # Create an instance of hte watchdog.observers.Observer thread class
    observer = Observer()
    # Schedule monitoring a few paths with the observer instance attaching the event handler
    observer.schedule(event_handler, path, recursive=True)
    print("Begin monitoring of: ", path)
    # Start the observer thread and wait for it to generate events withot blocking the main thread
    observer.start()
    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        observer.stop()
        observer.join()