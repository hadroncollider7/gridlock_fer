import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml

with open("config.yml", "r") as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

class OnMyWatch:
    # Set the directory on watch
    watchDirectory = config["sourcePaths"]["imagePath"]
    print("Observer thread started on: ", watchDirectory)
    
    def __init__(self):
        # Create observer object to watch directory and dispatch calls to event handlers
        self.observer = Observer()
        
    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDirectory, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
        finally:
            self.observer.stop()
            print("Observer thread stopped")
            self.observer.join()
        
        
        
class Handler(FileSystemEventHandler):
    
    @staticmethod
    def on_any_event(event):
    # Will execute for any event
        if event.is_directory:
            return None
        
        elif event.event_type == 'created':
            # Event is created, you can process it now
            print("Watchdog received created event - % s." %event.src_path)
        elif event.event_type == "deleted":
            print("Watchdog received deleted event - %s." %event.src_path)


if __name__ == "__main__":
    watch = OnMyWatch()
    watch.run()