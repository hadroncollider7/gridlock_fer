import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
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
        
        
        
class Handler(PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        PatternMatchingEventHandler.__init__(self, patterns=['*.jpg'],
                                             ignore_directories=True,
                                             case_sensitive=False)
    
    def on_created(self, event):
        # Will execute for creation events
        print("Watchdog received created event: {1:s}".format(event, event.src_path))
        
    def on_deleted(self, event):
        # Will execute for modified events
        print("Watchdog received delete event: {1:s}".format(event, event.src_path))

    


if __name__ == "__main__":
    watch = OnMyWatch()
    watch.run()