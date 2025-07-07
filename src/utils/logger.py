import time
import logging

from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.progress import track as rtrack
from rich.traceback import install; install()


class TaskLogger:
    def __init__(self, name: str, level: int = logging.INFO):
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.logger = logging.getLogger(name)

    def logdebug(self, message: str):
        self.logger.debug(message, extra={"markup": True})

    def loginfo(self, message: str):
        self.logger.info(message, extra={"markup": True})
        
    def logwarning(self, message: str):
        self.logger.warning(message, extra={"markup": True})
        
    def logerror(self, message: str):
        self.logger.error(message, extra={"markup": True})        

    def logcritical(self, message: str):
        self.logger.critical(message, extra={"markup": True})
            
    def logprocess(self, task_description: str):
        """
        Context manager to log started and elapsed time of a task.
        """
        def decorator(fn):
            def wrapper(*args, **kwargs):
                with Progress(
                    SpinnerColumn(),
                    TimeElapsedColumn(),
                    TextColumn(f"[progress.description]{task_description}"),
                    transient=True,
                ) as progress:
                    progress.add_task(description=task_description, total=None)
                    self.loginfo(f"[bold]{task_description}[/bold] started.")
                    tic = time.perf_counter()
                    result = fn(*args, **kwargs)
                    toc = time.perf_counter()
                    self.loginfo(f"[bold]{task_description}[/bold] finished in {(toc - tic):.4f} secs.")
                return result
            return wrapper
        return decorator
    
    def track(self, *args, **kwargs):
        return rtrack(*args, **kwargs)