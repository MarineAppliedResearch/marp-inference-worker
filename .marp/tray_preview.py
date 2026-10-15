# Shows the tray against whatever worker is already running on this machine.
from marp_inference_worker.installation.tray import WorkerTray
WorkerTray(port=8010).run()
