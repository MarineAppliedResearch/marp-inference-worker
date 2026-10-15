# tray.py
# Created: 2026-10-14
# Author: Isaac Travers
#
# The tray icon a volunteer sees while the worker is running, and the menu they
# control it from. It exists because a volunteer who installs the worker had no
# way to tell it was running and no way to get their machine back short of
# uninstalling it (#57).
#
# It owns no state. Everything it shows is read from the running worker's own
# /status, and everything it does is a POST to /status/control or /status/screen
# -- the same endpoints the command line already uses. That is deliberate: a
# tray holding its own copy of the state would disagree with the worker the
# first time anything else changed it.
#
# What belongs in this file: the icon, the menu, and the mapping from the three
# words a volunteer understands onto the actions the worker already has.
# What does not: any decision about jobs, the GPU or the coordinator.

# httpx talks to the worker's own loopback API.
import httpx

# threading runs the status poll beside the icon's own event loop.
import threading

# resources finds the MARP mark inside the installed package, which is the only
# way to reach it once the worker is installed rather than run from a checkout.
from importlib import resources


# How often the tray asks the worker what it is doing.
# Slow enough to be free, fast enough that the icon is not visibly stale after
# somebody uses the menu.
_POLL_SECONDS = 3.0


# How long a call to the worker's own loopback API is given.
# It is on this machine; a timeout here means the worker is gone, not busy.
_TIMEOUT_SECONDS = 4.0


# The three states a volunteer chooses between, and what each one means to the
# worker underneath. Background and Foreground are the existing screen modes;
# Pause is the operator action added for #57.
#
# Give me my GPU now is deliberately not in here. It is an action rather than a
# state -- you press it and land in Paused -- and the menu shows it separately.
# The action names are the ones /status/control accepts, not the ones the
# worker stores. It takes `resume` and turns it into the stored `running`;
# sending `running` straight at it is rejected, which is what made Pause work
# and un-pause do nothing at all. test_every_action_the_tray_sends_is_accepted
# is what stops that coming back.
_STATES = {
    "background": ("Background", "resume", "off"),
    "foreground": ("Foreground", "resume", "window"),
    "paused": ("Paused", "pause", None),
}


# colour_for()
# The icon colour that goes with a state.
# Inputs: the state key.
# Output: an RGB tuple.
# A glance at the tray is meant to answer "is it working", so the states are
# different colours rather than different shades of one (#57 R1, A3).
def colour_for(state: str) -> tuple[int, int, int]:

    if state == "paused":
        return (150, 150, 150)
    if state == "foreground":
        return (0, 170, 200)
    return (0, 120, 60)


# state_from_status()
# Works out which of the three states the worker is actually in.
# Inputs: the mapping /status returned.
# Output: one of the keys in _STATES.
# Reads the worker rather than remembering what was last clicked, so the icon is
# still right after the coordinator or the command line changes something.
def state_from_status(status: dict) -> str:

    if str(status.get("operator_action")) == "pause":
        return "paused"

    # finish and stop are not pause, but from the tray they look like it: the
    # worker is not going to take work. Showing them as Paused is honest enough
    # for an icon, and the tooltip carries the detail.
    if status.get("paused"):
        return "paused"

    if str(status.get("screen_mode", "off")) != "off":
        return "foreground"
    return "background"


# tooltip_for()
# The sentence shown when the volunteer hovers the icon.
# Inputs: the state key and the mapping /status returned.
# Output: the text.
# R1 asks for words, not just a colour: a volunteer should be able to find out
# what their machine is doing without clicking anything.
def tooltip_for(state: str, status: dict) -> str:

    # `active_jobs` is what /status calls it. Reading `jobs` found nothing, so
    # the tooltip said "waiting for work" while a job was on screen -- which is
    # worse than no tooltip, because it is confidently wrong about the one
    # thing a volunteer looks at it to learn.
    running = len(status.get("active_jobs") or [])

    if state == "paused":
        if running:
            return "MARP worker - paused. Your GPU is free; the job is held."
        return "MARP worker - paused. Not taking work."

    if running:
        plural = "job" if running == 1 else "jobs"
        return f"MARP worker - running {running} {plural}."
    return "MARP worker - running. Waiting for work."


# The MARP mark, which already travels with the package for the watch window.
# The tray shows the same logo as everything else rather than a shape of its
# own: a volunteer should recognise whose software is on their machine.
_MARK = resources.files("marp_inference_worker.watch") / "marp-mark.png"


# The disc the mark sits on: navy-900 from the platform's own token scale.
# The mark is bright cyan and green, which disappears against a light taskbar,
# so it is given a dark ground rather than being recoloured. navy-1000 is
# nearly black and loses its edge on a dark taskbar; this one holds on both.
_DISC = (0x03, 0x10, 0x1F)


# _image()
# Builds the tray bitmap: the MARP mark on a dark disc, badged with the state.
# Inputs: an RGB tuple for the badge.
# Output: a PIL image.
#
# The disc is what makes the logo readable on any taskbar. The badge is what
# the disc costs: once the icon is the logo it can no longer say whether the
# worker is paused, and that is the one thing a volunteer should be able to see
# without clicking. So the tray carries the dot and the browser favicon does
# not -- state matters here and nowhere else.
#
# Falls back to a bare dot if the mark cannot be loaded. A plainer icon is
# worth more than no icon, which is what an exception here would mean.
def _image(colour: tuple[int, int, int]):

    from PIL import Image, ImageDraw

    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    try:
        with resources.as_file(_MARK) as path:
            mark = Image.open(path).convert("RGBA")

        draw.ellipse((0, 0, size - 1, size - 1), fill=_DISC + (255,))

        # Trim the transparent padding baked into the source file before
        # scaling. Without this the fraction below is measured against the
        # padding as well as the mark, so the M came out noticeably smaller
        # than the number suggests -- and unreadable at tray size.
        bounds = mark.getbbox()
        if bounds:
            mark = mark.crop(bounds)

        # Four fifths of the disc. More and the M's legs run past the edge.
        mark.thumbnail((int(size * 0.80), int(size * 0.80)), Image.LANCZOS)
        image.paste(mark, ((size - mark.width) // 2, (size - mark.height) // 2), mark)
        badge = (size - 19, size - 19, size - 1, size - 1)
    except Exception:
        badge = (4, 4, size - 4, size - 4)

    # Outlined so the badge stays visible against the disc and against a light
    # or a dark taskbar alike.
    draw.ellipse(badge, fill=colour + (255,), outline=(255, 255, 255, 255), width=2)
    return image


# WorkerTray
# The icon, its menu, and the poll that keeps them true.
#
# Built around pystray, which supplies the platform notify-icon and its event
# loop. Everything platform-specific is behind that; this class knows only about
# states, menu items and HTTP.
class WorkerTray:

    # __init__()
    # Prepares a tray against a worker on the given loopback port.
    # Inputs: the worker's API port.
    # Output: an unstarted tray.
    def __init__(self, port: int) -> None:

        # Where the worker's own API is. Loopback only -- the tray controls this
        # machine's worker and must never be able to reach anybody else's.
        self._base = f"http://127.0.0.1:{port}"

        # Last state read from the worker, so the menu can mark the current one.
        self._state = "background"

        # Set to stop the poll thread.
        self._stopping = threading.Event()

        # Filled in by run(); None until pystray is imported successfully.
        self._icon = None


    # _get_status()
    # Asks the worker what it is doing.
    # Inputs: none.
    # Output: the status mapping, or None when the worker is not answering.
    def _get_status(self) -> dict | None:

        try:
            response = httpx.get(f"{self._base}/status", timeout=_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except Exception:
            # A worker that is starting, restarting or updating is not an error
            # worth showing a volunteer. The icon just stops changing.
            return None


    # _post()
    # Sends one control change to the worker.
    # Inputs: the path under /status, and the body.
    # Output: True when the worker accepted it.
    def _post(self, path: str, body: dict) -> bool:

        try:
            response = httpx.post(f"{self._base}{path}", json=body, timeout=_TIMEOUT_SECONDS)
            response.raise_for_status()
            return True
        except Exception:
            return False


    # choose()
    # Puts the worker into one of the three states.
    # Inputs: the state key.
    # Output: True when the worker accepted every call it needed.
    # Foreground and Background are one action plus a screen mode; the screen
    # call is what retargets a job that is already running (#57 R4).
    def choose(self, state: str) -> bool:

        _, action, screen = _STATES[state]
        ok = self._post("/status/control", {"action": action})

        if ok and screen is not None:
            ok = self._post("/status/screen", {"mode": screen})

        if ok:
            self._state = state
            self._refresh()
        return ok


    # release_gpu()
    # Gives the volunteer their GPU back now, without waiting.
    # Inputs: none.
    # Output: True when the worker accepted it.
    # This is `stop`, which yields the running job: the child unwinds, writes
    # the work it has, exits, and the VRAM goes with it. Pause on its own keeps
    # the model loaded so resuming is free, which is the wrong trade for
    # somebody about to start a game (#57 R10).
    def release_gpu(self) -> bool:

        if not self._post("/status/control", {"action": "stop"}):
            return False
        self._state = "paused"
        self._refresh()
        return True


    # _refresh()
    # Redraws the icon and tooltip from the current state.
    # Inputs: the latest status, when there is one.
    # Output: none.
    def _refresh(self, status: dict | None = None) -> None:

        if self._icon is None:
            return
        self._icon.icon = _image(colour_for(self._state))
        self._icon.title = tooltip_for(self._state, status or {})


    # _poll()
    # Keeps the icon in step with the worker, in its own thread.
    # Inputs: none.
    # Output: none.
    # Polling rather than being told, because the worker has no way to call out
    # to a tray and adding one would be a new surface for one cosmetic job.
    def _poll(self) -> None:

        while not self._stopping.wait(_POLL_SECONDS):
            status = self._get_status()
            if status is None:
                continue
            self._state = state_from_status(status)
            self._refresh(status)


    # build_menu()
    # Builds the tray menu: the three states, then the immediate release.
    # Inputs: none.
    # Output: a pystray.Menu.
    # Separate from run() so it can be built and checked without a desktop --
    # run() needs a display and a message pump, this needs neither.
    def build_menu(self):

        import pystray

        # One builder for the three radio items, so a state cannot be added to
        # the menu without also being in _STATES.
        #
        # Closures, not lambdas with a default argument to capture `key`.
        # pystray inspects an action's signature and refuses anything that does
        # not take exactly (icon, item) -- a `key=key` default counts as a third
        # parameter, so the menu raised ValueError before it was ever built and
        # the launcher's swallow turned that into a worker with no icon and no
        # error. Found on a real install, not by reading.
        def item(key: str):
            label = _STATES[key][0]

            def activate(icon, menu_item) -> None:
                self.choose(key)

            def is_current(menu_item) -> bool:
                return self._state == key

            return pystray.MenuItem(label, activate, checked=is_current, radio=True)

        def release(icon, menu_item) -> None:
            self.release_gpu()

        return pystray.Menu(
            item("background"),
            item("foreground"),
            item("paused"),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Give me my GPU now", release),
        )


    # run()
    # Shows the icon and blocks until the tray is stopped.
    # Inputs: none.
    # Output: none.
    # Raises ImportError when pystray or Pillow is missing, which the caller is
    # expected to swallow: a machine with no tray must still run the worker.
    def run(self) -> None:

        import pystray

        self._icon = pystray.Icon(
            "marp-worker",
            icon=_image(colour_for(self._state)),
            title=tooltip_for(self._state, {}),
            menu=self.build_menu(),
        )

        poller = threading.Thread(target=self._poll, daemon=True, name="marp-tray-poll")
        poller.start()
        try:
            self._icon.run()
        finally:
            self._stopping.set()


    # stop()
    # Takes the icon away.
    # Inputs: none.
    # Output: none.
    def stop(self) -> None:

        self._stopping.set()
        if self._icon is not None:
            self._icon.stop()
