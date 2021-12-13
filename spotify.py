import os
import sys
import json


import asyncio
from bleak import BleakScanner
from kivy.app import async_runTouchApp

from PIL import Image
import numpy as np
import spotipy
import webbrowser
import spotipy.util as util
from json.decoder import JSONDecodeError
import random
import time
from urllib.request import urlopen
import io
# from colorthief import ColorThief

from kivy.app import App
from kivy.metrics import dp
from kivy.graphics.vertex_instructions import Line
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.vertex_instructions import Ellipse
from kivy.graphics.context_instructions import Color
from kivy.properties import StringProperty, BooleanProperty, ListProperty, NumericProperty
from kivy.uix.button import Button
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.image import Image as Img
from kivy.properties import Clock
from kivy.uix.widget import Widget
from spotipy.oauth2 import SpotifyOAuth
from kivy.core.window import Window
from kivy.app import async_runTouchApp
# Window.size = (330, 600)

# username = 'insert username here
# 
# sys.argv[1]
# scope = 'playlist-read-private user-read-playback-state user-modify-playback-state'
# SPOTIPY_CLIENT_ID = 'insert client id here'
# SPOTIPY_CLIENT_SECRET = "insert secret key here
# SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback/'

# Erase cache and prompt for user permission
# try:
#    auth_manager = SpotifyOAuth(username=username, scope=scope, client_id=SPOTIPY_CLIENT_ID, client_secret
#    =SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI)
#    spotifyObject = spotipy.Spotify(auth_manager=auth_manager)
#    #token = "insert token here"
#    token = util.prompt_for_user_token(username, scope, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI)
#    print("Token is ", token)
# except:
#    #os.remove(f".cache-{username}")
#    print("can't print token")
#    token = util.prompt_for_user_token(username, scope, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI)


class MainWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backgroundImage1 = Img(source='images/backgroundgradient1.jpg', pos=(0, Window.height*5/8), size=(Window.width, Window.height*1/4), allow_stretch=True, keep_ratio=False)
        self.backgroundImage2 = Img(source='images/backgroundgradient2.jpg', pos=(0, Window.height*3/8), size=(Window.width, Window.height*1/4), allow_stretch=True, keep_ratio=False)
        self.backgroundImage3 = Img(source='images/backgroundgradient3.jpg', pos=(0, Window.height*1/8), size=(Window.width, Window.height*1/4), allow_stretch=True, keep_ratio=False)
        self.add_widget(self.backgroundImage1)
        self.add_widget(self.backgroundImage2)
        self.add_widget(self.backgroundImage3)
        Clock.schedule_once(lambda dt: self.create_interface(), 0)
        Clock.schedule_once(lambda dt: self.update_album(), 0.4)
        Clock.schedule_once(lambda dt: self.first(), 0)
        Clock.schedule_once(lambda dt: self.second(), 15)
        Clock.schedule_interval(self.update_device, 1)
        Clock.schedule_interval(self.update_progress, 1/30)
    logout_text = StringProperty("Logout")
    HRtext = StringProperty("80bpm")
    BPtext = StringProperty("None")
    my_text = StringProperty("1")
    imageURL = StringProperty("images/Queen.jpg")
    trackColor1 = StringProperty("100B11")
    trackColor2 = StringProperty("D7CED7")
    songTitle = StringProperty("Bohemian Rhapsody")
    songArtist = StringProperty("Queen")
    sliderProgress = StringProperty("0:00")
    sliderMax = StringProperty("3:14")
    songColor = ListProperty((1, 1, 1, 1))
    artistColor = ListProperty((.9, .9, .9, 1))
    progressMS = NumericProperty(0)
    my_Battery = StringProperty("100%")
    PowerLevel = StringProperty("images/Power100percent.jpg")
    batteryCharging = BooleanProperty(False)
    battery_index = NumericProperty(0)
    durationMS = NumericProperty(3000)
    opacityPrevious = NumericProperty(1)
    opacityPlay = NumericProperty(1)
    opacityNext = NumericProperty(1)
    deviceID = StringProperty("")
    deviceType = StringProperty("")
    auth_manager = None
    backgroundImage1 = "images/backgroundgradient1.jpg"
    backgroundImage2 = "images/backgroundgradient2.jpg"
    backgroundImage3 = "images/backgroundgradient3.jpg"

    ### STUFF FOR BLE
    macAddress = None
    device = None
    value_of_bpm = None
    value_of_bpress = None
    healthChar = None
    batteryVoltage = 4.2
    ### STUFF FOR BLE END

    # try:
    #    spotifyObject = spotipy.Spotify(auth_manager=auth_manager)
    # except:
    #    print("Not an object")
    #    pass
    def first(self):
        loop.create_task(main())
        print("here+ ", self.macAddress)
        #loop.create_task(connect(self.macAddress, loop))


    def second(self):
        loop.create_task(connect(self.macAddress, loop))
        self.logout_text = str(self.macAddress)
        #loop = asyncio.get_event_loop()
        #loop.run_until_complete(connect(self.macAddress, loop))

    def batteryLevel(self, y):
        x1 = (((14.697*(((843750 * (y**2)) - (5905000 * y) + 10359473)**(.5))) - (13500 * y) + 47240)**(1/3))
        x2 = (((7.3485 * (((843750 * (y**2)) - (5905000 * y) + 10359473)**(.5))) - (6750 * y) + 23620)**(1/3))
        x = 3.3333333 * (x1 - (144.45/x2) + 20)
        emptyBattery = 105.98060847412879
        percentUsed = (x / emptyBattery) * 100
        batteryLevel = 100 - percentUsed
        if batteryLevel < 0:
            batteryLevel = 0
        self.my_Battery = str(int(batteryLevel)) + "%"
        print(self.my_Battery)
        # if self.batteryCharging == True:
        #    self.PowerLevel = "images/PowerCharging.jpg"
        # elif
        if batteryLevel == 110:
            self.PowerLevel = "images/PowerCharging.jpg"
        elif batteryLevel > 91:
            self.PowerLevel = "images/Power100percent.jpg"
        elif batteryLevel > 64:
            self.PowerLevel = "images/Power75percent.jpg"
        elif batteryLevel > 37:
            self.PowerLevel = "images/Power50percent.jpg"
        elif batteryLevel > 10:
            self.PowerLevel = "images/Power25percent.jpg"
        elif batteryLevel >= 0:
            self.PowerLevel = "images/Power0percent.jpg"
        return

    # Dragging slider adjusts the track progress
    def sliderChange(self, newProgressMS):
        newProgressMS = int(newProgressMS)
        if self.progressMS != newProgressMS:
            spotifyObject = spotipy.Spotify(auth_manager=self.auth_manager)
            spotifyObject.seek_track(position_ms=newProgressMS)
            self.progressMS = newProgressMS
        Clock.schedule_interval(self.update_progress, 1 / 30)

    def sliderTouched(self):
        Clock.unschedule(self.update_progress)

    # Slider acts as progress bar
    def update_progress(self, dt):
        #try:
        spotifyObject = spotipy.Spotify(auth_manager=self.auth_manager)
        #spotifyObject = spotipy.Spotify(auth=token)
        try:
            track = spotifyObject.current_playback()
            self.imageURL = track['item']['album']['images'][0]['url']
        except:
            self.update_device(0)
            # print(self.deviceID)
            spotifyObject.transfer_playback(self.deviceID, force_play=True)
            track = spotifyObject.current_playback()
        self.progressMS = track['progress_ms']
        if track['item'] != None:
            self.durationMS = track['item']['duration_ms']
            self.songTitle = track['item']['name']
            self.songArtist = track['item']['artists'][0]['name']
            minutes = int(self.progressMS/60000)
            seconds = int((self.progressMS%60000)/1000)
            self.sliderProgress = "%i:%02i" % (minutes, seconds)
            minutes = int(self.durationMS/60000)
            seconds = int((self.durationMS%60000)/1000)
            self.sliderMax = "%i:%02i" % (minutes, seconds)
        #else:
        #    print("THIS WOULD HAVE SHUT IT DOWN SO IT'S FIXED1")
        #except:
        #print("nothing is playing")

        if self.progressMS < 2000:
            self.update_album()
            # print("Progress is less than 1000")

    def update_device(self, dt):
        try:
            self.HRtext = str(self.value_of_bpm)
            self.BPtext = str(self.value_of_bpress)
        except:
            pass
        self.batteryLevel(self.batteryVoltage)
        spotifyObject = spotipy.Spotify(auth_manager=self.auth_manager)
        #spotifyObject = spotipy.Spotify(auth=token)
        devices = spotifyObject.devices()
        # print("LENGTH OF DEVICES = ", len(devices['devices']))
        devices = devices["devices"]
        # print(json.dumps(devices, sort_keys=True, indent=4))

        for device in devices:
            if device["is_active"] == True:
                self.deviceID = device["id"]
                self.deviceType = device["name"]
                self.my_text = self.deviceType
                # print("Device ID is working")
                break
            else:
                self.deviceID = ""
        if self.deviceID == "":
            # print("DeviceID wasn't working")
            self.deviceID = "2ffdae0d996a4caa387100aca4d8e8e6cdc8dc4f"
            self.deviceType = "Default Speaker"
            self.my_text = self.deviceType

    def previous_track(self):
        #self.opacityChange("Previous")
        spotifyObject = spotipy.Spotify(auth_manager=self.auth_manager)
        #spotifyObject = spotipy.Spotify(auth=token)
        spotifyObject.previous_track()
        self.update_album()

    def next_track(self):
        #self.opacityChange("Next")
        spotifyObject = spotipy.Spotify(auth_manager=self.auth_manager)
        #spotifyObject = spotipy.Spotify(auth=token)
        spotifyObject.next_track()
        self.update_album()


    def play_pause(self):
        # self.opacityChange("Play")
        spotifyObject = spotipy.Spotify(auth_manager=self.auth_manager)
        #spotifyObject = spotipy.Spotify(auth=token)
        track = spotifyObject.current_user_playing_track()
        isPlaying = track["is_playing"]
        if isPlaying == True:
            spotifyObject.pause_playback()
        else:
            spotifyObject.start_playback(self.deviceID)

    def luminance(self, c):
        red = c[0]
        green = c[1]
        blue = c[2]
        return (.299 * red) + (.587 * green) + (.114 * blue)

    def opacityChange(self, buttonName):
        if buttonName == "Previous":
            if self.opacityPrevious == 1:
                self.opacityPrevious = 0.5
                Clock.schedule_once(lambda dt: self.opacityChange("Previous"), 1)
            else:
                self.opacityPrevious = 1
        if buttonName == "Play":
            if self.opacityPlay == 1:
                self.opacityPlay = 0.5
                Clock.schedule_once(lambda dt: self.opacityChange("Play"), 1)
            else:
                self.opacityPlay = 1
        if buttonName == "Next":
            if self.opacityNext == 1:
                self.opacityNext = 0.5
                Clock.schedule_once(lambda dt: self.opacityChange("Next"), 1)
            else:
                self.opacityNext = 1



    ## THIS CAN BE MADE A LOT FASTER
    def update_album(self):
        spotifyObject = spotipy.Spotify(auth_manager=self.auth_manager)
        # print(spotifyObject)
        # print("over here")
        #spotifyObject = spotipy.Spotify(auth=token)
        try:
            #print("IN HERE")
            track = spotifyObject.current_playback()
            #print(json.dumps(track, sort_keys=True, indent=4))
            self.imageURL = track['item']['album']['images'][0]['url']
        except:
            self.update_device(0)
            #print(self.deviceID)
            spotifyObject.transfer_playback(self.deviceID, force_play=True)
            track = spotifyObject.current_playback()
            if track['item'] != None:
                self.imageURL = track['item']['album']['images'][0]['url']


            #else:
            #    print("THIS WOULD HAVE SHUT IT DOWN SO IT'S FIXED2")
            # print(json.dumps(track, sort_keys=True, indent=4))

            # print(self.imageURL)

        try:
            fd = urlopen(self.imageURL)
            f = io.BytesIO(fd.read())
            color_thief = ColorThief(f)
            b = color_thief.get_palette(quality=10)
            b.append((255,255,255))
            C1 = b[0]
            C2 = b[1]
            black = (0, 0, 0)
            # self.trackColor1 = '%02x%02x%02x' % C1
            # self.trackColor2 = '%02x%02x%02x' % C2
            # self.songColor = (1, 1, 1)
            # self.artistColor = (.9, .9, .9)
        except:
            C1 = (255, 255, 255)
            C2 = (10, 10, 10)
            black = (0, 0, 0)
            # self.trackColor1 = "100B11"
            # self.trackColor2 = "D7CED7"

        try:
            os.remove("images/backgroundgradient1.jpg")
            os.remove("images/backgroundgradient2.jpg")
            os.remove("images/backgroundgradient3.jpg")
        except:
            print("no such file.")
        black = (0, 0, 0)

        array = self.get_gradient_3d(512, 256, black, C1, (False, False, False))
        Image.fromarray(np.uint8(array)).save('images/backgroundgradient1.jpg', quality=95)
        array = self.get_gradient_3d(512, 256, C1, C2, (False, False, False))
        Image.fromarray(np.uint8(array)).save('images/backgroundgradient2.jpg', quality=95)
        array = self.get_gradient_3d(512, 256, C2, black, (False, False, False))
        Image.fromarray(np.uint8(array)).save('images/backgroundgradient3.jpg', quality=95)

        self.backgroundImage1.reload()
        self.backgroundImage2.reload()
        self.backgroundImage3.reload()
        print("updating album")
        # lum1 = self.luminance(C1)
        # lum2 = self.luminance(C2)
        # if lum1 > lum2:
        #    lumsecond = lum2
        #    self.trackColor1 = '%02x%02x%02x' % C1
        #    self.trackColor2 = '%02x%02x%02x' % C2
        # else:
        #    lumsecond = lum1
        #    self.trackColor1 = '%02x%02x%02x' % C2
        #    self.trackColor2 = '%02x%02x%02x' % C1
        # if lumsecond > 255/2:
        #    self.songColor = (0, 0, 0)
        #    self.artistColor = (0, 0, 0, .9)
        # else:
        #    self.songColor = (1, 1, 1)
        #    self.artistColor = (.9, .9, .9)
        return

    def get_gradient_2d(self, start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    def get_gradient_3d(self, width, height, start_list, stop_list, is_horizontal_list):
        result = np.zeros((height, width, len(start_list)), dtype=float)

        for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
            result[:, :, i] = self.get_gradient_2d(start, stop, width, height, is_horizontal)

        return result

    def create_interface(self):
        # username = 'save username here
        username = 'insert username here'
        # sys.argv[1]
        scope = 'playlist-read-private user-read-playback-state user-modify-playback-state'
        SPOTIPY_CLIENT_ID = 'insert client id here'
        SPOTIPY_CLIENT_SECRET = 'insert client secret id here'
        SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback/'

        # Erase cache and prompt for user permission
        try:
            self.auth_manager = SpotifyOAuth(username=username, scope=scope, client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI)
            spotifyObject = spotipy.Spotify(auth_manager=auth_manager)
            # token = "insert token here"
            #token = util.prompt_for_user_token(username, scope, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI)
            #print("Token")
        except:
            # os.remove(f".cache-{username}")
            print("can't print token")
            #token = util.prompt_for_user_token(username, scope, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI)

    #def update_text_color(self, color1, color2, b):
    #    print()
    #    print()
    #    print()
    #    print("### IS THIS BEING EXECUTED")
    #    print()
    #    print()
    #    print()
    #    lum1 = self.luminance(color1)
    #    lum2 = self.luminance(color2)
    #    if lum1 > lum2:
    #        lumsecond = lum2
    #        self.trackColor1 = '%02x%02x%02x' % color1
    #        self.trackColor2 = '%02x%02x%02x' % color2
    #    else:
    #        lumsecond = lum1
    #        self.trackColor1 = '%02x%02x%02x' % color2
    #        self.trackColor2 = '%02x%02x%02x' % color1
    #    biggestDiff = 0
    #    for color in b:
    #        print(color)
    #        lumtest = self.luminance(color)
    #        lumDiff = abs(lumsecond - lumtest)
    #        if lumDiff > biggestDiff:
    #            biggestDiff = lumDiff
    #            textColor = color
    #    print(textColor)
    #    self.songColor = textColor
    #    return

        #b = tuple(bi / 255 for bi in b)

class cached_property(object):
    """Decorator that creates converts a method with a single
    self argument into a property cached on the instance.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type):
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res


class ColorThief(object):
    """Color thief main class."""
    def __init__(self, file):
        """Create one color thief for one image.
        :param file: A filename (string) or a file object. The file object
                     must implement `read()`, `seek()`, and `tell()` methods,
                     and be opened in binary mode.
        """
        self.image = Image.open(file)

    def get_color(self, quality=10):
        """Get the dominant color.
        :param quality: quality settings, 1 is the highest quality, the bigger
                        the number, the faster a color will be returned but
                        the greater the likelihood that it will not be the
                        visually most dominant color
        :return tuple: (r, g, b)
        """
        palette = self.get_palette(5, quality)
        return palette[0]

    def get_palette(self, color_count=10, quality=10):
        """Build a color palette.  We are using the median cut algorithm to
        cluster similar colors.
        :param color_count: the size of the palette, max number of colors
        :param quality: quality settings, 1 is the highest quality, the bigger
                        the number, the faster the palette generation, but the
                        greater the likelihood that colors will be missed.
        :return list: a list of tuple in the form (r, g, b)
        """
        image = self.image.convert('RGBA')
        width, height = image.size
        pixels = image.getdata()
        pixel_count = width * height
        valid_pixels = []
        for i in range(0, pixel_count, quality):
            r, g, b, a = pixels[i]
            # If pixel is mostly opaque and not white
            if a >= 125:
                if not (r > 250 and g > 250 and b > 250):
                    valid_pixels.append((r, g, b))

        # Send array to quantize function which clusters values
        # using median cut algorithm
        cmap = MMCQ.quantize(valid_pixels, color_count)
        return cmap.palette


class MMCQ(object):
    """Basic Python port of the MMCQ (modified median cut quantization)
    algorithm from the Leptonica library (http://www.leptonica.com/).
    """

    SIGBITS = 5
    RSHIFT = 8 - SIGBITS
    MAX_ITERATION = 1000
    FRACT_BY_POPULATIONS = 0.75

    @staticmethod
    def get_color_index(r, g, b):
        return (r << (2 * MMCQ.SIGBITS)) + (g << MMCQ.SIGBITS) + b

    @staticmethod
    def get_histo(pixels):
        """histo (1-d array, giving the number of pixels in each quantized
        region of color space)
        """
        histo = dict()
        for pixel in pixels:
            rval = pixel[0] >> MMCQ.RSHIFT
            gval = pixel[1] >> MMCQ.RSHIFT
            bval = pixel[2] >> MMCQ.RSHIFT
            index = MMCQ.get_color_index(rval, gval, bval)
            histo[index] = histo.setdefault(index, 0) + 1
        return histo

    @staticmethod
    def vbox_from_pixels(pixels, histo):
        rmin = 1000000
        rmax = 0
        gmin = 1000000
        gmax = 0
        bmin = 1000000
        bmax = 0
        for pixel in pixels:
            rval = pixel[0] >> MMCQ.RSHIFT
            gval = pixel[1] >> MMCQ.RSHIFT
            bval = pixel[2] >> MMCQ.RSHIFT
            rmin = min(rval, rmin)
            rmax = max(rval, rmax)
            gmin = min(gval, gmin)
            gmax = max(gval, gmax)
            bmin = min(bval, bmin)
            bmax = max(bval, bmax)
        return VBox(rmin, rmax, gmin, gmax, bmin, bmax, histo)

    @staticmethod
    def median_cut_apply(histo, vbox):
        if not vbox.count:
            return (None, None)

        rw = vbox.r2 - vbox.r1 + 1
        gw = vbox.g2 - vbox.g1 + 1
        bw = vbox.b2 - vbox.b1 + 1
        maxw = max([rw, gw, bw])
        # only one pixel, no split
        if vbox.count == 1:
            return (vbox.copy, None)
        # Find the partial sum arrays along the selected axis.
        total = 0
        sum_ = 0
        partialsum = {}
        lookaheadsum = {}
        do_cut_color = None
        if maxw == rw:
            do_cut_color = 'r'
            for i in range(vbox.r1, vbox.r2+1):
                sum_ = 0
                for j in range(vbox.g1, vbox.g2+1):
                    for k in range(vbox.b1, vbox.b2+1):
                        index = MMCQ.get_color_index(i, j, k)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        elif maxw == gw:
            do_cut_color = 'g'
            for i in range(vbox.g1, vbox.g2+1):
                sum_ = 0
                for j in range(vbox.r1, vbox.r2+1):
                    for k in range(vbox.b1, vbox.b2+1):
                        index = MMCQ.get_color_index(j, i, k)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        else:  # maxw == bw
            do_cut_color = 'b'
            for i in range(vbox.b1, vbox.b2+1):
                sum_ = 0
                for j in range(vbox.r1, vbox.r2+1):
                    for k in range(vbox.g1, vbox.g2+1):
                        index = MMCQ.get_color_index(j, k, i)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        for i, d in partialsum.items():
            lookaheadsum[i] = total - d

        # determine the cut planes
        dim1 = do_cut_color + '1'
        dim2 = do_cut_color + '2'
        dim1_val = getattr(vbox, dim1)
        dim2_val = getattr(vbox, dim2)
        for i in range(dim1_val, dim2_val+1):
            if partialsum[i] > (total / 2):
                vbox1 = vbox.copy
                vbox2 = vbox.copy
                left = i - dim1_val
                right = dim2_val - i
                if left <= right:
                    d2 = min([dim2_val - 1, int(i + right / 2)])
                else:
                    d2 = max([dim1_val, int(i - 1 - left / 2)])
                # avoid 0-count boxes
                while not partialsum.get(d2, False):
                    d2 += 1
                count2 = lookaheadsum.get(d2)
                while not count2 and partialsum.get(d2-1, False):
                    d2 -= 1
                    count2 = lookaheadsum.get(d2)
                # set dimensions
                setattr(vbox1, dim2, d2)
                setattr(vbox2, dim1, getattr(vbox1, dim2) + 1)
                return (vbox1, vbox2)
        return (None, None)

    @staticmethod
    def quantize(pixels, max_color):
        """Quantize.
        :param pixels: a list of pixel in the form (r, g, b)
        :param max_color: max number of colors
        """
        if not pixels:
            raise Exception('Empty pixels when quantize.')
        if max_color < 2 or max_color > 256:
            raise Exception('Wrong number of max colors when quantize.')

        histo = MMCQ.get_histo(pixels)

        # check that we aren't below maxcolors already
        if len(histo) <= max_color:
            # generate the new colors from the histo and return
            pass

        # get the beginning vbox from the colors
        vbox = MMCQ.vbox_from_pixels(pixels, histo)
        pq = PQueue(lambda x: x.count)
        pq.push(vbox)

        # inner function to do the iteration
        def iter_(lh, target):
            n_color = 1
            n_iter = 0
            while n_iter < MMCQ.MAX_ITERATION:
                vbox = lh.pop()
                if not vbox.count:  # just put it back
                    lh.push(vbox)
                    n_iter += 1
                    continue
                # do the cut
                vbox1, vbox2 = MMCQ.median_cut_apply(histo, vbox)
                if not vbox1:
                    raise Exception("vbox1 not defined; shouldn't happen!")
                lh.push(vbox1)
                if vbox2:  # vbox2 can be null
                    lh.push(vbox2)
                    n_color += 1
                if n_color >= target:
                    return
                if n_iter > MMCQ.MAX_ITERATION:
                    return
                n_iter += 1

        # first set of colors, sorted by population
        iter_(pq, MMCQ.FRACT_BY_POPULATIONS * max_color)

        # Re-sort by the product of pixel occupancy times the size in
        # color space.
        pq2 = PQueue(lambda x: x.count * x.volume)
        while pq.size():
            pq2.push(pq.pop())

        # next set - generate the median cuts using the (npix * vol) sorting.
        iter_(pq2, max_color - pq2.size())

        # calculate the actual colors
        cmap = CMap()
        while pq2.size():
            cmap.push(pq2.pop())
        return cmap


class VBox(object):
    """3d color space box"""
    def __init__(self, r1, r2, g1, g2, b1, b2, histo):
        self.r1 = r1
        self.r2 = r2
        self.g1 = g1
        self.g2 = g2
        self.b1 = b1
        self.b2 = b2
        self.histo = histo

    @cached_property
    def volume(self):
        sub_r = self.r2 - self.r1
        sub_g = self.g2 - self.g1
        sub_b = self.b2 - self.b1
        return (sub_r + 1) * (sub_g + 1) * (sub_b + 1)

    @property
    def copy(self):
        return VBox(self.r1, self.r2, self.g1, self.g2,
                    self.b1, self.b2, self.histo)

    @cached_property
    def avg(self):
        ntot = 0
        mult = 1 << (8 - MMCQ.SIGBITS)
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for i in range(self.r1, self.r2 + 1):
            for j in range(self.g1, self.g2 + 1):
                for k in range(self.b1, self.b2 + 1):
                    histoindex = MMCQ.get_color_index(i, j, k)
                    hval = self.histo.get(histoindex, 0)
                    ntot += hval
                    r_sum += hval * (i + 0.5) * mult
                    g_sum += hval * (j + 0.5) * mult
                    b_sum += hval * (k + 0.5) * mult

        if ntot:
            r_avg = int(r_sum / ntot)
            g_avg = int(g_sum / ntot)
            b_avg = int(b_sum / ntot)
        else:
            r_avg = int(mult * (self.r1 + self.r2 + 1) / 2)
            g_avg = int(mult * (self.g1 + self.g2 + 1) / 2)
            b_avg = int(mult * (self.b1 + self.b2 + 1) / 2)

        return r_avg, g_avg, b_avg

    def contains(self, pixel):
        rval = pixel[0] >> MMCQ.RSHIFT
        gval = pixel[1] >> MMCQ.RSHIFT
        bval = pixel[2] >> MMCQ.RSHIFT
        return all([
            rval >= self.r1,
            rval <= self.r2,
            gval >= self.g1,
            gval <= self.g2,
            bval >= self.b1,
            bval <= self.b2,
        ])

    @cached_property
    def count(self):
        npix = 0
        for i in range(self.r1, self.r2 + 1):
            for j in range(self.g1, self.g2 + 1):
                for k in range(self.b1, self.b2 + 1):
                    index = MMCQ.get_color_index(i, j, k)
                    npix += self.histo.get(index, 0)
        return npix


class CMap(object):
    """Color map"""
    def __init__(self):
        self.vboxes = PQueue(lambda x: x['vbox'].count * x['vbox'].volume)

    @property
    def palette(self):
        return self.vboxes.map(lambda x: x['color'])

    def push(self, vbox):
        self.vboxes.push({
            'vbox': vbox,
            'color': vbox.avg,
        })

    def size(self):
        return self.vboxes.size()

    def nearest(self, color):
        d1 = None
        p_color = None
        for i in range(self.vboxes.size()):
            vbox = self.vboxes.peek(i)
            d2 = math.sqrt(
                math.pow(color[0] - vbox['color'][0], 2) +
                math.pow(color[1] - vbox['color'][1], 2) +
                math.pow(color[2] - vbox['color'][2], 2)
            )
            if d1 is None or d2 < d1:
                d1 = d2
                p_color = vbox['color']
        return p_color

    def map(self, color):
        for i in range(self.vboxes.size()):
            vbox = self.vboxes.peek(i)
            if vbox['vbox'].contains(color):
                return vbox['color']
        return self.nearest(color)


class PQueue(object):
    """Simple priority queue."""
    def __init__(self, sort_key):
        self.sort_key = sort_key
        self.contents = []
        self._sorted = False

    def sort(self):
        self.contents.sort(key=self.sort_key)
        self._sorted = True

    def push(self, o):
        self.contents.append(o)
        self._sorted = False

    def peek(self, index=None):
        if not self._sorted:
            self.sort()
        if index is None:
            index = len(self.contents) - 1
        return self.contents[index]

    def pop(self):
        if not self._sorted:
            self.sort()
        return self.contents.pop()

    def size(self):
        return len(self.contents)

    def map(self, f):
        return list(map(f, self.contents))



class jamlabsApp(App):
    pass


async def main():
    if MainWidget.device == None:
        devices = await BleakScanner.discover()
        for device in devices:
            if str(device.name) == "MyLoopHealth":
                MainWidget.device = device
                MainWidget.macAddress = MainWidget.device.address
                MainWidget.logout_text = MainWidget.macAddress
                print("DEVICE IS ", MainWidget.device)
    MainWidget.second(MainWidget)

from bleak import BleakClient

async def connect(address, loop):
    print(str(address))
    async with BleakClient(str(address), loop=loop) as client:
        print("in async with BleakClient")
        services = await client.get_services()
        for ser in services:
            if str(ser.uuid) == "000000ff-0000-1000-8000-00805f9b34fb":
                print("did we find it?")
                for characteristic in ser.characteristics:
                    print(characteristic.uuid)
                    if str(characteristic.handle) == "41":
                        MainWidget.healthChar = characteristic
                        #value = bytes(await client.read_gatt_char(characteristic.uuid))
                        #print(value)
    loop.create_task(continuous(address, loop))

async def continuous(address, loop):
    async with BleakClient(str(address)) as client:
        battVoltage = [0, 0, 0, 0]
        HR = [0, 0, 0]
        BP = [0, 0, 0, 0, 0, 0]
        heartRateData = ""
        while True:
            try:
                MainWidget.value_of_char = bytes(await client.read_gatt_char(MainWidget.healthChar.uuid))
                print(MainWidget.value_of_char)
                heartRateData = str(MainWidget.value_of_char)

                try:
                    battVoltage[0] = int(heartRateData[5])
                    battVoltage[1] = int(heartRateData[9])
                    battVoltage[2] = int(heartRateData[13])
                    battVoltage[3] = int(heartRateData[21])
                except:
                    print("Incorrect number of characteristic values present in batteryVoltage.")

                try:
                    HR[0] = str(heartRateData[25])
                    HR[1] = str(heartRateData[29])
                    HR[2] = str(heartRateData[33])
                except:
                    print("Incorrect number of characteristic values present in HR.")

                try:
                    BP[0] = str(heartRateData[37])
                    BP[1] = str(heartRateData[41])
                    BP[2] = str(heartRateData[45])
                    BP[3] = str(heartRateData[49])
                    BP[4] = str(heartRateData[53])
                    BP[5] = str(heartRateData[58])
                except:
                    print("Incorrect number of characteristic values present in BP.")

                validVoltage = True
                validBPM = True
                validBP = True



                # For loops to determine if unusable characters are present
                for k in battVoltage:
                    k = str(k)
                    if k.isalpha():
                        validVoltage = False

                for i in HR:
                    i = str(i)
                    if i.isalpha():
                        validBPM = False
                for j in BP:
                    j = str(j)
                    if j.isalpha():
                        validBP = False
                if validVoltage:
                    print("battVoltage ", battVoltage)
                    MainWidget.batteryVoltage = battVoltage[0] + float(battVoltage[1] * 0.1) + float(battVoltage[2] * 0.01) + float(battVoltage[3] * 0.001)
                    MainWidget.batteryVoltage *= 3.7
                    print("VOLTAGE IS ", MainWidget.batteryVoltage)

                # If the BPM and BP contain only valid characters, print the BPM and BP
                if validBPM:
                    if int(HR[0]) != 0:
                        MainWidget.value_of_bpm = HR[0] + HR[1] + HR[2] + "bpm"
                    else:
                        MainWidget.value_of_bpm = HR[1] + HR[2] + "bpm"

                if validBP and validBPM:
                    if int(BP[3]) > 0 and int(BP[3]) < 10:
                        MainWidget.value_of_bpress = BP[0] + BP[1] + BP[2] + "/" + BP[3] + BP[4] + BP[5]
                    else:
                        MainWidget.value_of_bpress = BP[0] + BP[1] + BP[2] + "/" + BP[4] + BP[5]


                #asyncio.sleep(1)
            except:
                bpmDots = "."
                for i in range(5):
                    await asyncio.sleep(1)
                    MainWidget.value_of_bpm = bpmDots
                    MainWidget.value_of_bpress = bpmDots
                    bpmDots += "."
                print("Reconnecting to device")
                loop.create_task(connect(address, loop))
                loop.create_task(continuous(address, loop))

loop = asyncio.get_event_loop()
loop.run_until_complete(jamlabsApp().async_run())
loop.close()
#jamlabsApp().run()
