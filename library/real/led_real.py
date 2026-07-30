"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: led_real.py
File Description: Contains the LED module of the drone_core library. Drives a
WS2812 strip on the Raspberry Pi 5 over SPI via the rpi5-ws2812 library.
"""

from led import Led

# The strip is wired to SPI bus 0, device 0 (/dev/spidev0.0). LED_COUNT may be
# larger than the physical strip; extra pixels are simply ignored.
SPI_BUS = 0
SPI_DEVICE = 0
LED_COUNT = 100


class LedReal(Led):
    def __init__(self):
        self.__strip = None
        try:
            from rpi5_ws2812.ws2812 import WS2812SpiDriver
            self.__strip = WS2812SpiDriver(
                spi_bus=SPI_BUS, spi_device=SPI_DEVICE, led_count=LED_COUNT
            ).get_strip()
        except Exception as e:
            print(f"LED strip initialization failed. Reason: {e}")

    def fill(self, red: int, green: int, blue: int) -> None:
        if self.__strip is None:
            return
        from rpi5_ws2812.ws2812 import Color
        self.__strip.set_all_pixels(Color(int(red), int(green), int(blue)))
        self.__strip.show()
