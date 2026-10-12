// ESP-IDF binding for an ST7789-class panel on a plain 4-wire SPI bus, for a
// composed board (`chips.display.interface: "spi"`). It is a whole Display,
// like rgb_panel.cpp: the base's QSPI display.cpp stays out of the build.
//
// The pipeline itself (PSRAM framebuffer, staged DMA bands, rotate-on-flush
// landscape, LEDC backlight) lives with the M5Stack StickS3 target, and the
// Waveshare Touch LCD 1.54 already shares it the same way. It reads the
// board's St7789DisplayConfig from board.h and the GEA_ST7789_* definitions
// the composer writes into target.cmake: clock, SPI mode, colour order,
// inversion and the GRAM gaps. A reset or backlight line the board does not
// route is GPIO_NUM_NC and is skipped.
#include "../../../esp32-s3-m5stack-sticks3/main/display.cpp"
