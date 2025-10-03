# config.py
IMAGES_GLOB = "/Users/piaoruilin/Desktop/camera-calib/checkerboard*.jpg"   # put your checkerboard photos here
# Your board looks like 10x7 squares -> inner corners = (9, 6)
PATTERN_SIZE = (9, 6)      # (cols, rows) inner-corner count
SQUARE_SIZE_MM = 20.0      # change if your square edge is different
SUBPIX_WINDOW = (11, 11)
