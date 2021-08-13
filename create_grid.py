import sys
import glob
from PIL import Image

pattern = sys.argv[1]
rows = int(sys.argv[2])
cols = int(sys.argv[3])
filenames = glob.glob(pattern)

images = [Image.open(name).resize((224, 224)) for name in filenames]

new_image = Image.new('RGB', (cols*224, rows*224))

i = 0
for y in range(rows):
    if i >= len(images):
        break
    y *= 224
    for x in range(cols):
        x *= 224
        img = images[i]
        new_image.paste(img, (x, y, x+224, y+224))
        print('paste:', x, y)
        i += 1

new_image.save('output.jpg')
