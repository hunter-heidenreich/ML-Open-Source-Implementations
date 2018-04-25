from PIL import Image
from os import listdir
from os.path import join

dir_name = '/home/hunter/git/ML-Open-Source-Implementations/Generative-Adversarial-Networks/data/pokemon-gen5'
dir_name_final = '/home/hunter/git/ML-Open-Source-Implementations/Generative-Adversarial-Networks/data/pokemon-gen5-sized'

for f in listdir(dir_name):
	im = Image.open(join(dir_name, f))
	# width, height = im.size

	# n_x = (64 - width) // 2
	# n_y = (64 - height) // 2
	im = im.resize((64, 64), Image.ANTIALIAS)
	# resize = Image.new('RGB', (64, 64), (255, 255, 255))
	# resize.paste(im, (n_x, n_y))
	im.save(join(dir_name_final, f), 'PNG')
