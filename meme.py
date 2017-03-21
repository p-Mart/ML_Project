import sys,random

t = 0
while(True):
	t = t + 1
	sys.stdout.write(chr(
		(random.randint(0,255)|&t*2&t>>3|t*5&t>>7|t*9&t>>11|t*13&t>>17|t*19&t>>23|t*31&t>>29)%256))