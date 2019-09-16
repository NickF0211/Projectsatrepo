MROOT := $(shell pwd)
export MROOT

full:
	make -C core rs

full-debug:
	make -C core d

simp_build:
	make -C simp rs

debug:
	make -C simp d 

clean:
	make -C simp clean
	make -C core clean

