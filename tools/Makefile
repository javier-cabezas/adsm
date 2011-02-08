.PHONY: all

all: libmylib.so main

libmylib.so: shared.cpp
	gcc -shared -Wl,-soname,$@.0 -o $@.0 shared.cpp -fPIC
	ln -s $@.0 $@

main: main.cpp kernel.cl.o
	g++ -o $@ $^ -L. -lmylib

kernel.cl.o: kernel.cl kernel2.cl
	./gmac-compile-cl $@ $^

clean:
	rm -f *.o
	rm -f main
	rm -f *.so*
