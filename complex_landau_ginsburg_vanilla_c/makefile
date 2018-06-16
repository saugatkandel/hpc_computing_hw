CFLAGS = -std=c99 -Wall -Werror -O3
LIBS = -lm -lfftw3

OBJS_DEFAULT = cgl.o funcs.o

# executables
all : cgl

cgl.o: cgl.c
	gcc $(CFLAGS) -o $@ -c cgl.c

funcs.o: funcs.c funcs.h
	gcc $(CFLAGS) -c funcs.c

cgl: $(OBJS_DEFAULT)
	gcc -o $@ $(OBJS_DEFAULT) $(LIBS)

# generate plots of final absolute value and phase after T=10000
plot_default:
	./cgl 128 1.5 0.25 30000
	python plot_final_abs_phase.py

# use custom value of T to generate plot of dimension size vs runtime
# default value of T takes too long
plot_dim_times:
	python plot_dimsize_runtimes.py

# housekeeping
clean:
	rm -f core.* cgl *.o
