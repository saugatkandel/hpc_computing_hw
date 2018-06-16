PARALLEL_CC = mpicc
CFLAGS = -std=c99 -Wall -Werror -ffast-math -D_XOPEN_SOURCE #-DVERBOSE
LIBS = -lm
LIBS_PARALLEL = -lmpi

all: duffing_parallel

HDRS_PARALLEL = funcs.h
SRCS_PARALLEL = duffing_parallel.c funcs.o
OBJS_PARALLEL = $(SRCS_PARALLEL:%.c=%.o)

$(OBJS_PARALLEL): %.o: %.c $(HDRS_PARALLEL)
	$(PARALLEL_CC) $(CFLAGS) -c $< 

duffing_parallel: $(OBJS_PARALLEL)
	$(PARALLEL_CC)  $(CFLAGS) -o $@ $(OBJS_PARALLEL) $(LIBS) $(LIBS_PARALLEL)

run:
	msub run_plot_probs.sh
	msub run_plot_mn_runtime.sh
	msub run_plot_strong_scaling.sh
	msub run_plot_weak_scaling.sh

clean:
	-rm *.o* *.e* duffing_parallel *.png *.dat
