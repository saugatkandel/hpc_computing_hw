PARALLEL_CC = mpicc
CFLAGS = -std=c99 -Wall -Werror -D_XOPEN_SOURCE -Wno-error=unused-but-set-variable
LIBS = -lm -llapacke -llapack -L/software/lapack/3.6.0_gcc/lib64 -lgslcblas
LIBS_PARALLEL = -lmpi
INC = -I/software/lapack/3.6.0_gcc/include
all: euler2d_parallel

HDRS = funcs.h
SRCS = euler2d.c funcs.c
OBJS = $(SRCS:%.c=%.o)


$(OBJS): %.o: %.c $(HDRS)
	$(PARALLEL_CC) $(CFLAGS) $(INC) -c $< 

euler2d_parallel: $(OBJS)
	$(PARALLEL_CC)  $(CFLAGS) -o $@ $(OBJS) $(LIBS) $(LIBS_PARALLEL)

clean:
	-rm *.o* *.e* euler2d_parallel core.* *.png *.dat
run:
	msub run_plot_contours.sh
	msub run_plot_n_runtime.sh
	msub run_plot_strong_scaling.sh
	msub run_plot_weak_scaling.sh
