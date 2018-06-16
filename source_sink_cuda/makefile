CC = nvcc
CFLAGS = #-DVERBOSE -g
LIBS = -lm

all: source_sink

HDRS = funcs.h handleerror.h
SRCS = main.cu funcs.cu
OBJS = $(SRCS:%.cu=%.o)


$(OBJS): %.o: %.cu $(HDRS)
	$(CC) $(CFLAGS)  -c $< 

source_sink: $(OBJS)
	$(CC)  $(CFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	-rm *.o source_sink* core.* *.out
