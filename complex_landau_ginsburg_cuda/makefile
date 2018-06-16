CC = nvcc
CFLAGS = -DVERBOSE -g
LIBS = -lcufft -lcurand

all: cgl

HDRS = 
SRCS = cgl.cu 
OBJS = $(SRCS:%.cu=%.o)


$(OBJS): %.o: %.cu $(HDRS)
	$(CC) $(CFLAGS)  -c $< 

cgl: $(OBJS)
	$(CC)  $(CFLAGS) -o $@ $(OBJS) $(LIBS)

# housekeeping
clean:
	rm -f core.* cgl *.o *.out *.png
