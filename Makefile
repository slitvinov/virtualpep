CC = gcc
CFLAGS = -O3 -fopenmp
LDLIBS = -lm
NVCC = nvcc
NVFLAGS = -O3 -arch=sm_90

all: sdpd2 sdpd3 sdpd2w sdpd3w
sdpd2w: sdpd2.c; $(CC) $(CFLAGS) -DWENDLAND $< -o $@ $(LDLIBS)
sdpd3w: sdpd3.c; $(CC) $(CFLAGS) -DWENDLAND $< -o $@ $(LDLIBS)
sdpd3cu: sdpd3.cu; $(NVCC) $(NVFLAGS) $< -o $@
clean:; rm -f sdpd2 sdpd3 sdpd2w sdpd3w sdpd3cu
