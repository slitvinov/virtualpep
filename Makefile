CC = gcc
CFLAGS = -O3 -fopenmp
LDLIBS = -lm
NVCC = nvcc
NVFLAGS = -O3 -arch=sm_90 -fmad=false

all: sdpd2 sdpd3 sdpd2l sdpd3l
sdpd2l: sdpd2.c; $(CC) $(CFLAGS) -DLUCY $< -o $@ $(LDLIBS)
sdpd3l: sdpd3.c; $(CC) $(CFLAGS) -DLUCY $< -o $@ $(LDLIBS)
sdpd3cu: sdpd3.cu; $(NVCC) $(NVFLAGS) $< -o $@
sdpd3b: sdpd3b.cu; $(NVCC) $(NVFLAGS) $< -o $@
clean:; rm -f sdpd2 sdpd3 sdpd2l sdpd3l sdpd3cu sdpd3b
