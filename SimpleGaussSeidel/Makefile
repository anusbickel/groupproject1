PROGS = gs_nonblocking gs_blocking
all: $(PROGS)

%: %.c
	mpicc -Wall -O3 -ffast-math -lm  $< -o $@

clean:
	-rm $(PROGS)
