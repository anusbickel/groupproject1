PROG = solver
SRCS = domain.c  gf.c  solver.c  timer.c
OBJS =domain.o  gf.o  solver.o  timer.o
HDRS =  domain.h   gf.h   timer.h
%.o: %.c  $(HDRS)
	mpicc  -c -Wall -O3 -ffast-math  -g  $<

all: $(PROG)

$(PROG): $(OBJS)
	mpicc -g $(OBJS) -o $@ -lm

clean:
	-rm $(PROG) $(OBJS)
