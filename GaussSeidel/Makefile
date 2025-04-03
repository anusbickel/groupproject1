PROG = solver2
SRCS = domain.c  gf.c   timer.c solver2.c
OBJS =domain.o  gf.o   timer.o solver2.o
HDRS =  domain.h   gf.h   timer.h
%.o: %.c  $(HDRS)
	mpicc  -c -Wall -O3 -ffast-math  -g  $<

all: $(PROG)

$(PROG): $(OBJS)
	mpicc -g $(OBJS) -o $@ -lm

clean:
	-rm $(PROG) $(OBJS)
