PROG = test_domain
SRCS = domain.c  gf.c  test_domain.c  timer.c
OBJS =domain.o  gf.o  test_domain.o  timer.o
HDRS =  domain.h   gf.h   timer.h
%.o: %.c  $(HDRS)
	mpicc  -c -Wall -O3 -ffast-math  -g  $<

all: $(PROG)

$(PROG): $(OBJS)
	mpicc -g $(OBJS) -o $@ -lm

clean:
	-rm $(PROG) $(OBJS)
