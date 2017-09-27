CC=$(shell brew --prefix gcc)/bin/gcc-7
CXX=$(shell brew --prefix gcc)/bin/g++-7
RM=rm -f
CPPFLAGS=-g -pthread
LDFLAGS=-g
LDLIBS= $(shell pkg-config opencv --cflags --libs)

SRCS=display_image.cpp
OBJS=$(subst .cpp,.o, $(SRCS))

all: display_image

display_image:$(OBJS)
	$(CXX) $(LDFLAGS) -o display_image $(OBJS) $(LDLIBS) 

display_image.o: display_image.cpp

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) display_image
