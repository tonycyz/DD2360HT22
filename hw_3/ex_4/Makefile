# From https://x.momo86.net/?p=29

CXX=g++
CXXFLAGS=-std=c++11 -O3 -g -Xcompiler -Wall

NVCC=nvcc
NVCCFLAGS= -std=c++11 -O3 -g -Xcompiler -Wall --compiler-bindir=$(CXX)

SRCDIR=.
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')

OBJDIR=.
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))

BIN := .
TARGET=sputniPIC.out

all: dir $(BIN)/$(TARGET)

dir: ${BIN}
  
${BIN}:
	mkdir -p $(BIN)

$(BIN)/$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
