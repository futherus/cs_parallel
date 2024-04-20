SRC:=main.cc
OBJ:=$(SRC:.cc=.o)
TARGET:=int
CXX:=g++
FLAGS:=$(CXXFLAGS) -O3 -Wall -Wextra -DNDEBUG # -fsanitize=address

all: $(TARGET)

%.o: %.cc
	$(CXX) -c $^ -o $@ $(FLAGS)

$(TARGET): $(OBJ)
	$(CXX) $^ -o $@ $(FLAGS)

clean:
	rm -rf $(OBJ) $(TARGET)

.PHONY: all clean

