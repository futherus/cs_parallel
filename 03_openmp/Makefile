SRC:=life.cc
OBJ:=$(SRC:.cc=.o)
TARGET:=life
CXX:=g++
FLAGS:=$(CXXFLAGS) -std=c++20 -O3 -Wall -Wextra -lsfml-graphics -lsfml-window -lsfml-system -fopenmp # -DNDEBUG # -fsanitize=address

all: $(TARGET)

%.o: %.cc
	$(CXX) -c $^ -o $@ $(FLAGS)

$(TARGET): $(OBJ)
	$(CXX) $^ -o $@ $(FLAGS)

clean:
	rm -rf $(OBJ) $(TARGET)

.PHONY: all clean

