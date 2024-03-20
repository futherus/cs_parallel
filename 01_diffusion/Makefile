SRC:=main.cc
OBJ:=$(SRC:.cc=.o)
TARGET:=test
CXX:=mpic++
FLAGS:=$(CXXFLAGS) -Wall -Wextra #-fsanitize=address

all: $(TARGET)

%.o: %.cc
	$(CXX) -c $^ -o $@ $(FLAGS)

$(TARGET): $(OBJ)
	$(CXX) $^ -o $@ $(FLAGS)

clean:
	rm -rf $(OBJ) $(TARGET)

.PHONY: all clean

