CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra
LDFLAGS := -lcuda -lSDL2 -lGL -ldl -lpthread

TARGET := cuda_sphere

all: $(TARGET)

$(TARGET): main.cpp sphere_kernel.ptx
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
