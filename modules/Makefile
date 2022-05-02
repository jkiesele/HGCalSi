

all: libalpha.so _test_alpha

libalpha.so: alpha_intersect.cpp
	g++ -shared -fPIC -o $@ $^

_test_alpha: alpha_intersect.cpp
	g++  -o $@ $^
