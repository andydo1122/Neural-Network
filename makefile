# Makefile
CC = g++

CFLAGS = -g -Wall -c

TARGET = neuralNetTutorial

all : 
	g++ -o neuralNetTutorial neuralNetTutorial.cpp

#$(TARGET): $(TARGET).cpp	
#			$(CC) $(CFLAGS) $(TARGET).o -o $(TARGET) $(TARGET).cpp 

clean:
	rm -r *.o $(TARGET)