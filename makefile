# Makefile
CC = g++

CFLAGS = -g -Wall -c

TARGET = neuralNetTutorial

all : $(TARGET)

$(TARGET): $(TARGET).cpp	
			$(CC) $(CFLAGS) $(TARGET).o -o $(TARGET) $(TARGET).cpp 

clean:
	rm -r *.o $(TARGET)