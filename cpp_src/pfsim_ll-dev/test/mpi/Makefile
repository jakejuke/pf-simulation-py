CC:=mpic++
RM:=rm

LIBFLAGS:= -I./../../
STD:= -std=c++17
CFLAGS:= -Wall -g $(STD) $(LIBFLAGS)

.PHONY: clean 


clean: 
	$(RM) a.out 

%: %.cpp
	$(CC) $(CFLAGS) -o $@ $^
