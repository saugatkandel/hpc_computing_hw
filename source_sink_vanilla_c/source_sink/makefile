source_sink: main.o funcs.o
	gcc -o source_sink main.o funcs.o -lm
main.o: main.c
	gcc -c -std=c99 main.c
funcs.o: funcs.c funcs.h
	gcc -c -std=c99 funcs.c
run:
	python run.py

clean:
	rm source_sink *.o
